import Foundation

struct RuntimeError: Error, CustomStringConvertible {
    let message: String
    init(_ message: String) { self.message = message }
    var description: String { message }
}

struct CLIOptions {
    var manifestPath: String
    var outputDir: String
    var audioRoot: String
    var limit: Int?
    var delaySeconds: Double
    var localeIdentifier: String
}

func printUsageAndExit() -> Never {
    let usage = """
    Usage: CreakASR --manifest <path> --output <dir> [options]

    Required:
      --manifest <path>    Manifest CSV with utt_id, wav_path columns.
      --output <dir>       Output directory; writes hypotheses.csv and
                            failures.log here (created if missing).

    Options:
      --audio-root <dir>   Base directory to resolve relative wav_path
                            values against. Default: current directory.
      --limit <N>          Only process the first N manifest rows.
      --delay <seconds>    Sleep this long between files. Default: 0.
      --locale <id>        BCP-47 locale for recognition. Default: en-US.
    """
    print(usage)
    exit(1)
}

func parseArguments(_ args: [String]) -> CLIOptions {
    var manifestPath: String?
    var outputDir: String?
    var audioRoot = FileManager.default.currentDirectoryPath
    var limit: Int?
    var delaySeconds = 0.0
    var localeIdentifier = "en-US"

    var i = 0
    while i < args.count {
        let arg = args[i]
        func nextValue() -> String {
            guard i + 1 < args.count else { printUsageAndExit() }
            i += 1
            return args[i]
        }
        switch arg {
        case "--manifest": manifestPath = nextValue()
        case "--output": outputDir = nextValue()
        case "--audio-root": audioRoot = nextValue()
        case "--limit":
            guard let n = Int(nextValue()), n > 0 else { printUsageAndExit() }
            limit = n
        case "--delay":
            guard let d = Double(nextValue()), d >= 0 else { printUsageAndExit() }
            delaySeconds = d
        case "--locale":
            localeIdentifier = nextValue()
        case "--help", "-h":
            printUsageAndExit()
        default:
            FileHandle.standardError.write(Data("unrecognized argument: \(arg)\n".utf8))
            printUsageAndExit()
        }
        i += 1
    }

    guard let manifestPath, let outputDir else { printUsageAndExit() }
    return CLIOptions(manifestPath: manifestPath, outputDir: outputDir, audioRoot: audioRoot,
                       limit: limit, delaySeconds: delaySeconds, localeIdentifier: localeIdentifier)
}

func currentOSVersionString() -> String {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
}

func isoTimestamp() -> String {
    ISO8601DateFormatter().string(from: Date())
}

/// Retries `operation` with exponential backoff. SFSpeechRecognizer can
/// throttle under sustained batch use and its failure modes here aren't
/// precisely documented, so this treats any error as potentially
/// transient rather than trying to special-case error codes.
func withRetry<T>(
    maxAttempts: Int,
    initialDelaySeconds: Double,
    maxDelaySeconds: Double,
    onRetry: (Int, Error) -> Void,
    operation: () async throws -> T
) async throws -> T {
    var attempt = 1
    var delaySeconds = initialDelaySeconds
    while true {
        do {
            return try await operation()
        } catch {
            if attempt >= maxAttempts { throw error }
            onRetry(attempt, error)
            try? await Task.sleep(nanoseconds: UInt64(delaySeconds * 1_000_000_000))
            delaySeconds = min(delaySeconds * 2, maxDelaySeconds)
            attempt += 1
        }
    }
}

func run(options: CLIOptions) async throws {
    var manifestRows = try Manifest.read(path: options.manifestPath)
    if let limit = options.limit {
        manifestRows = Array(manifestRows.prefix(limit))
    }
    guard !manifestRows.isEmpty else {
        print("manifest has no rows to process after applying --limit")
        return
    }

    let fm = FileManager.default
    try fm.createDirectory(atPath: options.outputDir, withIntermediateDirectories: true)
    let hypothesesPath = (options.outputDir as NSString).appendingPathComponent("hypotheses.csv")
    let failuresPath = (options.outputDir as NSString).appendingPathComponent("failures.log")

    // Checkpoint/resume: anything already in hypotheses.csv from a prior
    // (possibly crashed) run is skipped rather than re-transcribed.
    let alreadyDone = try Manifest.completedUttIDs(hypothesesPath: hypothesesPath)
    if !fm.fileExists(atPath: hypothesesPath) {
        let header = CSV.row(["utt_id", "hypothesis", "os_version", "recognizer", "elapsed_ms"]) + "\n"
        guard fm.createFile(atPath: hypothesesPath, contents: Data(header.utf8)) else {
            throw RuntimeError("could not create \(hypothesesPath)")
        }
    }
    guard let hypothesesHandle = FileHandle(forWritingAtPath: hypothesesPath) else {
        throw RuntimeError("could not open \(hypothesesPath) for writing")
    }
    hypothesesHandle.seekToEndOfFile()
    defer { hypothesesHandle.closeFile() }

    if !fm.fileExists(atPath: failuresPath) {
        guard fm.createFile(atPath: failuresPath, contents: nil) else {
            throw RuntimeError("could not create \(failuresPath)")
        }
    }
    guard let failuresHandle = FileHandle(forWritingAtPath: failuresPath) else {
        throw RuntimeError("could not open \(failuresPath) for writing")
    }
    failuresHandle.seekToEndOfFile()
    defer { failuresHandle.closeFile() }

    let transcriber: Transcribing = try SFSpeechTranscriber(localeIdentifier: options.localeIdentifier)
    print("Requesting speech recognition authorization and checking on-device support for \(options.localeIdentifier)...")
    try await transcriber.prepare()

    let osVersion = currentOSVersionString()
    let audioRootURL = URL(fileURLWithPath: options.audioRoot, isDirectory: true)

    var written = 0
    var skipped = 0
    var failed = 0

    for row in manifestRows {
        if alreadyDone.contains(row.uttID) {
            skipped += 1
            continue
        }

        let wavURL = URL(fileURLWithPath: row.wavPath, relativeTo: audioRootURL).standardizedFileURL
        guard fm.fileExists(atPath: wavURL.path) else {
            let message = "\(isoTimestamp()) \(row.uttID) SKIPPED (no retry): audio not found at \(wavURL.path)\n"
            failuresHandle.write(Data(message.utf8))
            FileHandle.standardError.write(Data(message.utf8))
            failed += 1
            continue
        }

        do {
            let start = Date()
            let hypothesis = try await withRetry(
                maxAttempts: 5,
                initialDelaySeconds: 2,
                maxDelaySeconds: 30,
                onRetry: { attempt, error in
                    let message = "\(isoTimestamp()) \(row.uttID) retry \(attempt) after error: \(error)\n"
                    FileHandle.standardError.write(Data(message.utf8))
                }
            ) {
                try await transcriber.transcribe(url: wavURL)
            }
            let elapsedMs = Date().timeIntervalSince(start) * 1000

            let line = CSV.row([
                row.uttID,
                hypothesis,
                osVersion,
                transcriber.recognizerName,
                String(format: "%.1f", elapsedMs),
            ]) + "\n"
            hypothesesHandle.write(Data(line.utf8))
            written += 1
        } catch {
            let message = "\(isoTimestamp()) \(row.uttID) FAILED after retries: \(error)\n"
            failuresHandle.write(Data(message.utf8))
            FileHandle.standardError.write(Data(message.utf8))
            failed += 1
        }

        if options.delaySeconds > 0 {
            try? await Task.sleep(nanoseconds: UInt64(options.delaySeconds * 1_000_000_000))
        }
    }

    print("Wrote \(written) hypotheses to \(hypothesesPath) " +
          "(\(skipped) already done, \(failed) failed -- see \(failuresPath)).")
}

let options = parseArguments(Array(CommandLine.arguments.dropFirst()))
do {
    try await run(options: options)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
