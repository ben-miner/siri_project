import Foundation
import Speech

enum TranscriptionError: Error, CustomStringConvertible {
    case authorizationDenied(SFSpeechRecognizerAuthorizationStatus)
    case localeUnsupported(String)
    case onDeviceRecognitionUnsupported(String)
    case timedOut

    var description: String {
        switch self {
        case .authorizationDenied(let status):
            return "speech recognition authorization not granted (status: \(status))"
        case .localeUnsupported(let locale):
            return "SFSpeechRecognizer has no recognizer for locale \(locale)"
        case .onDeviceRecognitionUnsupported(let locale):
            return "on-device recognition is not available for locale \(locale) on this Mac -- " +
                   "check System Settings > Keyboard > Dictation for the offline/on-device language download"
        case .timedOut:
            return "recognition task did not complete within the per-file timeout"
        }
    }
}

/// Transcribing implementation backed by SFSpeechRecognizer with
/// requiresOnDeviceRecognition = true. This is the older ("legacy") Speech
/// framework API, used here only because this run's hardware (Intel iMac,
/// macOS Sequoia 15.7.9) cannot run SpeechAnalyzer, which requires macOS
/// 26. See Transcribing.swift for how a SpeechAnalyzer implementation
/// would slot in alongside this one later without touching the batch loop.
final class SFSpeechTranscriber: Transcribing {
    let recognizerName = "SFSpeechRecognizer"

    private let recognizer: SFSpeechRecognizer
    private let localeIdentifier: String

    /// SFSpeechRecognizer's per-request duration limit and its throttling
    /// behavior under sustained batch use are not precisely documented by
    /// Apple. This timeout exists so a stalled request fails (and gets
    /// retried/logged like any other failure) instead of hanging the batch
    /// loop forever. 90s is generous headroom over this project's
    /// few-second utterance clips.
    private static let perFileTimeoutNanoseconds: UInt64 = 90 * 1_000_000_000

    init(localeIdentifier: String = "en-US") throws {
        self.localeIdentifier = localeIdentifier
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier)) else {
            throw TranscriptionError.localeUnsupported(localeIdentifier)
        }
        self.recognizer = recognizer
    }

    func prepare() async throws {
        let status = await withCheckedContinuation { (continuation: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        guard status == .authorized else {
            throw TranscriptionError.authorizationDenied(status)
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw TranscriptionError.onDeviceRecognitionUnsupported(localeIdentifier)
        }
        if !recognizer.isAvailable {
            let message = "warning: SFSpeechRecognizer reports isAvailable == false at startup; " +
                           "will still attempt per-file recognition.\n"
            FileHandle.standardError.write(Data(message.utf8))
        }
    }

    func transcribe(url: URL) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask { try await self.recognize(url: url) }
            group.addTask {
                try await Task.sleep(nanoseconds: Self.perFileTimeoutNanoseconds)
                throw TranscriptionError.timedOut
            }
            guard let result = try await group.next() else {
                throw TranscriptionError.timedOut
            }
            group.cancelAll()
            return result
        }
    }

    private func recognize(url: URL) async throws -> String {
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        // .unspecified deliberately: .dictation/.confirmation/.search bias
        // formatting and punctuation heuristics in ways that would be an
        // unwanted confound in a WER comparison.
        request.taskHint = .unspecified

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            var resumed = false
            recognizer.recognitionTask(with: request) { result, error in
                guard !resumed else { return }
                if let error {
                    resumed = true
                    continuation.resume(throwing: error)
                    return
                }
                guard let result else { return }
                if result.isFinal {
                    resumed = true
                    continuation.resume(returning: result.bestTranscription.formattedString)
                }
            }
        }
    }
}
