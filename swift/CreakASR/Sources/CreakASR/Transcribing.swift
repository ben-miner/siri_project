import Foundation

/// Seam between the batch loop (main.swift) and whichever Speech API does
/// the actual work.
///
/// SFSpeechTranscriber (SFSpeechTranscriber.swift) implements this now,
/// using SFSpeechRecognizer with requiresOnDeviceRecognition = true --
/// chosen because this project currently runs on an Intel iMac (2019) on
/// macOS Sequoia 15.7.9, which cannot run SpeechAnalyzer (that requires
/// macOS 26). When a Mac that can run SpeechAnalyzer becomes available,
/// add a SpeechAnalyzerTranscriber conforming to this same protocol in its
/// own file. main.swift's argument parsing, checkpointing, retry/backoff,
/// and CSV writing all stay untouched -- the only change needed is the one
/// line in main.swift that constructs the `Transcribing` instance.
protocol Transcribing {
    /// Written to hypotheses.csv's `recognizer` column, so rows produced by
    /// different implementations stay distinguishable if a run ever mixes
    /// them (e.g. resuming an SFSpeechRecognizer run isn't safe to do with
    /// a SpeechAnalyzer implementation swapped in -- that's a different
    /// recognizer, not a resume).
    var recognizerName: String { get }

    /// One-time setup (authorization, capability/asset checks). Called
    /// once before the batch loop starts, not per file.
    func prepare() async throws

    /// Transcribe a single audio file, returning its best hypothesis text.
    func transcribe(url: URL) async throws -> String
}
