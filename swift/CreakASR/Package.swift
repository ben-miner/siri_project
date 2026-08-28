// swift-tools-version:5.9
import PackageDescription
import Foundation

// Absolute path to Info.plist, computed from this manifest's own location
// rather than the invoking shell's working directory -- see the comment on
// linkerSettings below for why this file needs to exist at all.
let infoPlistPath = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Sources/CreakASR/Info.plist")
    .path

let package = Package(
    name: "CreakASR",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "CreakASR",
            path: "Sources/CreakASR",
            linkerSettings: [
                // Embeds NSSpeechRecognitionUsageDescription into the built
                // binary via a raw __info_plist section. Apple's Speech
                // framework requires this key even for a bare command-line
                // tool with no .app bundle -- without it,
                // SFSpeechRecognizer.requestAuthorization fails closed
                // (denied, no prompt) instead of asking the user. See
                // ../README.md's Troubleshooting section if authorization
                // still doesn't behave as expected -- this is a build/TCC
                // mechanic, not part of the Speech framework's own API
                // surface, and is the one piece of this package I could not
                // verify by running it myself.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlistPath,
                ])
            ]
        )
    ]
)
