import ArgumentParser
import Foundation

@main
struct AnemoneSpeech: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anemone-speech",
        abstract: "Speech research tools: Orukeet through Core AI.",
        subcommands: [Transcribe.self, Bench.self],
        defaultSubcommand: Transcribe.self
    )
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func milliseconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1e3 + Double(duration.components.attoseconds) / 1e15
}
