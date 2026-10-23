import Anemone
import ArgumentParser
import Foundation

@main
struct AnemoneCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anemone",
        abstract: "Run LLMs on the Apple Neural Engine with a GPU drafter.",
        subcommands: [Run.self, Chat.self, Bench.self]
    )
}

struct ModelOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "Model folder made by the converter (default: $ANEMONE_MODEL).")
    var model: String = ProcessInfo.processInfo.environment["ANEMONE_MODEL"] ?? "models/qwen3-8b"

    func load() async throws -> LanguageModel {
        log("Loading \(model). The first load on a Mac compiles it for the Neural Engine, about 7 minutes; later loads are quick.")
        let start = ContinuousClock.now
        let languageModel = try await LanguageModel.load(from: URL(fileURLWithPath: model))
        log("Loaded in \((ContinuousClock.now - start).formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 1)))).")
        return languageModel
    }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func printPiece(_ piece: String) {
    print(piece, terminator: "")
    fflush(stdout)
}

func summary(_ stats: GenerationStats) -> String {
    var line = String(format: "%d tokens, %.1f tok/s, prefill %.0f ms for %d prompt tokens", stats.generatedTokens, stats.tokensPerSecond,
                      stats.prefillSeconds * 1000, stats.promptTokens)
    if stats.cycles > 0 {
        line += String(format: ", %.2f drafts accepted per cycle", stats.acceptedPerCycle)
    }
    return line
}
