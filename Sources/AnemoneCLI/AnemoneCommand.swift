import Anemone
import ArgumentParser
import Foundation
import Synchronization

@main
struct AnemoneCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anemone",
        abstract: "Run LLMs on the Apple Neural Engine with a GPU drafter.",
        subcommands: [Run.self, Chat.self, Serve.self, Bench.self]
    )
}

struct ModelOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "Model folder made by the converter, or a Hugging Face repo id to download (default: $ANEMONE_MODEL).")
    var model: String = ProcessInfo.processInfo.environment["ANEMONE_MODEL"] ?? "models/qwen3-8b"

    func load() async throws -> LanguageModel {
        let folder = try await resolve()
        log("Loading \(model). The first load on a Mac compiles it for the Neural Engine, about 7 minutes; later loads are quick.")
        let start = ContinuousClock.now
        let languageModel = try await LanguageModel.load(from: folder)
        log("Loaded in \((ContinuousClock.now - start).formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 1)))).")
        return languageModel
    }

    /// A local folder as given, or a repo id downloaded once into the cache.
    func resolve() async throws -> URL {
        let local = URL(fileURLWithPath: model)
        if FileManager.default.fileExists(atPath: local.path) || model.split(separator: "/").count != 2 { return local }
        log("Downloading \(model) from Hugging Face (about 5 GB the first time).")
        let reported = Mutex(-5)
        return try await ModelFolder.download(model) { fraction in
            let percent = Int(fraction * 100)
            let due = reported.withLock { last in
                guard percent >= last + 5 else { return false }
                last = percent
                return true
            }
            if due { log("\(percent)%") }
        }
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
