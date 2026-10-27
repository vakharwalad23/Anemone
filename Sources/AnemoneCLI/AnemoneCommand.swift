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
    @Option(name: .shortAndLong, help: "Model folder made by the converter, or a Hugging Face repo id to download; $ANEMONE_MODEL sets the default.")
    var model: String = ProcessInfo.processInfo.environment["ANEMONE_MODEL"] ?? "vakharwalad23/Qwen3-8B-Anemone"

    func load() async throws -> LanguageModel {
        let folder = try await resolve()
        log("Loading \(model). The first load compiles it for the Neural Engine (12 to 23 minutes for an 8B or 14B on an M3 Pro); later loads take seconds.")
        let start = ContinuousClock.now
        let languageModel = try await LanguageModel.load(from: folder)
        log("Loaded in \((ContinuousClock.now - start).formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 1)))).")
        if !languageModel.offNeuralEngine.isEmpty {
            log("Warning: the Neural Engine compiler rejected \(languageModel.offNeuralEngine.count) functions of this model, so Core AI runs them on another device and the model is much slower: \(languageModel.offNeuralEngine.joined(separator: ", ")).")
        }
        return languageModel
    }

    /// A local folder as given, or a repo id downloaded once into the cache.
    func resolve() async throws -> URL {
        let local = URL(fileURLWithPath: model)
        if FileManager.default.fileExists(atPath: local.path) || model.split(separator: "/").count != 2 { return local }
        log("Downloading \(model) from Hugging Face; later runs reuse the download.")
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
