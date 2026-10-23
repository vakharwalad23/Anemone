import Anemone
import ArgumentParser
import Foundation

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Plain and speculative decoding on six fixed prompts: speed, acceptance, and a token-for-token lossless check."
    )

    static let prompts: [(name: String, text: String)] = [
        ("math", "What is the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square?"),
        ("code", "Write a Python function that returns the longest palindromic substring of a string, with a short explanation."),
        ("explain", "Explain how a CPU cache works to a first-year computer science student."),
        ("summary", "Summarize the causes of the French Revolution in five bullet points."),
        ("advice", "I have a job interview tomorrow for a backend engineer role. How should I prepare tonight?"),
        ("translate", "Translate into French: The weather is lovely today, so we are going to walk along the river after lunch."),
    ]

    struct Row: Codable {
        let prompt: String
        let plain: GenerationStats
        let speculative: GenerationStats
        let lossless: Bool
    }

    @OptionGroup var model: ModelOptions
    @Option(help: "Tokens per reply.") var maxTokens = 256
    @Option(help: "Write the results as JSON to this file.") var json: String?

    func run() async throws {
        let languageModel = try await model.load()
        var rows: [Row] = []
        for (name, text) in Self.prompts {
            let messages = [ChatMessage(.user, text)]
            let plain = try await languageModel.respond(to: messages, options: GenerationOptions(maxTokens: maxTokens, speculative: false))
            let drafted = try await languageModel.respond(to: messages, options: GenerationOptions(maxTokens: maxTokens))
            let lossless = drafted.tokens == Array(plain.tokens.prefix(drafted.tokens.count))
            rows.append(Row(prompt: name, plain: plain.stats, speculative: drafted.stats, lossless: lossless))
            let cycles = Double(max(drafted.stats.cycles, 1))
            print(
                name.padding(toLength: 10, withPad: " ", startingAt: 0)
                    + String(
                        format: "plain %5.1f tok/s   speculative %5.1f tok/s   accepted %.2f   draft %4.1f ms   verify %4.1f ms   lossless %@",
                        plain.stats.tokensPerSecond, drafted.stats.tokensPerSecond, drafted.stats.acceptedPerCycle,
                        drafted.stats.draftSeconds * 1000 / cycles, drafted.stats.verifySeconds * 1000 / cycles, lossless ? "yes" : "NO"))
        }
        let mean = { (values: [Double]) in values.reduce(0, +) / Double(values.count) }
        print("mean      " + String(format: "plain %5.1f tok/s   speculative %5.1f tok/s   accepted %.2f   lossless %d/%d",
                     mean(rows.map(\.plain.tokensPerSecond)), mean(rows.map(\.speculative.tokensPerSecond)),
                     mean(rows.map(\.speculative.acceptedPerCycle)), rows.filter(\.lossless).count, rows.count))
        if let json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(rows).write(to: URL(fileURLWithPath: json))
        }
    }
}
