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

    enum Mode: String, ExpressibleByArgument {
        case plain, speculative
    }

    struct Row: Codable {
        let prompt: String
        let plain: GenerationStats?
        let speculative: GenerationStats?
        let lossless: Bool?
        /// The reply's token ids (plain greedy when it ran), so quality tools can score the text the engine served.
        let tokens: [Int32]
    }

    struct Report: Codable {
        let machine: Machine
        let model: String
        let maxTokens: Int
        let rows: [Row]
    }

    @OptionGroup var model: ModelOptions
    @Option(help: "Tokens per reply.") var maxTokens = 256
    @Option(help: "Run only one mode (no lossless check).") var only: Mode?
    @Option(help: "Write the results as JSON to this file.") var json: String?

    func run() async throws {
        let languageModel = try await model.load()
        var rows: [Row] = []
        for (name, text) in Self.prompts {
            let messages = [ChatMessage(.user, text)]
            let plainOptions = GenerationOptions(maxTokens: maxTokens, speculative: false)
            let plain = only == .speculative ? nil : try await languageModel.respond(to: messages, options: plainOptions)
            let drafted = only == .plain ? nil : try await languageModel.respond(to: messages, options: GenerationOptions(maxTokens: maxTokens))
            let lossless = plain.flatMap { plain in drafted.map { $0.tokens == Array(plain.tokens.prefix($0.tokens.count)) } }
            let tokens = (plain ?? drafted)?.tokens ?? []
            rows.append(Row(prompt: name, plain: plain?.stats, speculative: drafted?.stats, lossless: lossless, tokens: tokens))
            print(name.padding(toLength: 10, withPad: " ", startingAt: 0) + line(plain: plain?.stats, speculative: drafted?.stats, lossless: lossless))
        }
        print("mean      " + summary(rows))
        if let json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let report = Report(machine: .current(), model: model.model, maxTokens: maxTokens, rows: rows)
            try encoder.encode(report).write(to: URL(fileURLWithPath: json))
        }
    }

    func line(plain: GenerationStats?, speculative: GenerationStats?, lossless: Bool?) -> String {
        var text = ""
        if let plain { text += String(format: "plain %5.1f tok/s   ", plain.tokensPerSecond) }
        if let speculative {
            let cycles = Double(max(speculative.cycles, 1))
            text += String(
                format: "speculative %5.1f tok/s   accepted %.2f   draft %4.1f ms   verify %4.1f ms   ", speculative.tokensPerSecond,
                speculative.acceptedPerCycle, speculative.draftSeconds * 1000 / cycles, speculative.verifySeconds * 1000 / cycles)
        }
        if let lossless { text += "lossless \(lossless ? "yes" : "NO")" }
        return text
    }

    func summary(_ rows: [Row]) -> String {
        let mean = { (values: [Double]) in values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
        var text = ""
        if only != .speculative { text += String(format: "plain %5.1f tok/s   ", mean(rows.compactMap(\.plain?.tokensPerSecond))) }
        if only != .plain {
            text += String(
                format: "speculative %5.1f tok/s   accepted %.2f   ", mean(rows.compactMap(\.speculative?.tokensPerSecond)),
                mean(rows.compactMap(\.speculative?.acceptedPerCycle)))
        }
        if only == nil { text += "lossless \(rows.filter { $0.lossless == true }.count)/\(rows.count)" }
        return text
    }
}
