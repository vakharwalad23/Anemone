import Anemone
import ArgumentParser

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Answer one prompt, streaming the reply to stdout.")

    @OptionGroup var model: ModelOptions
    @Argument(help: "The prompt.") var prompt: String
    @Option(help: "Most tokens to generate.") var maxTokens = 512
    @Flag(help: "Plain greedy decoding on the ANE, without the drafter.") var noDraft = false
    @Flag(help: "Let the model think before it answers.") var thinking = false
    @Flag(help: "Print speed and acceptance to stderr.") var stats = false

    func run() async throws {
        let languageModel = try await model.load()
        let options = GenerationOptions(maxTokens: maxTokens, speculative: !noDraft, thinking: thinking)
        let response = try await languageModel.respond(to: [ChatMessage(.user, prompt)], options: options, onText: printPiece)
        print()
        if stats { log(summary(response.stats)) }
    }
}
