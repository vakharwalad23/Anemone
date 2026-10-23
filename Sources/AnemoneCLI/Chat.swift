import Anemone
import ArgumentParser

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Talk to the model in the terminal. An empty line or Ctrl-D quits.")

    @OptionGroup var model: ModelOptions
    @Option(help: "Most tokens per reply.") var maxTokens = 1024
    @Option(help: "System prompt.") var system: String?
    @Flag(help: "Let the model think before it answers.") var thinking = false
    @Flag(help: "Print speed and acceptance after each reply.") var stats = false

    func run() async throws {
        let languageModel = try await model.load()
        var messages = system.map { [ChatMessage(.system, $0)] } ?? []
        let options = GenerationOptions(maxTokens: maxTokens, thinking: thinking)
        while true {
            print("> ", terminator: "")
            guard let line = readLine(), !line.isEmpty else { return }
            messages.append(ChatMessage(.user, line))
            let response = try await languageModel.respond(to: messages, options: options, onText: printPiece)
            print("\n")
            messages.append(ChatMessage(.assistant, response.text))
            if stats { log(summary(response.stats)) }
        }
    }
}
