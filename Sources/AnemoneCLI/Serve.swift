import Anemone
import ArgumentParser
import Foundation
import Hummingbird

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Serve an OpenAI-compatible chat API (POST /v1/chat/completions, GET /v1/models). One request runs at a time."
    )

    @OptionGroup var model: ModelOptions
    @Option(help: "Address to listen on.") var host = "127.0.0.1"
    @Option(help: "Port to listen on.") var port = 8080
    @Option(help: "Most tokens per reply when a request does not set max_tokens.") var maxTokens = 1024

    func run() async throws {
        let api = ChatAPI(model: try await model.load(), name: URL(fileURLWithPath: model.model).lastPathComponent, defaultMaxTokens: maxTokens)
        let router = Router()
        router.get("/v1/models") { _, _ in try json(ModelList(data: [.init(id: api.name)])) }
        router.post("/v1/chat/completions") { request, _ in try await api.complete(request) }
        log("Serving \(api.name) at http://\(host):\(port)/v1")
        try await Application(router: router, configuration: .init(address: .hostname(host, port: port))).runService()
    }
}
