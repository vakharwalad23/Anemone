import Anemone
import Foundation
import Hummingbird

/// The subset of OpenAI's chat completions request Anemone honours; sampling settings are ignored (greedy only).
struct ChatRequest: Decodable {
    struct Message: Decodable {
        let role: String
        let content: Content
    }

    /// A message body: plain text, or a list of parts whose text parts are joined.
    struct Content: Decodable {
        struct Part: Decodable {
            let type: String
            let text: String?
        }

        let text: String

        init(from decoder: any Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self.text = text
            } else {
                text = try container.decode([Part].self).compactMap { $0.type == "text" ? $0.text : nil }.joined(separator: "\n")
            }
        }
    }

    let messages: [Message]
    let maxTokens: Int?
    let maxCompletionTokens: Int?
    let stream: Bool?
}

struct Completion: Encodable {
    struct Choice: Encodable {
        let index: Int
        let message: Reply
        let finishReason: String
    }

    struct Reply: Encodable {
        let role: String
        let content: String
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int
    }

    let id: String
    let object = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
}

struct CompletionChunk: Encodable {
    struct Choice: Encodable {
        let index: Int
        let delta: Delta
        let finishReason: String?
    }

    struct Delta: Encodable {
        let role: String?
        let content: String?
    }

    let id: String
    let object = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
}

struct ModelList: Encodable {
    struct Entry: Encodable {
        let id: String
        let object = "model"
        let ownedBy = "local"
    }

    let object = "list"
    let data: [Entry]
}

/// Answers `/v1/chat/completions` with the loaded model, streamed as server-sent events when asked.
struct ChatAPI: Sendable {
    let model: LanguageModel
    let name: String
    let defaultMaxTokens: Int

    func complete(_ request: Request) async throws -> Hummingbird.Response {
        let body = try await request.body.collect(upTo: 16 << 20)
        let chat: ChatRequest
        do {
            chat = try decoder.decode(ChatRequest.self, from: body)
        } catch {
            throw HTTPError(.badRequest, message: "not a chat completions request: \(error)")
        }
        let messages = chat.messages.map { ChatMessage(role(of: $0.role), $0.content.text) }
        let options = GenerationOptions(maxTokens: chat.maxCompletionTokens ?? chat.maxTokens ?? defaultMaxTokens)
        let id = "chatcmpl-\(UUID().uuidString)"
        let created = Int(Date().timeIntervalSince1970)
        guard chat.stream == true else {
            let reply = try await model.respond(to: messages, options: options)
            let usage = Completion.Usage(
                promptTokens: reply.stats.promptTokens, completionTokens: reply.tokens.count,
                totalTokens: reply.stats.promptTokens + reply.tokens.count)
            let choice = Completion.Choice(index: 0, message: .init(role: "assistant", content: reply.text), finishReason: finishReason(reply))
            return try json(Completion(id: id, created: created, model: name, choices: [choice], usage: usage))
        }
        let events = model.generate(messages, options: options)
        let name = name
        return Hummingbird.Response(
            status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"],
            body: ResponseBody { writer in
                let event = { (delta: CompletionChunk.Delta, finish: String?) in
                    try serverSentEvent(CompletionChunk(id: id, created: created, model: name, choices: [.init(index: 0, delta: delta, finishReason: finish)]))
                }
                try await writer.write(event(.init(role: "assistant", content: ""), nil))
                for try await update in events {
                    switch update {
                    case .text(let piece): try await writer.write(event(.init(role: nil, content: piece), nil))
                    case .finished(let reply): try await writer.write(event(.init(role: nil, content: nil), finishReason(reply)))
                    }
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                try await writer.finish(nil)
            })
    }
}

func role(of name: String) -> ChatMessage.Role {
    switch name {
    case "system", "developer": .system
    case "assistant": .assistant
    default: .user
    }
}

func serverSentEvent(_ chunk: CompletionChunk) throws -> ByteBuffer {
    ByteBuffer(string: "data: \(String(decoding: try encoder.encode(chunk), as: UTF8.self))\n\n")
}

func finishReason(_ reply: Anemone.Response) -> String {
    reply.hitTokenLimit ? "length" : "stop"
}

func json(_ value: some Encodable) throws -> Hummingbird.Response {
    Hummingbird.Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: try encoder.encode(value))))
}

let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return encoder
}()

let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}()
