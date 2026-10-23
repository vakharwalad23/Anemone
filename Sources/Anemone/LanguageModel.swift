import Foundation

public struct GenerationOptions: Sendable {
    public var maxTokens: Int
    /// Draft on the GPU and verify on the ANE; off means plain greedy decoding on the ANE.
    public var speculative: Bool
    public var thinking: Bool

    public init(maxTokens: Int = 512, speculative: Bool = true, thinking: Bool = false) {
        self.maxTokens = maxTokens
        self.speculative = speculative
        self.thinking = thinking
    }
}

public struct Response: Sendable {
    public let text: String
    public let tokens: [Int32]
    public let stats: GenerationStats
}

public enum GenerationEvent: Sendable {
    case text(String)
    case finished(Response)
}

/// A converted model loaded for generation: the target on the ANE, the drafter on the GPU. One request at a time.
public actor LanguageModel {
    public let folder: ModelFolder
    private let speculator: Speculator
    private let tokenizer: ChatTokenizer

    private init(folder: ModelFolder, speculator: Speculator, tokenizer: ChatTokenizer) {
        self.folder = folder
        self.speculator = speculator
        self.tokenizer = tokenizer
    }

    /// Loads a model folder. The first load on a Mac compiles the programs for the ANE, which takes minutes.
    public static func load(from url: URL) async throws -> LanguageModel {
        let folder = try ModelFolder(url)
        let target = try await AneTarget.load(folder)
        let drafter = try await Drafter.load(folder.file(folder.config.drafter))
        let tokenizer = try await ChatTokenizer.load(url)
        let speculator = Speculator(target: target, drafter: drafter, eos: Set(folder.config.eosTokenIds))
        return LanguageModel(folder: folder, speculator: speculator, tokenizer: tokenizer)
    }

    public func respond(
        to messages: [ChatMessage], options: GenerationOptions = GenerationOptions(), onText: (String) -> Void = { _ in }
    ) async throws -> Response {
        let prompt = try tokenizer.encode(messages, thinking: options.thinking)
        var stream = StreamingText()
        let stats = try await speculator.generate(prompt, maxTokens: options.maxTokens, speculative: options.speculative) { tokens in
            let piece = stream.append(tokens, decode: tokenizer.decode)
            if !piece.isEmpty { onText(piece) }
        }
        return Response(text: tokenizer.decode(stream.tokens), tokens: stream.tokens, stats: stats)
    }

    public nonisolated func generate(
        _ messages: [ChatMessage], options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<GenerationEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let response = try await self.respond(to: messages, options: options) { continuation.yield(.text($0)) }
                    continuation.yield(.finished(response))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
