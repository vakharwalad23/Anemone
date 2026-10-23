import Foundation

/// The `anemone.json` written by the converter next to the model's programs.
public struct ModelConfig: Decodable, Sendable {
    public let model: String
    public let chunks: [String]
    public let drafter: String
    public let embeddings: String
    public let hiddenSize: Int
    public let vocabSize: Int
    public let eosTokenIds: [Int32]
}

/// A converted model: ANE target chunks, GPU drafter, embedding table, tokenizer and config.
public struct ModelFolder: Sendable {
    public let url: URL
    public let config: ModelConfig

    public init(_ url: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.url = url
        config = try decoder.decode(ModelConfig.self, from: Data(contentsOf: url.appending(path: "anemone.json")))
    }

    func file(_ name: String) -> URL {
        url.appending(path: name)
    }
}

public enum AnemoneError: Error, CustomStringConvertible {
    case contextFull(context: Int)
    case batchTooWide(tokens: Int, widest: Int)
    case missingFunction(String)
    case missingOutput(String)
    case badEmbeddings(String)

    public var description: String {
        switch self {
        case .contextFull(let context): "the conversation no longer fits the model's \(context)-token context"
        case .batchTooWide(let tokens, let widest): "\(tokens) tokens do not fit one call of width \(widest)"
        case .missingFunction(let name): "the model has no function \(name)"
        case .missingOutput(let name): "a model call returned no \(name)"
        case .badEmbeddings(let reason): "embedding table: \(reason)"
        }
    }
}
