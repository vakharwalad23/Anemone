import Foundation
import Hub

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
        let manifest = url.appending(path: "anemone.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else { throw AnemoneError.notAModelFolder(url.path) }
        config = try decoder.decode(ModelConfig.self, from: Data(contentsOf: manifest))
    }

    func file(_ name: String) -> URL {
        url.appending(path: name)
    }

    /// Downloads a model folder from a Hugging Face repo (`org/name`) into `~/.cache/anemone`, reusing files already there.
    public static func download(_ repo: String, progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> URL {
        let base = FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/anemone")
        return try await HubApi(downloadBase: base).snapshot(from: Hub.Repo(id: repo)) { progress($0.fractionCompleted) }
    }
}

public enum AnemoneError: Error, CustomStringConvertible {
    case contextFull(context: Int)
    case batchTooWide(tokens: Int, widest: Int)
    case missingFunction(String)
    case missingOutput(String)
    case badEmbeddings(String)
    case notAModelFolder(String)
    case staleCompileCache

    public var description: String {
        switch self {
        case .contextFull(let context): "the conversation no longer fits the model's \(context)-token context"
        case .batchTooWide(let tokens, let widest): "\(tokens) tokens do not fit one call of width \(widest)"
        case .missingFunction(let name): "the model has no function \(name)"
        case .missingOutput(let name): "a model call returned no \(name)"
        case .badEmbeddings(let reason): "embedding table: \(reason)"
        case .notAModelFolder(let path): "\(path) is not an Anemone model folder (no anemone.json); make one with the converter"
        case .staleCompileCache:
            "Core AI could not use its compiled copy of this model (seen after a crash and for some older cache entries). Remove "
                + "~/Library/Caches/coreai-cache/<macOS build>/<app name> (for the CLI: rm -rf ~/Library/Caches/coreai-cache/*/anemone) "
                + "and load again; the next load compiles from scratch."
        }
    }
}
