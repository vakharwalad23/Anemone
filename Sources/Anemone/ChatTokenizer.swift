import Foundation
import Tokenizers

public struct ChatMessage: Sendable {
    public enum Role: String, Sendable {
        case system, user, assistant
    }

    public let role: Role
    public let content: String

    public init(_ role: Role, _ content: String) {
        self.role = role
        self.content = content
    }
}

/// The model's tokenizer and chat template.
struct ChatTokenizer: Sendable {
    let tokenizer: any Tokenizer

    static func load(_ folder: URL) async throws -> ChatTokenizer {
        ChatTokenizer(tokenizer: try await AutoTokenizer.from(modelFolder: folder))
    }

    func encode(_ messages: [ChatMessage], thinking: Bool) throws -> [Int32] {
        let chat: [Tokenizers.Message] = messages.map { ["role": $0.role.rawValue, "content": $0.content] }
        let ids = try tokenizer.applyChatTemplate(
            messages: chat, chatTemplate: nil, addGenerationPrompt: true, truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": thinking]
        )
        return ids.map { Int32($0) }
    }

    func decode(_ tokens: [Int32]) -> String {
        tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: true)
    }
}

/// Turns a growing token list into text pieces, holding back a piece that ends inside a UTF-8 character.
struct StreamingText {
    private(set) var tokens: [Int32] = []
    private var emitted = 0

    mutating func append(_ new: [Int32], decode: ([Int32]) -> String) -> String {
        tokens += new
        let scalars = decode(tokens).unicodeScalars
        guard scalars.last != "\u{FFFD}", scalars.count > emitted else { return "" }
        let piece = String(String.UnicodeScalarView(scalars.dropFirst(emitted)))
        emitted = scalars.count
        return piece
    }
}
