import Foundation

/// The target's fp16 token embedding table, memory-mapped so only the rows in use are read.
struct Embeddings: @unchecked Sendable {
    let hiddenSize: Int
    private let data: Data

    init(url: URL, hiddenSize: Int, vocabSize: Int) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        self.hiddenSize = hiddenSize
        let expected = vocabSize * hiddenSize * MemoryLayout<Float16>.size
        guard data.count == expected else {
            throw AnemoneError.badEmbeddings("\(url.lastPathComponent) has \(data.count) bytes, expected \(expected)")
        }
    }

    /// Rows for `ids`, row major, padded with the row of token 0 up to `width` rows.
    func rows(_ ids: [Int32], width: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: width * hiddenSize)
        data.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            out.withUnsafeMutableBufferPointer { buffer in
                for row in 0..<width {
                    let id = row < ids.count ? Int(ids[row]) : 0
                    let source = UnsafeBufferPointer(rebasing: table[(id * hiddenSize)..<((id + 1) * hiddenSize)])
                    _ = UnsafeMutableBufferPointer(rebasing: buffer[(row * hiddenSize)..<((row + 1) * hiddenSize)]).update(fromContentsOf: source)
                }
            }
        }
        return out
    }
}
