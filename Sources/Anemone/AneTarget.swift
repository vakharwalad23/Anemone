import CoreAI
import Foundation

/// The target's output for the tokens one forward added: its top choice per row and the drafter taps.
struct TargetStep {
    var tokens: [Int32] = []
    var taps: [Float16] = []
}

/// One 9-layer chunk program on the ANE: a function per (context rung, call width), and one KV cache they share.
final class AneChunk {
    let functions: [Int: [Int: InferenceFunction]]
    let output: String
    private var keys: NDArray
    private var values: NDArray

    init(model: AIModel, output: String) throws {
        var functions: [Int: [Int: InferenceFunction]] = [:]
        for name in model.functionNames {
            let parts = name.split(separator: "_")
            guard parts.count == 3, parts[0] == "extend", let rung = Int(parts[1]), let width = Int(parts[2]),
                let function = try model.loadFunction(named: name)
            else { continue }
            functions[rung, default: [:]][width] = function
        }
        guard let largest = functions.keys.max(), let reference = functions[largest]?.values.first else {
            throw AnemoneError.missingFunction("extend_<context>_<width>")
        }
        // a smaller rung reuses the largest rung's buffer only if it was compiled against the same strides
        let strides = try stateStrides(reference, "key_cache")
        self.functions = try functions.filter { _, byWidth in try byWidth.values.allSatisfy { try stateStrides($0, "key_cache") == strides } }
        self.output = output
        keys = try state(reference, "key_cache")
        values = try state(reference, "value_cache")
    }

    func run(rung: Int, width: Int, inputs: [String: NDArray]) async throws -> InferenceFunction.Outputs {
        guard let function = functions[rung]?[width] else { throw AnemoneError.missingFunction("extend_\(rung)_\(width)") }
        return try await Anemone.run(function, inputs: inputs, keys: &keys, values: &values, names: ("key_cache", "value_cache"))
    }
}

/// The large model on the ANE: four chunks fed one after another, with the KV window they share.
final class AneTarget {
    private let chunks: [AneChunk]
    private let embeddings: Embeddings
    private let rungs: [Int]
    private var window: KVWindow

    var context: Int { window.context }

    private init(chunks: [AneChunk], embeddings: Embeddings) {
        self.chunks = chunks
        self.embeddings = embeddings
        let common = chunks.dropFirst().reduce(Set(chunks[0].functions.keys)) { $0.intersection($1.functions.keys) }
        rungs = common.sorted()
        let largest = rungs[rungs.count - 1]
        window = KVWindow(context: largest, widths: chunks[0].functions[largest].map { Array($0.keys) } ?? [])
    }

    static func load(_ folder: ModelFolder) async throws -> AneTarget {
        let config = folder.config
        var chunks: [AneChunk] = []
        for (index, name) in config.chunks.enumerated() {
            let output = index == config.chunks.count - 1 ? "out_tokens" : "hidden"
            chunks.append(try await loadProgram(folder.file(name), on: .neuralEngine) { try AneChunk(model: $0, output: output) })
        }
        let embeddings = try Embeddings(url: folder.file(config.embeddings), hiddenSize: config.hiddenSize, vocabSize: config.vocabSize)
        let target = AneTarget(chunks: chunks, embeddings: embeddings)
        try await target.warm()
        return target
    }

    func reset() {
        window.reset()
    }

    func rewind(_ count: Int) {
        window.rewind(count)
    }

    /// Cached tokens the next call replays because KV writes start at multiples of 4.
    var replay: Int { window.replay }

    /// Runs `tree` after the cached tokens without keeping it: the target's choice and taps for every node.
    func verify(_ tree: DraftTree) async throws -> TargetStep {
        let plan = try window.plan(tree.tokens)
        let replay = plan.newRows.lowerBound
        let rung = smallestRung(holding: plan)
        let mask = treeMask(context: rung, width: plan.width, offset: plan.offset, replay: replay, tree: tree)
        let positions = treePositions(width: plan.width, offset: plan.offset, replay: replay, tree: tree)
        return try await execute(plan, rung: rung, positions: positions, mask: mask)
    }

    /// Keeps the accepted path of the last verified tree; only its first `inPlace` tokens have KV in their own slots.
    func accept(_ tokens: [Int32], inPlace: Int) {
        window.accept(tokens, inPlace: inPlace)
    }

    /// Runs `tokens` after the ones already in the cache, splitting them over as many calls as needed.
    func forward(_ tokens: [Int32]) async throws -> TargetStep {
        var step = TargetStep()
        var start = 0
        while start < tokens.count {
            let end = min(tokens.count, start + window.capacity)
            let chunk = Array(tokens[start..<end])
            let plan = try window.plan(chunk)
            let part = try await execute(plan, rung: smallestRung(holding: plan))
            window.commit(chunk)
            step.tokens += part.tokens
            step.taps += part.taps
            start = end
        }
        return step
    }

    private func warm() async throws {
        // the first call of each function prepares Core AI's delegated ops, so it is kept out of timed decoding
        for rung in rungs {
            for width in window.widths {
                _ = try await execute(TargetCall(offset: 0, width: width, batch: [], newRows: 0..<0), rung: rung)
            }
        }
    }

    /// The smallest rung that holds the call reads the least KV (lab night run: 60.5 ms at ctx 256 vs 65.3 at 1024).
    private func smallestRung(holding plan: TargetCall) -> Int {
        rungs.first { $0 >= plan.offset + plan.width } ?? window.context
    }

    private func execute(_ plan: TargetCall, rung: Int, positions: [UInt16]? = nil, mask: [Float16]? = nil) async throws -> TargetStep {
        let width = plan.width
        let positions = positions ?? (plan.offset..<(plan.offset + width)).map { UInt16($0) }
        let mask = mask ?? causalMask(context: rung, width: width, offset: plan.offset)
        let shared: [String: NDArray] = [
            "position_ids": NDArray(scalars: positions, shape: [1, width]),
            "in_step": NDArray(scalars: [Int32(plan.offset)], shape: [1]),
            "causal_mask": ndArray(mask, shape: [1, rung, 1, width], type: .float16),
        ]
        var current = ndArray(embeddings.rows(plan.batch, width: width), shape: [1, width, 1, embeddings.hiddenSize], type: .float16)
        var taps: [(values: [Float16], width: Int)] = []
        for chunk in chunks {
            var inputs = shared
            inputs["transformer_input"] = current
            var outputs = try await chunk.run(rung: rung, width: width, inputs: inputs)
            if outputs.names.contains("taps") {
                let array = try outputs.array("taps")
                taps.append((scalars(array, as: Float16.self), array.shape[array.shape.count - 1]))
            }
            current = try outputs.array(chunk.output)
        }
        return TargetStep(tokens: Array(scalars(current, as: Int32.self)[plan.newRows]), taps: tapRows(taps, rows: plan.newRows))
    }
}

/// Concatenates each chunk's taps per row, in layer order, keeping only `rows`.
func tapRows(_ parts: [(values: [Float16], width: Int)], rows: Range<Int>) -> [Float16] {
    var out: [Float16] = []
    out.reserveCapacity(rows.count * parts.reduce(0) { $0 + $1.width })
    for row in rows {
        for part in parts {
            out += part.values[(row * part.width)..<((row + 1) * part.width)]
        }
    }
    return out
}
