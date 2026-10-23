import CoreAI
import Foundation

struct Draft {
    let tokens: [Int32]
    let confidence: [Float]
}

/// One cache size of the drafter: its draft and inject functions and the KV cache they share.
final class DrafterCache {
    let slots: Int
    let main: InferenceFunction
    let inject: InferenceFunction?
    var keys: NDArray
    var values: NDArray

    init(main: InferenceFunction, inject: InferenceFunction?) throws {
        self.main = main
        self.inject = inject
        keys = try state(main, "k_cache")
        values = try state(main, "v_cache")
        slots = try stateShape(main, "k_cache")[3]
    }
}

/// The DSpark drafter as a static-shape Core AI program on the GPU: inject target taps, then draft a block (lab e12).
final class Drafter {
    static let tapRows = 8
    static let blockSize = 7
    static let maxPrefix = 4
    static let injectRows = 64

    let tapWidth: Int
    private let caches: [DrafterCache]
    private var active = 0
    private var length = 0

    private init(caches: [DrafterCache]) throws {
        guard case .ndArray(let taps) = caches[0].main.descriptor.inputDescriptor(of: "taps"), let width = taps.shape.last else {
            throw AnemoneError.missingFunction("drafter input taps")
        }
        self.caches = caches
        tapWidth = width
    }

    /// Loads every cache size the program has (`main_<slots>`, or a single `main` in older folders).
    static func load(_ url: URL) async throws -> Drafter {
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        var caches: [DrafterCache] = []
        for name in model.functionNames where name == "main" || name.hasPrefix("main_") {
            guard let main = try model.loadFunction(named: name) else { continue }
            // folders converted before the inject entry point existed still work, injecting 8 rows at a time
            caches.append(try DrafterCache(main: main, inject: try model.loadFunction(named: "inject" + name.dropFirst("main".count))))
        }
        guard !caches.isEmpty else { throw AnemoneError.missingFunction("main") }
        let drafter = try Drafter(caches: caches.sorted { $0.slots < $1.slots })
        for index in drafter.caches.indices {
            drafter.active = index
            try await drafter.inject(taps: [Float16](repeating: 0, count: drafter.tapWidth))
            _ = try await drafter.step(taps: [Float16](repeating: 0, count: drafter.tapWidth), golden: 0)
            drafter.length = 0
        }
        drafter.reset()
        return drafter
    }

    func reset() {
        active = 0
        length = 0
    }

    /// Adds prompt taps (rows of `tapWidth`) to the drafter's context without drafting.
    func inject(taps: [Float16]) async throws {
        let rows = taps.count / tapWidth
        for start in stride(from: 0, to: rows, by: Self.injectRows) {
            let count = min(Self.injectRows, rows - start)
            let slice = taps[(start * tapWidth)..<((start + count) * tapWidth)]
            try grow(toHold: length + Self.injectRows)
            guard let function = caches[active].inject else {
                _ = try await step(taps: Array(slice), golden: 0)
                continue
            }
            var padded = [Float16](repeating: 0, count: Self.injectRows * tapWidth)
            padded.replaceSubrange(0..<slice.count, with: slice)
            let inputs: [String: NDArray] = [
                "taps": ndArray(padded, shape: [1, Self.injectRows, tapWidth], type: .float16),
                "position_ids": NDArray(scalars: (0..<Self.injectRows).map { Int32(length + $0) }, shape: [1, Self.injectRows]),
                "offset": NDArray(scalars: [Int32(length)], shape: [1]),
            ]
            let cache = caches[active]
            _ = try await Anemone.run(function, inputs: inputs, keys: &cache.keys, values: &cache.values, names: ("k_cache", "v_cache"))
            length += count
        }
    }

    /// Adds the taps of the tokens the target just accepted (rows of `tapWidth`), then drafts the block after `golden`.
    func step(taps: [Float16], golden: Int32) async throws -> Draft {
        let rows = taps.count / tapWidth
        var start = 0
        var draft: Draft
        repeat {
            let end = min(rows, start + Self.tapRows)
            draft = try await call(taps[(start * tapWidth)..<(end * tapWidth)], rows: end - start, golden: golden)
            start = end
        } while start < rows
        return draft
    }

    private func call(_ taps: ArraySlice<Float16>, rows: Int, golden: Int32) async throws -> Draft {
        try grow(toHold: length + Self.tapRows + Self.blockSize)
        var padded = [Float16](repeating: 0, count: Self.tapRows * tapWidth)
        padded.replaceSubrange(0..<taps.count, with: taps)
        let positions = (0..<Self.tapRows).map { Int32(length + $0) } + (0..<Self.blockSize).map { Int32(length + rows + $0) }
        let inputs: [String: NDArray] = [
            "taps": ndArray(padded, shape: [1, Self.tapRows, tapWidth], type: .float16),
            "prefix_ids": NDArray(scalars: [golden] + [Int32](repeating: 0, count: Self.maxPrefix - 1), shape: [1, Self.maxPrefix]),
            "position_ids": NDArray(scalars: positions, shape: [1, positions.count]),
            "offset": NDArray(scalars: [Int32(length)], shape: [1]),
            "injected": NDArray(scalars: [Int32(rows)], shape: [1]),
            "prefix": NDArray(scalars: [Int32(1)], shape: [1]),
        ]
        let cache = caches[active]
        var outputs = try await Anemone.run(cache.main, inputs: inputs, keys: &cache.keys, values: &cache.values, names: ("k_cache", "v_cache"))
        length += rows
        return Draft(tokens: scalars(try outputs.array("tokens"), as: Int32.self), confidence: scalars(try outputs.array("confidence"), as: Float.self))
    }

    /// Moves to the smallest cache that holds `slots`, carrying the KV written so far.
    private func grow(toHold slots: Int) throws {
        guard slots > caches[active].slots else { return }
        guard let next = caches.firstIndex(where: { $0.slots >= slots }) else {
            throw AnemoneError.contextFull(context: caches[caches.count - 1].slots)
        }
        let (from, to) = (caches[active], caches[next])
        copyPrefix(from.keys, into: &to.keys, positions: length)
        copyPrefix(from.values, into: &to.values, positions: length)
        active = next
    }
}

/// Copies the first `positions` sequence entries of a `[layers, 1, heads, slots, head_dim]` cache into a larger one.
func copyPrefix(_ source: NDArray, into target: inout NDArray, positions: Int) {
    source.view(as: Float16.self).withUnsafePointer { from, shape, fromStrides in
        target.mutableView(as: Float16.self).withUnsafeMutablePointer { to, _, toStrides in
            for layer in 0..<shape[0] {
                for head in 0..<shape[2] {
                    for position in 0..<positions {
                        let source = layer * fromStrides[0] + head * fromStrides[2] + position * fromStrides[3]
                        let target = layer * toStrides[0] + head * toStrides[2] + position * toStrides[3]
                        (to + target).update(from: from + source, count: shape[4])
                    }
                }
            }
        }
    }
}
