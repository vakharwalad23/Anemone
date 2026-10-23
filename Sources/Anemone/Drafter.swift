import CoreAI
import Foundation

struct Draft {
    let tokens: [Int32]
    let confidence: [Float]
}

/// The DSpark drafter as one static-shape Core AI program on the GPU: inject target taps, then draft a block (lab e12).
final class Drafter {
    static let tapRows = 8
    static let blockSize = 7
    static let maxPrefix = 4
    static let injectRows = 64

    let tapWidth: Int
    private let function: InferenceFunction
    private let injectFunction: InferenceFunction?
    private var keys: NDArray
    private var values: NDArray
    private var length = 0

    private init(function: InferenceFunction, inject: InferenceFunction?) throws {
        guard case .ndArray(let taps) = function.descriptor.inputDescriptor(of: "taps"), let width = taps.shape.last else {
            throw AnemoneError.missingFunction("drafter input taps")
        }
        self.function = function
        injectFunction = inject
        tapWidth = width
        keys = try state(function, "k_cache")
        values = try state(function, "v_cache")
    }

    static func load(_ url: URL) async throws -> Drafter {
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let function = try model.loadFunction(named: "main") else { throw AnemoneError.missingFunction("main") }
        // folders converted before the inject entry point existed still work, injecting 8 rows at a time
        let drafter = try Drafter(function: function, inject: try model.loadFunction(named: "inject"))
        try await drafter.inject(taps: [Float16](repeating: 0, count: drafter.tapWidth))
        _ = try await drafter.step(taps: [Float16](repeating: 0, count: drafter.tapWidth), golden: 0)
        drafter.reset()
        return drafter
    }

    func reset() {
        length = 0
    }

    /// Adds prompt taps (rows of `tapWidth`) to the drafter's context without drafting.
    func inject(taps: [Float16]) async throws {
        guard let injectFunction else {
            _ = try await step(taps: taps, golden: 0)
            return
        }
        let rows = taps.count / tapWidth
        for start in stride(from: 0, to: rows, by: Self.injectRows) {
            let count = min(Self.injectRows, rows - start)
            var padded = [Float16](repeating: 0, count: Self.injectRows * tapWidth)
            padded.replaceSubrange(0..<(count * tapWidth), with: taps[(start * tapWidth)..<((start + count) * tapWidth)])
            let inputs: [String: NDArray] = [
                "taps": ndArray(padded, shape: [1, Self.injectRows, tapWidth], type: .float16),
                "position_ids": NDArray(scalars: (0..<Self.injectRows).map { Int32(length + $0) }, shape: [1, Self.injectRows]),
                "offset": NDArray(scalars: [Int32(length)], shape: [1]),
            ]
            _ = try await Anemone.run(injectFunction, inputs: inputs, keys: &keys, values: &values, names: ("k_cache", "v_cache"))
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
        var outputs = try await Anemone.run(function, inputs: inputs, keys: &keys, values: &values, names: ("k_cache", "v_cache"))
        length += rows
        return Draft(tokens: scalars(try outputs.array("tokens"), as: Int32.self), confidence: scalars(try outputs.array("confidence"), as: Float.self))
    }
}
