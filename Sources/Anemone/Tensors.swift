import CoreAI

/// Elements of an NDArray in logical row-major order, whatever its strides.
func scalars<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [T] {
    array.view(as: type).withUnsafePointer { pointer, shape, strides in
        let dims = Array(shape.indices.map { shape[$0] })
        let steps = Array(strides.indices.map { strides[$0] })
        let count = dims.reduce(1, *)
        guard let inner = dims.last, inner > 0, count > 0 else { return [] }
        var result = [T]()
        result.reserveCapacity(count)
        var index = [Int](repeating: 0, count: dims.count)
        for _ in 0..<(count / inner) {
            let base = zip(index, steps).reduce(0) { $0 + $1.0 * $1.1 }
            if steps[steps.count - 1] == 1 {
                result.append(contentsOf: UnsafeBufferPointer(start: pointer + base, count: inner))
            } else {
                for column in 0..<inner { result.append(pointer[base + column * steps[steps.count - 1]]) }
            }
            var dim = dims.count - 2
            while dim >= 0 {
                index[dim] += 1
                if index[dim] < dims[dim] { break }
                index[dim] = 0
                dim -= 1
            }
        }
        return result
    }
}

/// Additive attention mask `[1, context, 1, width]`: query `q` sees keys up to `offset + q`.
func causalMask(context: Int, width: Int, offset: Int) -> [Float16] {
    var mask = [Float16](repeating: 0, count: context * width)
    for key in 0..<context {
        for query in 0..<width where key > offset + query {
            mask[key * width + query] = -.infinity
        }
    }
    return mask
}

extension InferenceFunction.Outputs {
    mutating func array(_ name: String) throws -> NDArray {
        guard let array = remove(name)?.ndArray else { throw AnemoneError.missingOutput(name) }
        return array
    }
}

/// A zeroed buffer for one of a function's states.
func state(_ function: InferenceFunction, _ name: String) throws -> NDArray {
    guard case .ndArray(let descriptor) = function.descriptor.stateDescriptor(of: name) else { throw AnemoneError.missingFunction("state \(name)") }
    return NDArray(descriptor: descriptor)
}

/// Runs `function` with its two KV states bound in place.
func run(
    _ function: InferenceFunction, inputs: [String: NDArray], keys: inout NDArray, values: inout NDArray, names: (keys: String, values: String)
) async throws -> InferenceFunction.Outputs {
    let keyShape = try stateShape(function, names.keys), valueShape = try stateShape(function, names.values)
    var states = InferenceFunction.MutableViews()
    // a function may declare a prefix of the buffer (a smaller context rung); bind exactly what it declares
    states.insert(keys.mutableRawView().slice(at: keyShape.map { 0..<$0 }), for: names.keys)
    states.insert(values.mutableRawView().slice(at: valueShape.map { 0..<$0 }), for: names.values)
    return try await function.run(inputs: inputs, states: states)
}

func stateShape(_ function: InferenceFunction, _ name: String) throws -> [Int] {
    guard case .ndArray(let descriptor) = function.descriptor.stateDescriptor(of: name) else { throw AnemoneError.missingFunction("state \(name)") }
    return descriptor.shape
}

func stateStrides(_ function: InferenceFunction, _ name: String) throws -> [Int] {
    guard case .ndArray(let descriptor) = function.descriptor.stateDescriptor(of: name) else { throw AnemoneError.missingFunction("state \(name)") }
    return descriptor.preferredStrides
}
