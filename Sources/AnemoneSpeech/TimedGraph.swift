import Anemone
import ArgumentParser
import CoreAI
import CoreML
import Foundation

enum Unit: String, ExpressibleByArgument, Encodable {
    case ane, gpu, cpu

    var coreAI: ComputeUnitKind {
        switch self {
        case .ane: .neuralEngine
        case .gpu: .gpu
        case .cpu: .cpu
        }
    }

    var coreML: MLComputeUnits {
        switch self {
        case .ane: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        case .cpu: .cpuOnly
        }
    }
}

/// One loaded graph whose main function can be called with fixed inputs.
struct TimedGraph {
    let path: String
    let loadSeconds: Double
    let offNeuralEngine: [String]?
    let call: () async throws -> Void

    /// `path` is a graph, or `graph.aimodel#function` for one function of a multi-function program.
    static func load(_ path: String, on unit: Unit) async throws -> TimedGraph {
        let parts = path.split(separator: "#", maxSplits: 1).map(String.init)
        let url = URL(fileURLWithPath: parts[0]), function = parts.count > 1 ? parts[1] : "main"
        let start = ContinuousClock.now
        let isCoreML = url.pathExtension == "mlmodelc"
        let call = isCoreML ? try await coreMLCall(url, on: unit) : try await coreAICall(url, function: function, on: unit)
        let off = !isCoreML && unit == .ane ? functionsOffNeuralEngine(url, functions: [function]) : nil
        return TimedGraph(path: path, loadSeconds: milliseconds(ContinuousClock.now - start) / 1e3, offNeuralEngine: off, call: call)
    }
}

private func coreAICall(_ url: URL, function name: String, on unit: Unit) async throws -> () async throws -> Void {
    let (model, function) = try await loadProgram(url, on: unit.coreAI) { model in
        guard let function = try model.loadFunction(named: name) else { throw ValidationError("\(url.path) has no function \(name); it has \(model.functionNames)") }
        return (model, function)
    }
    var inputs: [String: NDArray] = [:]
    for name in function.descriptor.inputNames { inputs[name] = try coreAIInput(function, name) }
    return {
        _ = model
        _ = try await function.run(inputs: inputs)
    }
}

/// A zeroed input; an attention mask is all true, as for a window filled with real audio.
private func coreAIInput(_ function: InferenceFunction, _ name: String) throws -> NDArray {
    guard case .ndArray(let descriptor) = function.descriptor.inputDescriptor(of: name) else { throw ValidationError("input \(name) is not an array") }
    var array = NDArray(descriptor: descriptor)
    if name == "attention_mask" { fillMask(&array, validFrames: descriptor.shape.reduce(1, *)) }
    return array
}

private func coreMLCall(_ url: URL, on unit: Unit) async throws -> () async throws -> Void {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = unit.coreML
    let model = try await MLModel.load(contentsOf: url, configuration: configuration)
    let inputs = try MLDictionaryFeatureProvider(dictionary: try coreMLInputs(model.modelDescription))
    return { _ = try model.prediction(from: inputs) }
}

/// Zeroed arrays; a one-element int input is a length (FluidAudio's `mel_length`) and gets the whole window.
private func coreMLInputs(_ description: MLModelDescription) throws -> [String: MLMultiArray] {
    let constraints = description.inputDescriptionsByName.compactMapValues(\.multiArrayConstraint)
    let frames = constraints.values.compactMap { $0.shape.last?.intValue }.max() ?? 0
    return try constraints.mapValues { constraint in
        let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
        array.withUnsafeMutableBytes { bytes, _ in _ = bytes.initializeMemory(as: UInt8.self, repeating: 0) }
        if constraint.dataType == .int32 && array.count == 1 { array[0] = NSNumber(value: frames) }
        return array
    }
}
