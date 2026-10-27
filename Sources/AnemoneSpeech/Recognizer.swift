import Anemone
import ArgumentParser
import CoreAI
import CoreAISpeech
import Foundation
import Tokenizers

/// A Core AI Parakeet TDT bundle with its encoder and its two decoder graphs each on a chosen compute unit.
struct Recognizer {
    struct Transcript {
        let tokens: [Int32]
        let text: String
        let encodeMs: Double
        let decodeMs: Double
    }

    /// One static encoder function: `main`, or `main_w<frames>` of a program with several window lengths, and its tail program's.
    struct Window {
        let function: InferenceFunction
        let features: NDArrayDescriptor
        let mask: NDArrayDescriptor
        let tail: (function: InferenceFunction, mask: NDArrayDescriptor)?
        let mel: MelConfig
        let basis: MelSpectrogram.Basis

        /// Holds `samples` without cutting any off.
        func holds(_ samples: Int) -> Bool { samples / mel.hopLength < features.shape[1] }
    }

    private let models: [AIModel]
    /// Shortest first.
    private let windows: [Window]
    private let decoder: ParakeetTDTDecoder
    private let resources: DecoderResources
    private let tokenizer: any Tokenizer
    private let subsampling: Int

    static func load(_ bundle: URL, encoderOn encoderUnit: Unit, decoderOn decoderUnit: Unit) async throws -> Recognizer {
        let metadata = try BundleMetadata.read(bundle)
        let encoderModel = try await loadProgram(bundle.appending(path: metadata.assets.encoder), on: encoderUnit.coreAI) { $0 }
        var tailModel: AIModel?
        if let tail = metadata.assets.encoderTail { tailModel = try await loadProgram(bundle.appending(path: tail), on: encoderUnit.coreAI) { $0 } }
        let step = try await loadProgram(bundle.appending(path: metadata.assets.decoderStep), on: decoderUnit.coreAI) { $0 }
        let joint = try await loadProgram(bundle.appending(path: metadata.assets.joint), on: decoderUnit.coreAI) { $0 }
        let names = encoderModel.functionNames.contains("main") ? ["main"] : encoderModel.functionNames.filter { $0.hasPrefix("main_w") }
        let windows = try names.map { try window(encoderModel, tail: tailModel, $0) }.sorted { $0.features.shape[1] < $1.features.shape[1] }
        guard !windows.isEmpty else { throw ValidationError("\(bundle.path) has no encoder function main or main_w<frames>") }
        let config = metadata.tdtConfig
        return Recognizer(
            models: [encoderModel, step, joint] + (tailModel.map { [$0] } ?? []), windows: windows, decoder: try ParakeetTDTDecoder(decoderStep: step, joint: joint),
            resources: .parakeetTDT(decoderStep: step, joint: joint, config: config),
            tokenizer: try await AutoTokenizer.from(modelFolder: bundle.appending(path: "processor"), strict: false),
            subsampling: config.encoderSubsamplingFactor)
    }

    private static func window(_ model: AIModel, tail tailModel: AIModel?, _ name: String) throws -> Window {
        guard let function = try model.loadFunction(named: name), let descriptor = model.functionDescriptor(for: name),
            case .ndArray(let features) = descriptor.inputDescriptor(of: "input_features"),
            case .ndArray(let mask) = descriptor.inputDescriptor(of: "attention_mask")
        else { throw ValidationError("encoder function \(name) lacks input_features and attention_mask") }
        var tail: (InferenceFunction, NDArrayDescriptor)?
        if let tailModel {
            guard let tailFunction = try tailModel.loadFunction(named: name),
                case .ndArray(let tailMask) = tailModel.functionDescriptor(for: name)?.inputDescriptor(of: "attention_mask")
            else { throw ValidationError("encoder tail lacks function \(name) with an attention_mask") }
            tail = (tailFunction, tailMask)
        }
        let mel = staticParakeetMel(frames: features.shape[1])
        return Window(function: function, features: features, mask: mask, tail: tail, mel: mel, basis: MelSpectrogram.Basis(config: mel))
    }

    var sampleRate: Double { windows[0].mel.sampleRate }

    func transcribe(_ pcm: [Float]) async throws -> Transcript {
        let start = ContinuousClock.now
        // the longest window cuts off what it cannot hold, as a single static window does
        let window = windows.first { $0.holds(pcm.count) } ?? windows[windows.count - 1]
        var featureArray = NDArray(descriptor: window.features)
        fillFloats(&featureArray, MelSpectrogram.fromPCM(pcm, config: window.mel, basis: window.basis))
        let validMel = MelSpectrogram.validFrameCount(forPCMLength: pcm.count, config: window.mel)
        var maskArray = NDArray(descriptor: window.mask)
        fillMask(&maskArray, validFrames: validMel)
        var outputs = try await window.function.run(inputs: ["input_features": featureArray, "attention_mask": maskArray])
        guard var hidden = outputs.remove("encoder_hidden_states")?.ndArray else { throw ValidationError("encoder returned no encoder_hidden_states") }
        let validFrames = min(hidden.shape[1], encoderFrames(validMel))
        if let tail = window.tail {
            var tailMask = NDArray(descriptor: tail.mask)
            fillMask(&tailMask, validFrames: validFrames)
            var tailOutputs = try await tail.function.run(inputs: ["hidden_states": hidden, "attention_mask": tailMask])
            guard let output = tailOutputs.remove("encoder_hidden_states")?.ndArray else { throw ValidationError("encoder tail returned no encoder_hidden_states") }
            hidden = output
        }
        let encoded = ContinuousClock.now
        let (tokens, _) = try await decoder.decode(
            encoderOutput: hidden, encoderOutputShape: hidden.shape, validEncoderFrames: validFrames, resources: resources)
        let decoded = ContinuousClock.now
        let text = tokenizer.decode(tokens: tokens.map(Int.init)).trimmingCharacters(in: .whitespaces)
        return Transcript(tokens: tokens, text: text, encodeMs: milliseconds(encoded - start), decodeMs: milliseconds(decoded - encoded))
    }

    /// Encoder frames backed by `melFrames` real frames: each stride-2 stage maps `T` to `(T - 1) / 2 + 1`.
    private func encoderFrames(_ melFrames: Int) -> Int {
        var frames = melFrames, factor = subsampling
        while factor > 1, frames > 0 {
            frames = (frames - 1) / 2 + 1
            factor /= 2
        }
        return frames
    }
}

/// CoreAISpeech's Parakeet front end with the static window's mel frame count baked in.
private func staticParakeetMel(frames: Int) -> MelConfig {
    let p = MelConfig.parakeet
    return MelConfig(
        sampleRate: p.sampleRate, nFFT: p.nFFT, winLength: p.winLength, hopLength: p.hopLength, nMelBins: p.nMelBins, nFrames: frames,
        preemphasis: p.preemphasis, normalization: p.normalization, layout: p.layout, padMode: p.padMode, melScale: p.melScale,
        windowPeriodicity: p.windowPeriodicity)
}

private func fillFloats(_ array: inout NDArray, _ values: [Float]) {
    if array.scalarType == .float16 {
        var view = array.mutableView(as: Float16.self)
        view.copyElements(fromContentsOf: values.map(Float16.init))
    } else {
        var view = array.mutableView(as: Float.self)
        view.copyElements(fromContentsOf: values)
    }
}
