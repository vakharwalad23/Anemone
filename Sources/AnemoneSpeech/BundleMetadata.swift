import CoreAISpeech
import Foundation

/// The `metadata.json` Apple's Parakeet exporter writes next to the three graphs.
struct BundleMetadata: Decodable {
    struct Assets: Decodable {
        let encoder: String
        let decoderStep: String
        let joint: String
        /// Encoder layers from the split on, when the encoder is two programs.
        let encoderTail: String?
    }

    struct Config: Decodable {
        struct Encoder: Decodable {
            let numMelBins: Int
            let subsamplingFactor: Int
        }

        let vocabSize: Int
        let blankTokenId: Int32
        let decoderHiddenSize: Int
        let numDecoderLayers: Int
        let maxSymbolsPerStep: Int
        let durations: [Int]
        let encoder: Encoder
    }

    let assets: Assets
    let config: Config

    static func read(_ bundle: URL) throws -> BundleMetadata {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(BundleMetadata.self, from: Data(contentsOf: bundle.appending(path: "metadata.json")))
    }

    var tdtConfig: ParakeetTDTConfig {
        ParakeetTDTConfig(
            vocabSize: config.vocabSize, blankTokenId: config.blankTokenId, decoderHiddenSize: config.decoderHiddenSize,
            numDecoderLayers: config.numDecoderLayers, maxSymbolsPerStep: config.maxSymbolsPerStep, durations: config.durations,
            encoderNumMelBins: config.encoder.numMelBins, encoderSubsamplingFactor: config.encoder.subsamplingFactor)
    }
}
