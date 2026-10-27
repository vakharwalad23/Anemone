import ArgumentParser
import CoreAISpeech
import Foundation

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Transcribe every clip of a corpus manifest with a Core AI Parakeet bundle.")

    @Option(help: "Compute unit for the encoder: ane, gpu or cpu.")
    var encoderUnit: Unit = .ane

    @Option(help: "Compute unit for decoder_step and joint: ane, gpu or cpu.")
    var decoderUnit: Unit = .cpu

    @Option(help: "Core AI Parakeet bundle folder made by speech/convert/export_orukeet.py.")
    var bundle: String

    @Option(help: "Corpus manifest.json: {\"clips\": [{\"id\", \"wav\", \"seconds\"}]}.")
    var manifest: String

    @Option(help: "Folder holding the manifest's wav files.")
    var audio: String

    @Option(help: "Hypotheses file to write: a list of {id, ms, seconds, text, tokens, encodeMs, decodeMs}.")
    var out: String

    func run() async throws {
        let recognizer = try await Recognizer.load(URL(fileURLWithPath: bundle), encoderOn: encoderUnit, decoderOn: decoderUnit)
        let clips = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifest))).clips
        var rows: [Hypothesis] = []
        for (index, clip) in clips.enumerated() {
            rows.append(await transcribe(clip, with: recognizer))
            if (index + 1) % 100 == 0 { log("\(index + 1)/\(clips.count) clips") }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        try encoder.encode(rows).write(to: URL(fileURLWithPath: out))
        log("Wrote \(rows.count) hypotheses, \(rows.filter { $0.error != nil }.count) failed, to \(out).")
    }

    func transcribe(_ clip: Manifest.Clip, with recognizer: Recognizer) async -> Hypothesis {
        do {
            let pcm = try MelSpectrogram.loadAndResample(URL(fileURLWithPath: audio).appending(path: clip.wav), targetSampleRate: recognizer.sampleRate)
            let start = ContinuousClock.now
            let transcript = try await recognizer.transcribe(pcm)
            let ms = milliseconds(ContinuousClock.now - start)
            return Hypothesis(
                id: clip.id, ms: ms, seconds: clip.seconds, text: transcript.text, tokens: transcript.tokens.map(Int.init),
                encodeMs: transcript.encodeMs, decodeMs: transcript.decodeMs, error: nil)
        } catch {
            return Hypothesis(id: clip.id, ms: 0, seconds: clip.seconds, text: "", tokens: [], encodeMs: nil, decodeMs: nil, error: "\(error)")
        }
    }
}
