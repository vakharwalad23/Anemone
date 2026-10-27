import ArgumentParser
import CoreAISpeech
import Foundation

@main
struct AnemoneSpeech: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anemone-speech",
        abstract: "Transcribe every clip of an oruk corpus manifest with a Core AI Parakeet bundle."
    )

    @Option(help: "Core AI Parakeet bundle folder made by speech/convert/export_orukeet.py.")
    var bundle: String

    @Option(help: "oruk corpus manifest.json.")
    var manifest: String

    @Option(help: "Folder holding the manifest's wav files.")
    var audio: String

    @Option(help: "Hypotheses file to write, in the oruk format plus tokens.")
    var out: String

    func run() async throws {
        let model = try await SpeechRecognitionModel(resourcesAt: URL(fileURLWithPath: bundle))
        let sampleRate = await model.sampleRate
        let clips = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifest))).clips
        var rows: [Hypothesis] = []
        for (index, clip) in clips.enumerated() {
            rows.append(await transcribe(clip, with: model, sampleRate: sampleRate))
            if (index + 1) % 100 == 0 { log("\(index + 1)/\(clips.count) clips") }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        try encoder.encode(rows).write(to: URL(fileURLWithPath: out))
        log("Wrote \(rows.count) hypotheses, \(rows.filter { $0.error != nil }.count) failed, to \(out).")
    }

    func transcribe(_ clip: Manifest.Clip, with model: SpeechRecognitionModel, sampleRate: Double) async -> Hypothesis {
        do {
            let pcm = try MelSpectrogram.loadAndResample(URL(fileURLWithPath: audio).appending(path: clip.wav), targetSampleRate: sampleRate)
            let start = ContinuousClock.now
            let capture = try await model.transcribeCapturingStages(pcm: pcm)
            let elapsed = ContinuousClock.now - start
            let ms = Double(elapsed.components.seconds) * 1e3 + Double(elapsed.components.attoseconds) / 1e15
            return Hypothesis(id: clip.id, ms: ms, seconds: clip.seconds, text: capture.text, tokens: capture.tokens.map(Int.init), error: nil)
        } catch {
            return Hypothesis(id: clip.id, ms: 0, seconds: clip.seconds, text: "", tokens: [], error: "\(error)")
        }
    }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}
