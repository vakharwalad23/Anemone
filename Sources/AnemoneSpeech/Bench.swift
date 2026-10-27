import Anemone
import ArgumentParser
import Foundation

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Time the main function of Core AI graphs and compiled Core ML models on one compute unit, interleaved, as JSON."
    )

    @Argument(help: "Core AI .aimodel graphs (graph.aimodel#function for one function) or compiled Core ML .mlmodelc models.")
    var graphs: [String]

    @Option(help: "Compute unit: ane, gpu or cpu.")
    var unit: Unit = .ane

    @Option(help: "Timed calls per graph; the median is reported.")
    var runs = 20

    @Option(help: "Untimed calls per graph before timing.")
    var warmups = 5

    struct Report: Encodable {
        let machine: Machine
        let unit: Unit
        let runs: Int
        let warmups: Int
        let graphs: [Timing]
    }

    struct Timing: Encodable {
        let graph: String
        let loadSeconds: Double
        let offNeuralEngine: [String]?
        let medianMs: Double
        let p10Ms: Double
        let p90Ms: Double
    }

    func run() async throws {
        var timed: [TimedGraph] = []
        for path in graphs { timed.append(try await TimedGraph.load(path, on: unit)) }
        for graph in timed { for _ in 0..<warmups { try await graph.call() } }
        var samples = [[Double]](repeating: [], count: timed.count)
        for _ in 0..<runs {
            for (index, graph) in timed.enumerated() {
                let start = ContinuousClock.now
                try await graph.call()
                samples[index].append(milliseconds(ContinuousClock.now - start))
            }
        }
        let timings = zip(timed, samples).map { graph, times in
            let sorted = times.sorted()
            return Timing(
                graph: graph.path, loadSeconds: graph.loadSeconds, offNeuralEngine: graph.offNeuralEngine, medianMs: sorted[sorted.count / 2],
                p10Ms: sorted[sorted.count / 10], p90Ms: sorted[(sorted.count - 1) * 9 / 10])
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        let report = Report(machine: .current(), unit: unit, runs: runs, warmups: warmups, graphs: timings)
        print(String(decoding: try encoder.encode(report), as: UTF8.self))
    }
}
