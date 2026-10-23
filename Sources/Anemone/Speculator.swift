import Foundation

public struct GenerationStats: Sendable, Codable {
    public var promptTokens = 0
    public var generatedTokens = 0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var cycles = 0
    public var acceptedDrafts = 0
    public var draftSeconds = 0.0
    public var verifySeconds = 0.0

    /// Decode speed after the first token, which comes from prefill (the lab's definition).
    public var tokensPerSecond: Double {
        decodeSeconds > 0 ? Double(generatedTokens - 1) / decodeSeconds : 0
    }

    public var acceptedPerCycle: Double {
        cycles > 0 ? Double(acceptedDrafts) / Double(cycles) : 0
    }
}

/// Greedy decoding with the ANE target, optionally drafted by the GPU drafter and verified in one call per cycle.
final class Speculator {
    let target: AneTarget
    let drafter: Drafter
    let eos: Set<Int32>

    init(target: AneTarget, drafter: Drafter, eos: Set<Int32>) {
        self.target = target
        self.drafter = drafter
        self.eos = eos
    }

    func generate(_ prompt: [Int32], maxTokens: Int, speculative: Bool, emit: ([Int32]) -> Void) async throws -> GenerationStats {
        target.reset()
        drafter.reset()
        var stats = GenerationStats(promptTokens: prompt.count)
        let prefillStart = ContinuousClock.now
        let prefill = try await target.forward(prompt)
        guard var golden = prefill.tokens.last else { return stats }
        var pending = prefill.taps
        if speculative {
            // the prompt goes into the drafter during prefill; its last 1 to 8 rows ride along with the first draft
            let split = prompt.count - (prompt.count - 1) % Drafter.tapRows - 1
            if split > 0 {
                _ = try await drafter.step(taps: Array(pending[..<(split * drafter.tapWidth)]), golden: golden)
            }
            pending = Array(pending[(split * drafter.tapWidth)...])
        }
        stats.prefillSeconds = seconds(since: prefillStart)
        var tokens = [golden]
        emit(tokens)
        let decodeStart = ContinuousClock.now
        while tokens.count < maxTokens && !eos.contains(golden) {
            var fresh: [Int32]
            if speculative {
                var began = ContinuousClock.now
                let draft = try await drafter.step(taps: pending, golden: golden)
                stats.draftSeconds += seconds(since: began)
                began = ContinuousClock.now
                let verify = try await target.forward([golden] + draft.tokens)
                stats.verifySeconds += seconds(since: began)
                let accepted = acceptedPrefix(draft.tokens, verify.tokens)
                target.rewind(draft.tokens.count - accepted)
                pending = Array(verify.taps[..<((accepted + 1) * drafter.tapWidth)])
                golden = verify.tokens[accepted]
                fresh = Array(draft.tokens[..<accepted]) + [golden]
                stats.cycles += 1
                stats.acceptedDrafts += accepted
            } else {
                golden = try await target.forward([golden]).tokens[0]
                fresh = [golden]
            }
            let before = tokens.count
            tokens += fresh
            let done = trimAtStop(&tokens, eos: eos, limit: maxTokens)
            if tokens.count > before {
                emit(Array(tokens[before...]))
            }
            if done { break }
        }
        stats.decodeSeconds = seconds(since: decodeStart)
        stats.generatedTokens = tokens.count
        return stats
    }
}

func seconds(since start: ContinuousClock.Instant) -> Double {
    let elapsed = ContinuousClock.now - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}
