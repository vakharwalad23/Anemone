import Testing

@testable import Anemone

@Suite struct DecodingTests {
    @Test func acceptsUpToTheFirstMismatch() {
        #expect(acceptedPrefix([5, 6, 7, 8], [5, 6, 9, 8, 1]) == 2)
        #expect(acceptedPrefix([5, 6], [5, 6, 3]) == 2)
        #expect(acceptedPrefix([5], [4, 5]) == 0)
    }

    @Test func stopsAtEndOfText() {
        var tokens: [Int32] = [1, 2, 9, 3, 4]
        #expect(trimAtStop(&tokens, eos: [9], limit: 10))
        #expect(tokens == [1, 2, 9])
    }

    @Test func keepsGoingWithoutStopOrLimit() {
        var tokens: [Int32] = [1, 2, 3]
        #expect(!trimAtStop(&tokens, eos: [9], limit: 10))
        #expect(tokens == [1, 2, 3])
    }

    @Test func stopsAtTheLimit() {
        var tokens: [Int32] = [1, 2, 3, 4]
        #expect(trimAtStop(&tokens, eos: [9], limit: 3))
        #expect(tokens == [1, 2, 3])
    }
}
