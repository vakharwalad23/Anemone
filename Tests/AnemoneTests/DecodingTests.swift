import Testing

@testable import Anemone

@Suite struct DecodingTests {
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
