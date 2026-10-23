import Testing

@testable import Anemone

@Suite struct StreamingTextTests {
    let table: [Int32: String] = [1: "Hel", 2: "lo", 3: "\u{FFFD}", 4: "é"]

    func decode(_ tokens: [Int32]) -> String {
        var text = tokens.map { table[$0] ?? "" }.joined()
        if tokens.last != 3 { text = text.replacingOccurrences(of: "\u{FFFD}", with: "") }
        return text
    }

    @Test func emitsOnlyTheNewText() {
        var stream = StreamingText()
        #expect(stream.append([1], decode: decode) == "Hel")
        #expect(stream.append([2], decode: decode) == "lo")
    }

    @Test func holdsBackAnUnfinishedCharacter() {
        var stream = StreamingText()
        #expect(stream.append([1, 3], decode: decode) == "")
        #expect(stream.append([4], decode: decode) == "Helé")
    }
}
