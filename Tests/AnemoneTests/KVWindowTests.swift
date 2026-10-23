import Testing

@testable import Anemone

@Suite struct KVWindowTests {
    @Test func firstCallStartsAtZeroWithTheSmallestWidth() throws {
        let call = try KVWindow(context: 1024, widths: [64, 8, 16]).plan([1, 2, 3])
        #expect(call == TargetCall(offset: 0, width: 8, batch: [1, 2, 3], newRows: 0..<3))
    }

    @Test func replaysTokensSinceTheLastAlignedOffset() throws {
        var window = KVWindow(context: 1024, widths: [8, 16, 64])
        window.commit([1, 2, 3, 4, 5, 6])
        let call = try window.plan([7, 8, 9, 10, 11, 12, 13, 14])
        #expect(call.offset == 4)
        #expect(call.batch == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        #expect(call.width == 16)
        #expect(call.newRows == 2..<10)
    }

    @Test func rewindDropsRejectedDrafts() throws {
        var window = KVWindow(context: 1024, widths: [8, 16])
        window.commit([1, 2, 3, 4, 5, 6, 7, 8, 9])
        window.rewind(4)
        #expect(window.history == [1, 2, 3, 4, 5])
        #expect(window.offset == 4)
        #expect(window.capacity == 15)
    }

    @Test func refusesCallsPastTheContext() {
        var window = KVWindow(context: 16, widths: [8])
        window.commit(Array(repeating: 1, count: 12))
        #expect(throws: AnemoneError.self) { try window.plan([2]) }
    }

    @Test func refusesBatchesWiderThanTheWidestFunction() {
        #expect(throws: AnemoneError.self) { try KVWindow(context: 1024, widths: [8]).plan(Array(repeating: 1, count: 9)) }
    }
}
