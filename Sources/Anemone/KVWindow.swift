/// One ANE call: the tokens to run, where their KV is written, and which rows are new.
struct TargetCall: Equatable {
    let offset: Int
    let width: Int
    let batch: [Int32]
    let newRows: Range<Int>
}

/// Tracks which tokens the ANE KV cache holds and plans calls that respect its write alignment.
struct KVWindow {
    // ANE KV-cache writes fail unless the write offset is a multiple of 4, so calls replay up to 3 earlier tokens (lab e05)
    static let writeAlign = 4

    let context: Int
    let widths: [Int]
    private(set) var history: [Int32] = []
    /// History entries whose KV sits in its own slot; a side branch accepted from a tree is in history but not in place.
    private(set) var valid = 0

    init(context: Int, widths: [Int]) {
        precondition(!widths.isEmpty, "a target needs at least one call width")
        self.context = context
        self.widths = widths.sorted()
    }

    var offset: Int {
        valid - valid % Self.writeAlign
    }

    /// Tokens the next call re-runs before the new ones.
    var replay: Int {
        history.count - offset
    }

    /// Most new tokens one call can take after the replayed ones.
    var capacity: Int {
        widths[widths.count - 1] - replay
    }

    func plan(_ tokens: [Int32]) throws -> TargetCall {
        let batch = Array(history[offset...]) + tokens
        guard let width = widths.first(where: { $0 >= batch.count }) else {
            throw AnemoneError.batchTooWide(tokens: batch.count, widest: widths[widths.count - 1])
        }
        guard offset + width <= context else { throw AnemoneError.contextFull(context: context) }
        return TargetCall(offset: offset, width: width, batch: batch, newRows: (batch.count - tokens.count)..<batch.count)
    }

    mutating func commit(_ tokens: [Int32]) {
        history += tokens
        valid = history.count
    }

    /// Appends `tokens`, of which only the first `inPlace` had their KV written in their own slots.
    mutating func accept(_ tokens: [Int32], inPlace: Int) {
        valid = history.count + inPlace
        history += tokens
    }

    mutating func rewind(_ count: Int) {
        history.removeLast(count)
        valid = min(valid, history.count)
    }

    mutating func reset() {
        history = []
        valid = 0
    }
}
