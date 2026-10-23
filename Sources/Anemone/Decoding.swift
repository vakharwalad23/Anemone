/// Drops tokens after the first end-of-text and past the limit; true when generation is done.
func trimAtStop(_ tokens: inout [Int32], eos: Set<Int32>, limit: Int) -> Bool {
    if let end = tokens.firstIndex(where: eos.contains) {
        tokens.removeSubrange((end + 1)...)
    }
    if tokens.count > limit {
        tokens.removeSubrange(limit...)
    }
    return tokens.count >= limit || tokens.last.map(eos.contains) == true
}
