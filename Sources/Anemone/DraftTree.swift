/// A draft chain plus single-token side branches, laid out as the rows of one verify call.
///
/// Node 0 is the last accepted (golden) token, nodes 1...n the drafted chain, and every later node a drafter
/// second choice that replaces the chain token at its depth.
struct DraftTree: Equatable {
    var tokens: [Int32]
    var parents: [Int]
    var depths: [Int]

    /// The chain alone, then the side branches worth the `budget` spare rows, most valuable first.
    init(golden: Int32, chain: [Int32], second: [Int32], confidence: [Float], budget: Int) {
        tokens = [golden] + chain
        parents = [-1] + Array(0..<chain.count)
        depths = Array(0...chain.count)
        for depth in branchDepths(confidence: confidence, budget: budget) {
            tokens.append(second[depth - 1])
            parents.append(depth - 1)
            depths.append(depth)
        }
    }

    var count: Int { tokens.count }

    /// Whether `node` may attend to `other`: itself and its ancestors.
    func attends(node: Int, toNode other: Int) -> Bool {
        var current = node
        while current >= 0 {
            if current == other { return true }
            current = parents[current]
        }
        return false
    }

    /// Follows the target's choices from the root: the accepted nodes in order, and the node whose prediction comes free.
    func acceptedPath(predicted: [Int32]) -> [Int] {
        var path = [0]
        while let next = (1..<count).first(where: { parents[$0] == path[path.count - 1] && tokens[$0] == predicted[path[path.count - 1]] }) {
            path.append(next)
        }
        return path
    }
}

/// Depths (1-based) of the side branches to add: the chain reaches depth d with probability prod(c[0..<d-1]) and
/// misses there with 1 - c[d-1]; a second choice is worth that product.
func branchDepths(confidence: [Float], budget: Int) -> [Int] {
    guard budget > 0 else { return [] }
    var reach: Float = 1
    var values: [(depth: Int, value: Float)] = []
    for (index, c) in confidence.enumerated() {
        values.append((index + 1, reach * (1 - c)))
        reach *= c
    }
    return values.sorted { $0.value > $1.value }.prefix(budget).map(\.depth).sorted()
}

/// Additive mask `[1, context, 1, width]` for a call that replays `replay` cached tokens and then verifies `tree`.
func treeMask(context: Int, width: Int, offset: Int, replay: Int, tree: DraftTree) -> [Float16] {
    var mask = [Float16](repeating: -.infinity, count: context * width)
    let base = offset + replay
    for row in 0..<width {
        let node = row - replay
        for key in 0..<min(context, offset + width) {
            let visible: Bool
            if row < replay {
                visible = key <= offset + row
            } else if node < tree.count {
                visible = key < base || (key < base + tree.count && tree.attends(node: node, toNode: key - base))
            } else {
                // a fully masked query row turns into NaN on the ANE and poisons its KV slot (lab e10)
                visible = key == offset + row
            }
            if visible { mask[key * width + row] = 0 }
        }
    }
    return mask
}

/// Position ids for the same call: replayed rows in order, tree nodes by depth, padding rows after.
func treePositions(width: Int, offset: Int, replay: Int, tree: DraftTree) -> [UInt16] {
    (0..<width).map { row in
        let node = row - replay
        return UInt16(row >= replay && node < tree.count ? offset + replay + tree.depths[node] : offset + row)
    }
}
