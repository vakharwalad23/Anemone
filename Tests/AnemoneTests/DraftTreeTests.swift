import Testing

@testable import Anemone

@Suite struct DraftTreeTests {
    // branch values: depth 1 = 1 x 0.1, depth 2 = 0.9 x 0.5, depth 3 = 0.45 x 0.2
    let tree = DraftTree(golden: 10, chain: [11, 12, 13], second: [21, 22, 23], confidence: [0.9, 0.5, 0.8], budget: 2)

    @Test func spendsTheBudgetOnTheLikeliestMisses() {
        #expect(branchDepths(confidence: [0.9, 0.5, 0.8], budget: 2) == [1, 2])
        #expect(branchDepths(confidence: [0.9, 0.5, 0.8], budget: 1) == [2])
        #expect(branchDepths(confidence: [0.9, 0.5, 0.8], budget: 0) == [])
    }

    @Test func layoutPutsBranchesAfterTheChain() {
        #expect(tree.tokens == [10, 11, 12, 13, 21, 22])
        #expect(tree.parents == [-1, 0, 1, 2, 0, 1])
        #expect(tree.depths == [0, 1, 2, 3, 1, 2])
    }

    @Test func nodesSeeOnlyTheirAncestors() {
        #expect(tree.attends(node: 5, toNode: 1))
        #expect(tree.attends(node: 5, toNode: 0))
        #expect(!tree.attends(node: 5, toNode: 2))
        #expect(!tree.attends(node: 2, toNode: 5))
    }

    @Test func followsTheChainWhileTheTargetAgrees() {
        #expect(tree.acceptedPath(predicted: [11, 12, 13, 99, 0, 0]) == [0, 1, 2, 3])
    }

    @Test func switchesToASideBranchWhenItMatches() {
        #expect(tree.acceptedPath(predicted: [11, 22, 0, 0, 0, 77]) == [0, 1, 5])
        #expect(tree.acceptedPath(predicted: [21, 0, 0, 0, 0, 0]) == [0, 4])
    }

    @Test func stopsAtTheRootWhenNothingMatches() {
        #expect(tree.acceptedPath(predicted: [99, 0, 0, 0, 0, 0]) == [0])
    }
}

@Suite struct TreeMaskTests {
    // two cached tokens to replay at offset 4, then golden, a 2-token chain and a branch at depth 1
    let tree = DraftTree(golden: 10, chain: [11, 12], second: [21, 22], confidence: [0.5, 0.9], budget: 1)

    func visible(_ mask: [Float16], key: Int, row: Int) -> Bool {
        mask[key * 8 + row] == 0
    }

    @Test func branchSeesContextAndGoldenButNotTheChain() {
        let mask = treeMask(context: 16, width: 8, offset: 4, replay: 2, tree: tree)
        let branchRow = 2 + 3
        #expect(visible(mask, key: 3, row: branchRow))
        #expect(visible(mask, key: 6, row: branchRow))
        #expect(!visible(mask, key: 7, row: branchRow))
        #expect(visible(mask, key: 9, row: branchRow))
    }

    @Test func replayRowsStayCausalAndPaddingSeesItself() {
        let mask = treeMask(context: 16, width: 8, offset: 4, replay: 2, tree: tree)
        #expect(visible(mask, key: 4, row: 0) && !visible(mask, key: 5, row: 0))
        #expect(visible(mask, key: 11, row: 7) && !visible(mask, key: 10, row: 7))
    }

    @Test func nodesTakeTheirDepthAsPosition() {
        #expect(treePositions(width: 8, offset: 4, replay: 2, tree: tree) == [4, 5, 6, 7, 8, 7, 10, 11])
    }
}
