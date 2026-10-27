import Testing

@testable import Anemone

@Suite struct NeuralEnginePlacementTests {
    // region names as Core AI writes them in a cache manifest (published 8B chunk, 2026-10-27)
    let regions = ["extend_1024_16_Bn61_I05alAFYS8Bwtqi7g_0_ANE_region_0_0", "extend_1024_64_f2fOFjg2L-Jh0S7J5aIleQ_1_ANE_region_0_0"]

    @Test func functionsWithARegionRunOnTheNeuralEngine() {
        #expect(functionsWithoutRegion(["extend_1024_16", "extend_1024_64"], regions: regions) == [])
    }

    @Test func aFunctionWithoutARegionIsReported() {
        #expect(functionsWithoutRegion(["extend_1024_16", "extend_4096_16"], regions: regions) == ["extend_4096_16"])
    }

    @Test func aRungIsNotMistakenForALongerOne() {
        #expect(functionsWithoutRegion(["extend_102_16"], regions: regions) == ["extend_102_16"])
    }
}
