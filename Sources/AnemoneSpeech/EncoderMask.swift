import CoreAI

/// Writes a `[1, frames]` attention mask, true or 1.0 for the first `validFrames` frames, in the bool or fp16 type the encoder declares.
func fillMask(_ array: inout NDArray, validFrames: Int) {
    let frames = array.shape.reduce(1, *)
    if array.scalarType == .float16 {
        var view = array.mutableView(as: Float16.self)
        view.copyElements(fromContentsOf: (0..<frames).map { $0 < validFrames ? Float16(1) : 0 })
    } else {
        var view = array.mutableView(as: Bool.self)
        view.copyElements(fromContentsOf: (0..<frames).map { $0 < validFrames })
    }
}
