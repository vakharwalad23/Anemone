struct Manifest: Decodable {
    struct Clip: Decodable {
        let id: String
        let wav: String
        let seconds: Double
    }

    let clips: [Clip]
}

struct Hypothesis: Encodable {
    let id: String
    let ms: Double
    let seconds: Double
    let text: String
    let tokens: [Int]
    let error: String?
}
