// swift-tools-version: 6.0
import PackageDescription

let settings: [SwiftSetting] = [
    .enableExperimentalFeature("Lifetimes"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
]

let package = Package(
    name: "Anemone",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "Anemone", targets: ["Anemone"]),
        .executable(name: "anemone", targets: ["AnemoneCLI"]),
        .executable(name: "anemone-speech", targets: ["AnemoneSpeech"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.2.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.7.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", exact: "2.22.0"),
        .package(url: "https://github.com/apple/coreai-models", revision: "9e120659a5643ad77f09125ba75eb6c9291e6ef4"),
    ],
    targets: [
        .target(
            name: "Anemone",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: settings
        ),
        .executableTarget(
            name: "AnemoneCLI",
            dependencies: [
                "Anemone", .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            swiftSettings: settings
        ),
        .executableTarget(
            name: "AnemoneSpeech",
            dependencies: [
                .product(name: "CoreAISpeech", package: "coreai-models"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: settings
        ),
        .testTarget(name: "AnemoneTests", dependencies: ["Anemone"], swiftSettings: settings),
    ]
)
