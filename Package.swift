// swift-tools-version: 6.0

import PackageDescription

// MLX code has to be built by Xcode, not SwiftPM: `swift build` and
// `swift test` produce binaries without MLX's compiled Metal library, and the
// first kernel launch fails with `Failed to load the default metallib`. See
// README.md and the Makefile.

let package = Package(
    name: "OmniShotCutMLX",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "OmniShotCutMLX", targets: ["OmniShotCutMLX"]),
        .executable(name: "omnishotcut", targets: ["OmniShotCutCLI"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.3")),
        // Only `Hub`, to download the converted weights.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "OmniShotCutMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "OmniShotCutCLI",
            dependencies: ["OmniShotCutMLX"],
            path: "Sources/omnishotcut",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "OmniShotCutMLXTests",
            dependencies: ["OmniShotCutMLX"],
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
