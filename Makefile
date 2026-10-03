# MLX has to be built by Xcode: SwiftPM's binaries lack MLX's Metal library.
DERIVED ?= .derivedData
PRODUCTS = $(DERIVED)/Build/Products/Release
TEST_PRODUCTS = $(DERIVED)/Build/Products/Debug
DESTINATION = 'platform=macOS,arch=arm64'

.PHONY: cli build-tests test videos format

cli:
	xcodebuild build -scheme omnishotcut -configuration Release \
		-destination $(DESTINATION) -derivedDataPath "$(DERIVED)" -quiet
	@echo "$(PRODUCTS)/omnishotcut"

build-tests:
	xcodebuild build-for-testing -scheme OmniShotCutMLX-Package \
		-destination $(DESTINATION) -derivedDataPath "$(DERIVED)" -quiet

# Without the videos only the stitching tests run; the others skip.
test: build-tests
	xcrun xctest "$(TEST_PRODUCTS)/OmniShotCutMLXTests.xctest"

# The Blender open movies the reference was made from, 388 MB.
videos:
	scripts/fetch-videos.sh videos

format:
	xcrun swift-format format --in-place --recursive Sources Tests Package.swift
