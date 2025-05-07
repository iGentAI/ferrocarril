#!/bin/bash
# run_full_inference_test.sh
# Script to run the full inference pipeline test with proper flags

set -e

echo "=== Ferrocarril Full Inference Pipeline Test ==="
echo

# Check if weights have been downloaded
WEIGHTS_DIR="ferrocarril_weights"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "Weights not found in $WEIGHTS_DIR"
    echo "Downloading weights using download_and_convert_weights.sh..."
    ./download_and_convert_weights.sh
    
    # Exit if weights could not be downloaded
    if [ ! -d "$WEIGHTS_DIR" ]; then
        echo "Error: Failed to download weights. Please check download_and_convert_weights.sh"
        exit 1
    fi
fi

echo "Weights found in $WEIGHTS_DIR"
echo

# Create test output directory
mkdir -p test_output

# Navigate to ferrocarril directory
cd ferrocarril

# Make sure the weights are accessible to the test
# This is a critical step - the symbolic link must be correctly created
if [ ! -e "./ferrocarril_weights" ]; then
    echo "Creating symbolic link to weights directory..."
    ln -sf "../$WEIGHTS_DIR" ./ferrocarril_weights
    
    # Verify the link was created properly
    if [ ! -e "./ferrocarril_weights" ]; then
        echo "Error: Failed to create symbolic link to weights directory"
        exit 1
    fi
    
    # Verify the weight directories are accessible
    if [ ! -e "./ferrocarril_weights/model" ] || [ ! -e "./ferrocarril_weights/voices" ]; then
        echo "Error: Symbolic link created but weights directories are not accessible"
        ls -la "./ferrocarril_weights"
        exit 1
    fi
fi

echo "Verifying weight directory structure..."
ls -la "./ferrocarril_weights/"
echo "Verifying model directory structure..."
ls -la "./ferrocarril_weights/model/" | head -10

# Build and run the test with proper features
echo "Building and running the full inference test..."
RUST_BACKTRACE=1 cargo test --test full_inference_test --features weights -- --ignored --nocapture

echo
echo "=== Test Complete ==="
echo
echo "Check the test_output directory for generated audio files."

# List any generated audio files
echo "Generated audio files:"
ls -la ../test_output/*.wav 2>/dev/null || echo "No audio files were generated."