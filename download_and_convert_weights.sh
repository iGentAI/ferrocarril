#!/bin/bash
# download_and_convert_weights.sh
# Script to download and convert Kokoro model and voice files
#
# IMPORTANT: This script is ESSENTIAL for testing the functional correctness of the TTS components.
# The tests require real weights to verify that components are not just structurally correct,
# but actually transform data in meaningful ways.

set -e

REPO_ID="hexgrad/Kokoro-82M"
OUTPUT_DIR="ferrocarril_weights"
SCRIPT_PATH="weight_converter.py"

echo "======================================================================================"
echo "             Downloading and Converting Weights for Functional Testing"
echo "======================================================================================"
echo ""
echo "IMPORTANT: Proper testing requires REAL weights, not random initialization."
echo "All tests that include '_with_real_weights' in their name will fail without these weights."
echo ""

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch numpy huggingface_hub

# Make script executable
chmod +x $SCRIPT_PATH

# Download and convert
echo "Downloading and converting models and voices..."
python3 "$SCRIPT_PATH" --huggingface "$REPO_ID" --output "$OUTPUT_DIR"

# Verify the conversion was successful
if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/model/metadata.json" ]; then
    echo "Conversion successful!"
    echo ""
    echo "======================================================================================"
    echo "  Weight conversion complete. You can now run tests that require real weights."
    echo "  Use 'cargo test -- --nocapture test_with_real_weights' to run integration tests."
    echo "======================================================================================"

    # Check output directory
    echo "Output directory contents:"
    find "$OUTPUT_DIR" -type f | sort | head -n 20
    if [ "$(find "$OUTPUT_DIR" -type f | wc -l)" -gt 20 ]; then
        echo "... and more files"
    fi
else
    echo "Weight conversion failed. Please check the error messages above."
    exit 1
fi