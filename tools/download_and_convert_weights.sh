#!/bin/bash
# download_and_convert_weights.sh
# THE CANONICAL SCRIPT for downloading and converting real Kokoro weights
#
# This script downloads the production Kokoro-82M model and converts it to
# the binary format used by Ferrocarril. This is the ONLY supported process.
#
# NO SYNTHETIC OR FAKE WEIGHTS - Real model weights only.

set -e

REPO_ID="hexgrad/Kokoro-82M"
OUTPUT_DIR="ferrocarril_weights"
SCRIPT_PATH="weight_converter.py"

echo "======================================================================================"
echo "             Downloading and Converting REAL Kokoro-82M Model Weights"
echo "======================================================================================"
echo ""
echo "This script downloads and converts the real production Kokoro-82M model:"
echo "- 81,763,410 parameters (81.8M)"
echo "- 5 components: bert, bert_encoder, predictor, decoder, text_encoder"  
echo "- 548 weight files in binary format"
echo "- 313MB total converted size"
echo ""

# Check if already converted
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/metadata.json" ]; then
    echo "⚠️  Ferrocarril weights already exist at: $OUTPUT_DIR"
    echo "Delete the directory to re-download, or use the existing weights."
    exit 0
fi

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

# Download and convert REAL model
echo "Downloading and converting REAL Kokoro-82M model..."
python3 "$SCRIPT_PATH" --huggingface "$REPO_ID" --output "$OUTPUT_DIR"

# Verify the conversion was successful
if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/metadata.json" ]; then
    echo "✅ REAL WEIGHT CONVERSION SUCCESSFUL!"
    echo ""
    echo "======================================================================================"
    echo "  Real Kokoro-82M weights ready at: $OUTPUT_DIR"
    echo "  Use these weights for ALL Ferrocarril testing and validation."
    echo "  No synthetic or fake weights should be used."
    echo "======================================================================================"

    # Verify parameter count
    echo "Validating converted weights..."
    python3 -c "
import json
metadata = json.load(open('$OUTPUT_DIR/metadata.json'))
total = sum(
    sum(
        int(param_info['shape'][0] if len(param_info['shape']) > 0 else 1) * 
        int(param_info['shape'][1] if len(param_info['shape']) > 1 else 1) *
        int(param_info['shape'][2] if len(param_info['shape']) > 2 else 1) *
        int(param_info['shape'][3] if len(param_info['shape']) > 3 else 1)
        for param_info in comp_data['parameters'].values()
    ) 
    for comp_data in metadata['components'].values()
)
print(f'Converted parameters: {total:,}')
if total == 81763410:
    print('✅ Parameter count validation: SUCCESS')
else:
    print(f'❌ Parameter count mismatch: expected 81,763,410, got {total:,}')
    exit 1
"
    
    if [ $? -eq 0 ]; then
        echo "✅ COMPLETE VALIDATION: SUCCESS"
        echo "Real Kokoro weights are ready for Ferrocarril!"
    fi
else
    echo "❌ Weight conversion failed. Check the error messages above."
    exit 1
fi