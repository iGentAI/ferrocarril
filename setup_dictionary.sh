#!/bin/bash
# Script to download and set up the Phonesis dictionary

set -e

# Default paths
DICTIONARY_PATH="${PHONESIS_DICTIONARY_PATH:-phonesis_data/data/en_us_dictionary.bin}"
WIKIPRON_URL="https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/eng_latn_us_broad.tsv"
TEMP_DIR=$(mktemp -d)

echo "Phonesis Dictionary Setup"
echo "========================"

# Check if dictionary already exists
if [ -f "$DICTIONARY_PATH" ]; then
    echo "Dictionary already exists at: $DICTIONARY_PATH"
    read -p "Do you want to replace it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing dictionary."
        exit 0
    fi
fi

# Create directory if needed
mkdir -p "$(dirname "$DICTIONARY_PATH")"

# Download WikiPron data
echo "Downloading WikiPron data..."
curl -L -o "$TEMP_DIR/eng_latn_us_broad.tsv" "$WIKIPRON_URL"

# Check if Python scripts exist
if [ ! -f "process_wikipron.py" ] || [ ! -f "generate_binary_dictionary.py" ]; then
    echo "Error: Python scripts not found in current directory"
    echo "Please run this script from the directory containing:"
    echo "  - process_wikipron.py"
    echo "  - generate_binary_dictionary.py"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Generate binary dictionary
echo "Converting to binary format..."
python generate_binary_dictionary.py \
    "$TEMP_DIR/eng_latn_us_broad.tsv" \
    "$DICTIONARY_PATH" \
    --subset-size 75000

# Clean up
rm -rf "$TEMP_DIR"

echo
echo "Dictionary successfully installed at: $DICTIONARY_PATH"
echo
echo "To use this dictionary, either:"
echo "1. Place your application in the same directory structure, or"
echo "2. Set the environment variable:"
echo "   export PHONESIS_DICTIONARY_PATH=\"$(realpath "$DICTIONARY_PATH")\""
echo