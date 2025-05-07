#!/bin/bash

# Download the Kokoro model for testing weight loading
echo "Downloading Kokoro model from Hugging Face..."

MODEL_URL="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth"
MODEL_FILE="kokoro-v1_0.pth"

if [ -f "$MODEL_FILE" ]; then
    echo "Model file already exists: $MODEL_FILE"
    echo "To re-download, delete it first."
else
    curl -L "$MODEL_URL" -o "$MODEL_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded to $MODEL_FILE"
        echo "Model size: $(du -h $MODEL_FILE | cut -f1)"
    else
        echo "Failed to download the model"
        exit 1
    fi
fi

# Run the test
echo -e "\nRunning weight loading test..."
cargo test -p ferrocarril-core --test weight_loading_kokoro_test -- --ignored --nocapture

if [ $? -eq 0 ]; then
    echo -e "\n✅ Weight loading test completed successfully!"
else
    echo -e "\n❌ Weight loading test failed!"
    exit 1
fi