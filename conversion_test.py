#!/usr/bin/env python3
# conversion_test.py
# Create a small test PyTorch model, save it, and convert it using weight_converter.py

import torch
import os
import subprocess
import shutil
import json
import numpy as np

class SimpleLinear(torch.nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)

def create_test_model():
    """Create a simple test model and save it in PyTorch format"""
    # Create model
    model = SimpleLinear()
    
    # Set deterministic weights for testing
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(50, dtype=torch.float).reshape(5, 10) * 0.01)
        model.linear.bias.copy_(torch.arange(5, dtype=torch.float) * 0.1)
    
    # Create state dict format
    state_dict = {
        "linear": model.state_dict()
    }
    
    # Save model
    os.makedirs("test_data", exist_ok=True)
    torch.save(state_dict, "test_data/simple_model.pth")
    
    # Create a voice tensor
    voice_tensor = torch.randn(256, dtype=torch.float)
    torch.save(voice_tensor, "test_data/test_voice.pt")
    
    return model

def convert_test_model():
    """Convert the test model using weight_converter.py"""
    # Call the conversion script
    cmd = [
        "python", "weight_converter.py",
        "--model", "test_data/simple_model.pth",
        "--voice", "test_data/test_voice.pt",
        "--output", "test_data/converted"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion failed with error: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def verify_conversion():
    """Verify the converted model matches the original"""
    # Check if metadata.json exists
    metadata_path = "test_data/converted/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return False
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check components
    if "components" not in metadata:
        print("Missing 'components' in metadata")
        return False
    
    # Check if linear component exists
    if "linear" not in metadata["components"]:
        print("Missing 'linear' component in metadata")
        return False
    
    # Check parameters
    linear_params = metadata["components"]["linear"]["parameters"]
    if "linear.weight" not in linear_params or "linear.bias" not in linear_params:
        print("Missing linear.weight or linear.bias parameters in linear component")
        return False
    
    # Check weight shape
    weight_shape = linear_params["linear.weight"]["shape"]
    if weight_shape != [5, 10]:
        print(f"Incorrect weight shape: {weight_shape}, expected [5, 10]")
        return False
    
    # Verify weight values
    weight_file = os.path.join("test_data/converted", linear_params["linear.weight"]["file"])
    if not os.path.exists(weight_file):
        print(f"Weight file not found: {weight_file}")
        return False
    
    # Read binary weight data
    weight_data = np.fromfile(weight_file, dtype=np.float32).reshape(5, 10)
    expected_weight = np.arange(50, dtype=np.float32).reshape(5, 10) * 0.01
    
    if not np.allclose(weight_data, expected_weight, rtol=1e-5):
        print("Weight values don't match!")
        return False
    
    # Check bias values
    bias_file = os.path.join("test_data/converted", linear_params["linear.bias"]["file"])
    bias_data = np.fromfile(bias_file, dtype=np.float32)
    expected_bias = np.arange(5, dtype=np.float32) * 0.1
    
    if not np.allclose(bias_data, expected_bias, rtol=1e-5):
        print("Bias values don't match!")
        return False
    
    # Check voice file
    voice_metadata_file = "test_data/converted/test_voice.json"
    if not os.path.exists(voice_metadata_file):
        print(f"Voice metadata file not found: {voice_metadata_file}")
        return False
    
    print("Conversion successful! All values match.")
    return True

def main():
    # Clean up previous test data
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    
    print("Step 1: Creating test model...")
    model = create_test_model()
    
    print("Step 2: Converting test model...")
    if not convert_test_model():
        print("Conversion failed!")
        return
    
    print("Step 3: Verifying conversion...")
    if verify_conversion():
        print("\n✅ Test passed! The weight converter works correctly.")
    else:
        print("\n❌ Test failed! The conversion process has issues.")

if __name__ == "__main__":
    main()