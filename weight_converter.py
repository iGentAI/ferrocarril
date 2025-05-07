#!/usr/bin/env python3
# weight_converter.py
# Convert PyTorch model and voice weights to a binary format usable in Rust

import torch
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Union, Any

def convert_tensor_to_binary(tensor: torch.Tensor, output_path: Path):
    """Convert a single PyTorch tensor to a binary file"""
    # Convert to numpy and save as binary
    np_array = tensor.detach().cpu().numpy()
    np_array.tofile(output_path)
    
    # Return metadata about this tensor
    return {
        "shape": list(np_array.shape),
        "dtype": str(np_array.dtype),
        "byte_size": np_array.nbytes
    }

def convert_model_weights(input_path: str, output_dir: str, use_mmap: bool = True):
    """Convert PyTorch model weights to binary format"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the PyTorch model weights
    print(f"Loading PyTorch model from {input_path}...")
    try:
        if use_mmap:
            # Use map_location to load as CPU tensors
            state_dict = torch.load(input_path, map_location="cpu")
        else:
            # For machines with limited memory
            state_dict = {}
            checkpoint = torch.load(input_path, map_location="cpu")
            for component_name, component_dict in checkpoint.items():
                state_dict[component_name] = {}
                if isinstance(component_dict, dict):
                    for param_name, param in component_dict.items():
                        state_dict[component_name][param_name] = param
                else:
                    state_dict[component_name] = component_dict
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        # Try loading with pickle protocol workarounds
        print("Attempting to load with pickle protocol workarounds...")
        import pickle
        with open(input_path, 'rb') as f:
            # Use highest protocol available
            state_dict = pickle.load(f, encoding='bytes')
        
    # Create metadata structure
    metadata = {
        "format_version": "1.0",
        "original_file": input_path.name,
        "components": {}
    }
    
    # Process each component
    for component_name, component_dict in state_dict.items():
        print(f"Processing component: {component_name}")
        
        # Create component directory
        component_dir = output_dir / component_name
        component_dir.mkdir(exist_ok=True)
        
        # Initialize component metadata
        component_metadata = {
            "parameters": {}
        }
        
        # Process each parameter in this component
        if isinstance(component_dict, dict):
            for param_name, param in component_dict.items():
                # Create safe filename - Make sure this path handling matches our test expectations
                safe_name = param_name.replace(".", "_")
                output_file = component_dir / f"{safe_name}.bin"
                
                # Convert and save tensor
                print(f"  - {param_name} → {output_file.relative_to(output_dir)}")
                param_meta = convert_tensor_to_binary(param, output_file)
                
                # Add to metadata - Store parameters directly with proper naming
                component_metadata["parameters"][param_name] = {
                    "file": str(output_file.relative_to(output_dir)),
                    **param_meta
                }
        else:
            # Handle cases where component_dict is not a dict (e.g., a tensor)
            output_file = component_dir / "tensor.bin"
            print(f"  - {component_name} (tensor) → {output_file.relative_to(output_dir)}")
            param_meta = convert_tensor_to_binary(component_dict, output_file)
            
            component_metadata["parameters"]["tensor"] = {
                "file": str(output_file.relative_to(output_dir)),
                **param_meta
            }
        
        # Add component metadata
        metadata["components"][component_name] = component_metadata
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Conversion complete. Metadata saved to {metadata_file}")
    return metadata_file

def convert_voice_file(input_path: str, output_dir: str):
    """Convert a single voice file to binary format"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the voice file
    print(f"Loading voice from {input_path}...")
    try:
        voice_tensor = torch.load(input_path, map_location="cpu") 
    except Exception as e:
        print(f"Error loading voice file: {e}")
        # Try loading with pickle protocol workarounds
        print("Attempting to load with pickle protocol workarounds...")
        import pickle
        with open(input_path, 'rb') as f:
            voice_tensor = pickle.load(f, encoding='bytes')
    
    # Create output path
    voice_name = input_path.stem
    output_file = output_dir / f"{voice_name}.bin"
    
    # Convert and save
    voice_meta = convert_tensor_to_binary(voice_tensor, output_file)
    
    # Create metadata
    metadata = {
        "name": voice_name,
        "file": str(output_file.relative_to(output_dir)),
        **voice_meta
    }
    
    # Save metadata
    metadata_file = output_dir / f"{voice_name}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Voice conversion complete. Metadata saved to {metadata_file}")
    return metadata_file

def convert_voices_directory(input_dir: str, output_dir: str):
    """Convert all voice files in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .pt files
    voice_files = list(input_dir.glob("*.pt"))
    if not voice_files:
        print(f"No voice files found in {input_dir}")
        return
    
    print(f"Found {len(voice_files)} voice files")
    
    # Process each voice file
    voice_metadata = {}
    for voice_file in voice_files:
        voice_name = voice_file.stem
        voice_meta_file = convert_voice_file(voice_file, output_dir)
        
        # Load individual metadata
        with open(voice_meta_file, 'r') as f:
            voice_meta = json.load(f)
        
        voice_metadata[voice_name] = voice_meta
    
    # Save combined metadata
    metadata_file = output_dir / "voices.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "format_version": "1.0",
            "voices": voice_metadata
        }, f, indent=2)
    
    print(f"Converted {len(voice_files)} voices. Combined metadata saved to {metadata_file}")
    return metadata_file

def download_and_convert(repo_id: str, output_dir: str):
    """Download model and voices from HF and convert them"""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model filename
    if repo_id == 'hexgrad/Kokoro-82M':
        model_filename = 'kokoro-v1_0.pth'
    elif repo_id == 'hexgrad/Kokoro-82M-v1.1-zh':
        model_filename = 'kokoro-v1_1-zh.pth'
    else:
        model_filename = 'model.pth'  # Generic fallback
    
    # Download model file
    print(f"Downloading {model_filename} from {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    
    # Download config file
    print(f"Downloading config.json from {repo_id}...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    # Copy config file to output dir
    import shutil
    shutil.copy(config_path, output_dir / "config.json")
    
    # Convert model weights
    model_output_dir = output_dir / "model"
    convert_model_weights(model_path, model_output_dir)
    
    # Download and convert voices
    print(f"Listing files in {repo_id} repository...")
    all_files = list_repo_files(repo_id)
    voice_files = [f for f in all_files if f.startswith("voices/") and f.endswith(".pt")]
    
    if not voice_files:
        print("No voice files found in repository")
        return
    
    # Create voices directory
    voices_output_dir = output_dir / "voices"
    voices_output_dir.mkdir(exist_ok=True)
    
    # Process each voice file
    print(f"Found {len(voice_files)} voice files")
    voice_metadata = {}
    
    for voice_file in voice_files:
        voice_name = Path(voice_file).stem
        print(f"Downloading {voice_file}...")
        
        try:
            voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file)
            voice_meta_file = convert_voice_file(voice_path, voices_output_dir)
            
            # Load individual metadata
            with open(voice_meta_file, 'r') as f:
                voice_meta = json.load(f)
                
            voice_metadata[voice_name] = voice_meta
        except Exception as e:
            print(f"Error processing {voice_file}: {e}")
    
    # Save combined metadata
    metadata_file = voices_output_dir / "voices.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "format_version": "1.0", 
            "voices": voice_metadata
        }, f, indent=2)
    
    print(f"Conversion complete. Model and voices saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model weights to binary format for Rust")
    parser.add_argument("--model", type=str, help="Path to PyTorch model file (.pth)")
    parser.add_argument("--voice", type=str, help="Path to a single voice file (.pt)")
    parser.add_argument("--voices-dir", type=str, help="Path to directory containing voice files")
    parser.add_argument("--huggingface", type=str, help="HuggingFace repo ID to download and convert (e.g., hexgrad/Kokoro-82M)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--no-mmap", action="store_true", help="Disable memory mapping for lower memory usage")
    
    args = parser.parse_args()
    
    if args.huggingface:
        download_and_convert(args.huggingface, args.output)
    else:
        if args.model:
            convert_model_weights(args.model, args.output, not args.no_mmap)
        
        if args.voice:
            convert_voice_file(args.voice, args.output)
        
        if args.voices_dir:
            convert_voices_directory(args.voices_dir, args.output)
    
    if not (args.model or args.voice or args.voices_dir or args.huggingface):
        parser.print_help()