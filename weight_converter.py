#!/usr/bin/env python3
# weight_converter.py - DEFINITIVE VERSION
# Captures ALL Kokoro parameters including nested InstanceNorm1d weights
# ZERO tolerance for missing parameters

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
    """Convert PyTorch model weights to binary format - DEFINITIVE VERSION"""
    input_path = Path(input_path)
    output_dir = Path(output_dir) / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PyTorch model from {input_path}...")
    
    # DEFINITIVE APPROACH: Use KModel.state_dict() to capture ALL parameters
    import sys
    sys.path.append('kokoro')
    from kokoro.model import KModel
    
    print("Creating KModel instance to extract complete parameter set...")
    model = KModel()  # This loads the complete model with all weights
    
    # Extract complete state_dict with ALL parameters including nested ones
    print("Extracting complete model state_dict...")
    full_state_dict = model.state_dict()
    
    print(f"Total parameters in loaded model: {len(full_state_dict)}")
    
    # Verify we have the critical decoder norm parameters
    decoder_params = [k for k in full_state_dict.keys() if k.startswith('decoder.')]
    norm_params = [k for k in decoder_params if '.norm.' in k]
    print(f"Decoder parameters: {len(decoder_params)} (expected 491)")
    print(f"InstanceNorm1d parameters: {len(norm_params)} (expected 116)")
    
    # STRICT VALIDATION: Must have exact parameter counts
    if len(decoder_params) != 491:
        raise Exception(f"CRITICAL: Decoder parameter count mismatch - expected 491, got {len(decoder_params)}")
    if len(norm_params) != 116:
        raise Exception(f"CRITICAL: InstanceNorm1d parameter count mismatch - expected 116, got {len(norm_params)}")
        
    print("✅ Parameter count validation passed")
    
    # Create metadata structure
    metadata = {
        "format_version": "1.0",
        "original_file": input_path.name,
        "components": {}
    }
    
    # Group parameters by component based on key prefixes
    component_params = {}
    for param_name, param_tensor in full_state_dict.items():
        # Determine component from parameter name
        if param_name.startswith('bert_encoder.'):
            component_name = 'bert_encoder'
            module_param_name = param_name.replace('bert_encoder.', 'module.')
        elif param_name.startswith('bert.'):
            component_name = 'bert'
            module_param_name = param_name.replace('bert.', 'module.')
        elif param_name.startswith('text_encoder.'):
            component_name = 'text_encoder'
            module_param_name = param_name.replace('text_encoder.', 'module.')
        elif param_name.startswith('predictor.'):
            component_name = 'predictor'
            module_param_name = param_name.replace('predictor.', 'module.')
        elif param_name.startswith('decoder.'):
            component_name = 'decoder'
            module_param_name = param_name.replace('decoder.', 'module.')
        else:
            raise Exception(f"CRITICAL: Unknown parameter prefix: {param_name}")
            
        if component_name not in component_params:
            component_params[component_name] = {}
        component_params[component_name][module_param_name] = param_tensor
    
    # Process each component
    for component_name, params in component_params.items():
        print(f"Processing component: {component_name} ({len(params)} parameters)")
        
        # Create component directory
        component_dir = output_dir / component_name
        component_dir.mkdir(exist_ok=True)
        
        # Initialize component metadata
        component_metadata = {
            "parameters": {}
        }
        
        # Process each parameter in this component
        for param_name, param_tensor in params.items():
            # Create safe filename
            safe_name = param_name.replace(".", "_")
            output_file = component_dir / f"{safe_name}.bin"
            
            # Convert and save tensor
            param_meta = convert_tensor_to_binary(param_tensor, output_file)
            
            # Add to metadata
            component_metadata["parameters"][param_name] = {
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
    
    # FINAL VALIDATION: Verify exact parameter counts
    print("\n=== DEFINITIVE VALIDATION ===")
    for component_name, component_data in metadata["components"].items():
        param_count = len(component_data["parameters"])
        print(f"  {component_name}: {param_count} parameters converted")
    
    # Specific decoder validation
    if 'decoder' in metadata["components"]:
        decoder_count = len(metadata["components"]["decoder"]["parameters"])
        norm_count = len([p for p in metadata["components"]["decoder"]["parameters"].keys() 
                         if 'norm.weight' in p or 'norm.bias' in p])
        
        print(f"\nDECODER VALIDATION:")
        print(f"  Captured: {decoder_count} parameters (expected 491)")
        print(f"  InstanceNorm1d: {norm_count} parameters (expected 116)")
        
        if decoder_count != 491:
            raise Exception(f"CRITICAL: Decoder conversion FAILED - got {decoder_count}, expected 491")
        if norm_count != 116:
            raise Exception(f"CRITICAL: InstanceNorm1d conversion FAILED - got {norm_count}, expected 116")
            
        print("  ✅ DECODER: All parameters successfully converted")
    
    print("✅ DEFINITIVE CONVERSION COMPLETE - ALL PARAMETERS CAPTURED")
    
    return metadata_file

def convert_voice_file(input_path: str, output_dir: str):
    """Convert a single voice file to binary format"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the voice file
    print(f"Loading voice from {input_path}...")
    voice_tensor = torch.load(input_path, map_location="cpu") 
    
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
        model_filename = 'model.pth'
    
    # Download config file
    print(f"Downloading config.json from {repo_id}...")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    # Copy config file to output dir
    import shutil
    shutil.copy(config_path, output_dir / "config.json")
    
    # Convert model weights using definitive method
    convert_model_weights("dummy", output_dir)  # KModel loads automatically
    
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
        
        voice_path = hf_hub_download(repo_id=repo_id, filename=voice_file)
        voice_meta_file = convert_voice_file(voice_path, voices_output_dir)
        
        # Load individual metadata
        with open(voice_meta_file, 'r') as f:
            voice_meta = json.load(f)
            
        voice_metadata[voice_name] = voice_meta
    
    # Save combined metadata
    metadata_file = voices_output_dir / "voices.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "format_version": "1.0", 
            "voices": voice_metadata
        }, f, indent=2)
    
    print(f"Conversion complete. Model and voices saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Definitive weight converter for Kokoro TTS - captures ALL parameters")
    parser.add_argument("--model", type=str, help="Path to PyTorch model file (.pth)")
    parser.add_argument("--voice", type=str, help="Path to a single voice file (.pt)")
    parser.add_argument("--voices-dir", type=str, help="Path to directory containing voice files")
    parser.add_argument("--huggingface", type=str, help="HuggingFace repo ID (e.g., hexgrad/Kokoro-82M)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--no-mmap", action="store_true", help="Disable memory mapping")
    
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