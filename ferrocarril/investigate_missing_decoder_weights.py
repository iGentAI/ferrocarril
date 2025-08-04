#!/usr/bin/env python3
"""
Critical Investigation: Missing Decoder Parameters
Compare Original Kokoro vs Our Conversion to find the missing 116 parameters
"""

import json
import torch
import sys
import os

# Add kokoro to path
sys.path.append('kokoro')
from kokoro.model import KModel

def main():
    print("=== CRITICAL INVESTIGATION: MISSING 116 DECODER PARAMETERS ===")
    
    # Load original PyTorch model
    print("Loading original Kokoro model...")
    model = KModel('hexgrad/Kokoro-82M')
    original_state_dict = model.state_dict()
    
    # Get all decoder parameters from original model
    original_decoder_params = {}
    for key, tensor in original_state_dict.items():
        if key.startswith('decoder.'):
            original_decoder_params[key] = {
                'shape': list(tensor.shape),
                'numel': tensor.numel(),
                'dtype': str(tensor.dtype)
            }
    
    print(f"Original decoder parameters: {len(original_decoder_params)}")
    
    # Load our converted metadata
    print("Loading our converted metadata...")
    with open('ferrocarril_weights/model/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    our_decoder_params = metadata['components']['decoder']['parameters']
    print(f"Our converted decoder parameters: {len(our_decoder_params)}")
    
    # Find missing parameters
    print("\n=== MISSING PARAMETER ANALYSIS ===")
    
    # Convert our parameter names to PyTorch format for comparison
    our_param_names = set()
    for param_name in our_decoder_params.keys():
        # Convert metadata names back to PyTorch format if needed
        pytorch_name = param_name.replace('module.', 'decoder.')
        our_param_names.add(pytorch_name)
    
    original_param_names = set(original_decoder_params.keys())
    
    # Find what we're missing
    missing_params = original_param_names - our_param_names
    extra_params = our_param_names - original_param_names
    
    print(f"Missing parameters: {len(missing_params)}")
    print(f"Extra parameters: {len(extra_params)}")
    
    if missing_params:
        print("\n=== MISSING PARAMETERS (Critical Issue) ===")
        
        # Analyze missing components
        missing_by_component = {}
        for param in sorted(missing_params):
            component = param.split('.')[1] if len(param.split('.')) > 1 else 'unknown'
            if component not in missing_by_component:
                missing_by_component[component] = []
            missing_by_component[component].append(param)
        
        for component, params in missing_by_component.items():
            print(f"\n{component}: {len(params)} missing parameters")
            for param in params[:5]:  # Show first 5
                shape = original_decoder_params[param]['shape']
                numel = original_decoder_params[param]['numel']
                print(f"  {param}: {shape} ({numel:,} parameters)")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")
    
    if extra_params:
        print("\n=== EXTRA PARAMETERS (Unexpected) ===")
        for param in sorted(extra_params)[:10]:
            print(f"  {param}")
    
    # Parameter count verification
    print("\n=== PARAMETER COUNT VERIFICATION ===")
    original_total = sum(p['numel'] for p in original_decoder_params.values())
    
    our_total = 0
    for param_data in our_decoder_params.values():
        shape = param_data['shape']
        param_count = 1
        for dim in shape:
            param_count *= dim
        our_total += param_count
    
    print(f"Original decoder parameter count: {original_total:,}")
    print(f"Our converted parameter count: {our_total:,}")
    print(f"Missing parameter count: {original_total - our_total:,}")
    
    # Check specific components that might be missing
    print("\n=== COMPONENT COMPLETENESS CHECK ===")
    
    # Look for patterns in missing parameters
    if missing_params:
        # Generator components
        generator_missing = [p for p in missing_params if 'generator' in p]
        if generator_missing:
            print(f"Missing generator parameters: {len(generator_missing)}")
            
        # ResBlock components  
        resblock_missing = [p for p in missing_params if 'resblocks' in p]
        if resblock_missing:
            print(f"Missing resblock parameters: {len(resblock_missing)}")
            
        # Decoder blocks
        decode_missing = [p for p in missing_params if 'decode.' in p]
        if decode_missing:
            print(f"Missing decode block parameters: {len(decode_missing)}")
    
    # Show final verdict
    print("\n=== ROOT CAUSE ANALYSIS ===")
    if len(missing_params) == 0:
        print("✅ No missing parameters - issue is elsewhere")
    else:
        print(f"❌ CRITICAL: {len(missing_params)} parameters missing from conversion")
        print("This explains why decoder produces all-zero outputs and crashes")
        print("\nMost likely causes:")
        print("1. Weight converter bug - not processing all decoder components")
        print("2. Missing architecture components in our Rust implementation")
        print("3. Different parameter grouping between PyTorch and our conversion")

if __name__ == "__main__":
    main()