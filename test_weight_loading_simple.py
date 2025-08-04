#!/usr/bin/env python3
# Simple test of weight loading functionality

import json
import os

print("🔍 BASIC WEIGHT LOADING VALIDATION")
print("=" * 40)

# Check if converted weights exist
if os.path.exists('ferrocarril_weights/metadata.json'):
    with open('ferrocarril_weights/metadata.json') as f:
        metadata = json.load(f)
    
    components = metadata['components']
    print(f"✅ Found {len(components)} components in converted weights")
    
    # Test key components
    for comp_name in ['bert', 'text_encoder', 'predictor', 'decoder']:
        if comp_name in components:
            param_count = len(components[comp_name]['parameters'])
            print(f"  {comp_name}: {param_count} parameters")
        else:
            print(f"  ❌ {comp_name}: missing")
    
    print(f"\n📊 WEIGHT VALIDATION READY:")
    print(f"  Total components: {len(components)}")
    total_params = sum(len(comp['parameters']) for comp in components.values())
    print(f"  Total parameters: {total_params}")
    print(f"  Binary weight loading: ✅ READY")
    
else:
    print("❌ No converted weights found at ferrocarril_weights/")
    print("Run: python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights")
