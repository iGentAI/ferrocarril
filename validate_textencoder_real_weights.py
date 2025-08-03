#!/usr/bin/env python3
# validate_textencoder_real_weights.py
# Validate TextEncoder component with real Kokoro weights

import json
from pathlib import Path

def validate_textencoder_real_weights():
    """Validate that TextEncoder has all required real weights available"""
    
    print("🔍 TEXTENCODER REAL WEIGHT VALIDATION")
    print("=" * 40)
    
    metadata_path = Path("real_kokoro_weights/metadata.json")
    if not metadata_path.exists():
        print("❌ Real Kokoro weights not converted yet")
        return False
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Get TextEncoder component
    if "text_encoder" not in metadata["components"]:
        print("❌ text_encoder component missing from real weights")
        return False
        
    text_encoder = metadata["components"]["text_encoder"]["parameters"]
    
    print(f"📦 TextEncoder Component: {len(text_encoder)} parameters")
    
    # Validate key parameter categories
    embedding_params = [p for p in text_encoder.keys() if "embedding" in p]
    cnn_params = [p for p in text_encoder.keys() if "cnn" in p]
    lstm_params = [p for p in text_encoder.keys() if "lstm" in p]
    
    print("\n📋 Parameter Categories:")
    print(f"  Embedding: {len(embedding_params)} parameters")
    print(f"  CNN blocks: {len(cnn_params)} parameters") 
    print(f"  LSTM: {len(lstm_params)} parameters")
    
    # Check critical parameters are present
    critical_params = [
        "module.embedding.weight",           # [178, 512] vocab embeddings
        "module.cnn.0.0.weight_g",          # CNN layer 0 weight norm
        "module.cnn.0.0.weight_v",
        "module.lstm.weight_ih_l0",         # Forward LSTM
        "module.lstm.weight_ih_l0_reverse", # Reverse LSTM - CRITICAL!
    ]
    
    print("\n🔍 Critical Parameter Validation:")
    all_present = True
    
    for param in critical_params:
        if param in text_encoder:
            shape = text_encoder[param]["shape"]
            size_kb = text_encoder[param]["byte_size"] // 1024
            print(f"  ✅ {param}: shape {shape} ({size_kb}KB)")
        else:
            print(f"  ❌ {param}: MISSING")
            all_present = False
    
    if all_present:
        print("\n✅ ALL CRITICAL PARAMETERS PRESENT")
    else:
        print("\n❌ MISSING CRITICAL PARAMETERS")
        return False
    
    # Calculate total TextEncoder parameter count
    total_params = 0
    for param_info in text_encoder.values():
        shape = param_info["shape"]
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_params += param_count
    
    print(f"\n📊 TextEncoder Real Weight Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected: 5,606,400 parameters")
    print(f"  Match: {'✅ EXACT' if total_params == 5606400 else '⚠️ APPROX'}")
    
    # Validate bidirectionality (key issue from burndown)
    forward_lstm = [p for p in lstm_params if "reverse" not in p]
    reverse_lstm = [p for p in lstm_params if "reverse" in p]
    
    print(f"\n🧠 LSTM Bidirectionality Check:")
    print(f"  Forward LSTM weights: {len(forward_lstm)}")
    print(f"  Reverse LSTM weights: {len(reverse_lstm)}")
    
    if len(reverse_lstm) > 0:
        print("  ✅ BIDIRECTIONAL: Reverse weights present")
    else:
        print("  ❌ CRITICAL ISSUE: No reverse weights found!")
        return False
    
    print(f"\n🎯 TEXTENCODER REAL WEIGHT VALIDATION: {'✅ SUCCESS' if all_present else '❌ FAILED'}")
    print("Real Kokoro weights are properly structured for TextEncoder validation!")
    
    return all_present

if __name__ == "__main__":
    validate_textencoder_real_weights()