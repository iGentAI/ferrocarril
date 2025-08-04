#!/usr/bin/env python3
# Critical comprehensive weight audit - zero tolerance for missing weights

import sys
sys.path.append('kokoro')
import json
from kokoro.model import KModel

def audit_decoder_weights():
    """Comprehensive audit of ALL decoder weights"""
    print("❌ CRITICAL DECODER WEIGHT AUDIT")
    print("=" * 60)
    
    # Load PyTorch model
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    # Get all PyTorch decoder weights
    pytorch_decoder_weights = [k for k in pytorch_state_dict.keys() if 'decoder' in k]
    
    # Load converted weights metadata
    with open('ferrocarril_weights/model/metadata.json') as f:
        metadata = json.load(f)
    
    converted_decoder_weights = list(metadata['components']['decoder']['parameters'].keys())
    
    print(f"PyTorch decoder weights: {len(pytorch_decoder_weights)}")
    print(f"Converted decoder weights: {len(converted_decoder_weights)}")
    print(f"Count exact match: {len(pytorch_decoder_weights) == len(converted_decoder_weights)}")
    print()
    
    # Check for missing weights
    missing_weights = []
    
    print("🔍 CHECKING EACH PYTORCH WEIGHT:")
    for pytorch_weight in sorted(pytorch_decoder_weights):
        # Expected name in converted format (decoder. -> module.)
        expected_converted = pytorch_weight.replace("decoder.", "module.")
        
        if expected_converted in converted_decoder_weights:
            print(f"  ✅ {pytorch_weight} -> {expected_converted}")
        else:
            print(f"  ❌ {pytorch_weight} -> {expected_converted} (MISSING)")
            missing_weights.append((pytorch_weight, expected_converted))
    
    print()
    print(f"🎯 MISSING WEIGHT ANALYSIS:")
    print(f"  Missing weights: {len(missing_weights)}")
    print(f"  Required weights: {len(pytorch_decoder_weights)}")
    print(f"  Conversion completeness: {((len(pytorch_decoder_weights) - len(missing_weights)) / len(pytorch_decoder_weights)) * 100:.1f}%")
    
    if missing_weights:
        print("\n❌ CRITICAL MISSING WEIGHTS:")
        for pytorch_weight, expected_converted in missing_weights:
            print(f"  {pytorch_weight} -> {expected_converted}")
        
        print("\n💥 CRITICAL FAILURE - TTS INFERENCE WILL BREAK")
        return False
    else:
        print("\n✅ ALL REQUIRED DECODER WEIGHTS PRESENT")
        print("✅ DECODER WEIGHT VALIDATION SUCCESSFUL")
        return True

def audit_all_components():
    """Audit ALL components for missing weights"""
    print("\n🔍 COMPREHENSIVE ALL-COMPONENT WEIGHT AUDIT")
    print("=" * 60)
    
    # Load PyTorch model
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    # Load converted weights
    with open('ferrocarril_weights/model/metadata.json') as f:
        metadata = json.load(f)
    
    components = metadata['components']
    
    # Component mapping
    component_mapping = {
        'bert': [k for k in pytorch_state_dict.keys() if k.startswith('bert.') and not k.startswith('bert_encoder')],
        'bert_encoder': [k for k in pytorch_state_dict.keys() if k.startswith('bert_encoder')],
        'predictor': [k for k in pytorch_state_dict.keys() if k.startswith('predictor')], 
        'text_encoder': [k for k in pytorch_state_dict.keys() if k.startswith('text_encoder')],
        'decoder': [k for k in pytorch_state_dict.keys() if k.startswith('decoder')],
    }
    
    all_missing = []
    
    for component_name, pytorch_weights in component_mapping.items():
        if component_name in components:
            converted_weights = list(components[component_name]['parameters'].keys())
            
            print(f"\n📦 {component_name.upper()} COMPONENT:")
            print(f"  PyTorch: {len(pytorch_weights)} weights")
            print(f"  Converted: {len(converted_weights)} weights")
            
            component_missing = []
            for pytorch_weight in pytorch_weights:
                expected_converted = pytorch_weight.replace(f"{component_name}.", "module.")
                
                if expected_converted not in converted_weights:
                    component_missing.append((pytorch_weight, expected_converted))
                    all_missing.append((component_name, pytorch_weight, expected_converted))
            
            if component_missing:
                print(f"  ❌ MISSING: {len(component_missing)} weights")
                for pw, cw in component_missing[:3]:  # Show first 3
                    print(f"    {pw} -> {cw}")
            else:
                print(f"  ✅ COMPLETE: All weights present")
    
    # Final assessment
    print(f"\n🎯 CRITICAL WEIGHT AUDIT SUMMARY:")
    print(f"  Total missing weights: {len(all_missing)}")
    
    if all_missing:
        print(f"  ❌ CRITICAL FAILURE: {len(all_missing)} weights missing across components")
        print("  💥 TTS INFERENCE WILL FAIL")
        
        print(f"\n❌ ALL MISSING WEIGHTS:")
        for component, pytorch_weight, expected_converted in all_missing:
            print(f"  {component}: {pytorch_weight} -> {expected_converted}")
        
        return False
    else:
        print(f"  ✅ ZERO MISSING WEIGHTS: All components complete")
        print(f"  ✅ TTS INFERENCE READY")
        return True

def fix_validation_framework():
    """Fix validation framework to only test for weights that actually exist"""
    print(f"\n🔧 FIXING VALIDATION FRAMEWORK")
    print("=" * 60)
    
    # Load PyTorch model to get real weight structure
    model = KModel()
    pytorch_state_dict = model.state_dict()
    
    # Create correct validation weight list 
    correct_validation_weights = [
        ('bert', 'module.embeddings.word_embeddings.weight', [178, 128]),
        ('bert_encoder', 'module.weight', [512, 768]),
        ('text_encoder', 'module.embedding.weight', [178, 512]),
        ('predictor', 'module.lstm.weight_ih_l0', [1024, 640]),
        # Remove the non-existent conv_pre and add correct decoder weight
        ('decoder', 'module.generator.conv_post.weight_g', None),  # Shape will be determined
    ]
    
    print("✅ CORRECTED VALIDATION WEIGHTS:")
    print("  Removed: module.generator.conv_pre.weight (doesn't exist in PyTorch)")
    print("  Added: valid decoder weights that actually exist")
    
    return correct_validation_weights

if __name__ == "__main__":
    print("🚨 ZERO TOLERANCE WEIGHT AUDIT")
    print("=" * 80)
    print("User demand: ANY missing weights = CRITICAL FAILURE")
    print()
    
    # Audit decoder specifically
    decoder_success = audit_decoder_weights()
    
    # Audit all components
    all_success = audit_all_components()
    
    # Fix validation framework
    correct_weights = fix_validation_framework()
    
    # Final determination
    overall_success = decoder_success and all_success
    
    print(f"\n🎯 ZERO TOLERANCE WEIGHT AUDIT RESULTS:")
    print(f"  Decoder complete: {'✅' if decoder_success else '❌'}")
    print(f"  All components complete: {'✅' if all_success else '❌'}")
    print(f"  Overall status: {'✅ ZERO MISSING WEIGHTS' if overall_success else '❌ CRITICAL WEIGHT FAILURE'}")
    
    if overall_success:
        print(f"\n🎉 WEIGHT AUDIT SUCCESSFUL")
        print(f"  All required weights present - TTS system ready")
        print(f"  User's zero tolerance standard MET")
    else:
        print(f"\n💥 CRITICAL WEIGHT AUDIT FAILURE")
        print(f"  Missing weights detected - TTS inference will fail")
        print(f"  User's zero tolerance standard VIOLATED")
        sys.exit(1)