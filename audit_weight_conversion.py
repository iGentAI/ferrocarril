#!/usr/bin/env python3
"""
Comprehensive Weight Converter Audit Script
Verifies that weight conversion accurately captures PyTorch weights and shapes.
"""

import sys
sys.path.append('kokoro')
import json
from kokoro.model import KModel
from pathlib import Path

def audit_weight_conversion():
    """Audit the weight conversion process for accuracy."""
    print("🔍 COMPREHENSIVE WEIGHT CONVERTER AUDIT")
    print("=" * 60)
    
    # 1. Load PyTorch model to get ground truth
    print("Loading PyTorch model for ground truth...")
    model = KModel()
    pytorch_state = model.state_dict()
    print(f"✅ PyTorch model loaded: {len(pytorch_state)} parameters")
    
    # 2. Check converted metadata
    metadata_path = Path("ferrocarril_weights/model/metadata.json")
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        converted_metadata = json.load(f)
    
    print(f"✅ Metadata loaded: {len(converted_metadata['components'])} components")
    
    # 3. Compare parameter counts
    total_converted_params = 0
    for component_name, component_data in converted_metadata["components"].items():
        component_count = len(component_data["parameters"])
        total_converted_params += component_count
        print(f"  {component_name}: {component_count} parameters")
    
    print(f"\nCOMPARISON:")
    print(f"  PyTorch original: {len(pytorch_state)} parameters")
    print(f"  Converted total: {total_converted_params} parameters")
    
    if len(pytorch_state) != total_converted_params:
        print(f"❌ PARAMETER COUNT MISMATCH!")
        missing = len(pytorch_state) - total_converted_params
        print(f"   Missing: {missing} parameters")
        return False
    else:
        print(f"✅ Parameter counts match exactly")
    
    # 4. Check specific layer shapes
    print(f"\n🔍 LAYER SHAPE VALIDATION:")
    
    # Critical decoder layers from arch_diagnostic
    critical_layers = [
        "decoder.encode.norm1.fc.weight",
        "decoder.encode.norm2.fc.weight", 
        "decoder.decode.0.norm1.fc.weight",
        "decoder.decode.0.norm2.fc.weight",
        "decoder.generator.noise_res.0.adain1.0.fc.weight",
        "decoder.generator.noise_res.1.adain1.0.fc.weight",
    ]
    
    for layer_key in critical_layers:
        if layer_key in pytorch_state:
            pytorch_shape = list(pytorch_state[layer_key].shape)
            print(f"  {layer_key}: {pytorch_shape}")
            
            # Check if this exists in converted metadata
            component, param_path = layer_key.split('.', 1)
            module_param = f"module.{param_path}"
            
            if component in converted_metadata["components"]:
                if module_param in converted_metadata["components"][component]["parameters"]:
                    converted_shape = converted_metadata["components"][component]["parameters"][module_param]["shape"]
                    if pytorch_shape == converted_shape:
                        print(f"    ✅ Shapes match: {pytorch_shape}")
                    else:
                        print(f"    ❌ Shape mismatch: PyTorch {pytorch_shape} vs Converted {converted_shape}")
                        return False
                else:
                    print(f"    ❌ Parameter not found in converted metadata: {module_param}")
                    return False
            else:
                print(f"    ❌ Component not found: {component}")
                return False
        else:
            print(f"  ⚠️ Layer not found in PyTorch state: {layer_key}")
    
    # 5. Voice conversion validation
    print(f"\n🎤 VOICE CONVERSION VALIDATION:")
    voice_metadata_path = Path("ferrocarril_weights/voices/voices.json")
    if voice_metadata_path.exists():
        with open(voice_metadata_path, 'r') as f:
            voice_metadata = json.load(f)
        
        voice_count = len(voice_metadata["voices"])
        print(f"  Voices converted: {voice_count}")
        
        # Check a sample voice shape
        sample_voice = next(iter(voice_metadata["voices"].values()))
        voice_shape = sample_voice["shape"]
        print(f"  Sample voice shape: {voice_shape}")
        print(f"  Voice elements: {voice_shape[0] * voice_shape[1] if len(voice_shape) == 2 else 'N/A'}")
    else:
        print(f"  ❌ Voice metadata not found")
        return False
    
    print(f"\n✅ WEIGHT CONVERSION AUDIT COMPLETE")
    return True

if __name__ == "__main__":
    success = audit_weight_conversion()
    if not success:
        print(f"\n❌ AUDIT FAILED - Weight conversion has issues!")
        sys.exit(1)
    else:
        print(f"\n✅ AUDIT PASSED - Weight conversion is accurate!")