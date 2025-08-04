# Check how Kokoro stores LSTM weights in the real model
import json
from pathlib import Path

print("=== Kokoro LSTM Weight Structure Analysis ===")

# Check if we have the metadata from converted weights
metadata_paths = [
    Path("ferrocarril_weights/model/metadata.json"),
    Path("../ferrocarril_weights/model/metadata.json"),
    Path("real_kokoro_weights/metadata.json")
]

metadata_path = None
for path in metadata_paths:
    if path.exists():
        metadata_path = path
        break

if not metadata_path:
    print("❌ No converted weights found. Need to run weight_converter.py first.")
else:
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\nLoaded metadata from: {metadata_path}")
    
    # Find all LSTM-related weights
    lstm_weights = {}
    
    for component_name, component in metadata["components"].items():
        for param_name, param_info in component["parameters"].items():
            if "lstm" in param_name.lower():
                key = f"{component_name}.{param_name}"
                lstm_weights[key] = param_info
    
    print(f"\nFound {len(lstm_weights)} LSTM-related parameters:")
    
    # Group by component
    by_component = {}
    for key, info in lstm_weights.items():
        comp = key.split(".")[0]
        if comp not in by_component:
            by_component[comp] = []
        by_component[comp].append((key, info))
    
    for comp, params in by_component.items():
        print(f"\n{comp}:")
        for key, info in sorted(params):
            print(f"  {key}: shape={info['shape']}")
    
    # Check for bidirectional patterns
    print("\n=== Bidirectional LSTM Analysis ===")
    forward_params = [k for k in lstm_weights if "reverse" not in k]
    reverse_params = [k for k in lstm_weights if "reverse" in k]
    
    print(f"Forward parameters: {len(forward_params)}")
    print(f"Reverse parameters: {len(reverse_params)}")
    
    if reverse_params:
        print("\n✅ BIDIRECTIONAL LSTMs detected!")
        print("\nReverse parameter examples:")
        for param in sorted(reverse_params)[:5]:
            print(f"  {param}: shape={lstm_weights[param]['shape']}")
    else:
        print("\n⚠️ No reverse parameters found - may be unidirectional LSTMs")
        
    # Check for stacked weights pattern
    print("\n=== Checking for 'stacked' weight patterns ===")
    ih_weights = [k for k in lstm_weights if "weight_ih" in k]
    print(f"\nInput-hidden weights found: {len(ih_weights)}")
    for w in sorted(ih_weights)[:5]:
        shape = lstm_weights[w]['shape']
        print(f"  {w}: shape={shape}")
        if len(shape) == 2 and shape[0] % 4 == 0:
            hidden_size = shape[0] // 4
            print(f"    → Appears to be 4*{hidden_size} = {shape[0]} (IFGO gates stacked)")
