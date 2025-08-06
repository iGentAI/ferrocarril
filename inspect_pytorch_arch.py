import sys
sys.path.append('kokoro')
from kokoro.model import KModel

print("Loading PyTorch model to inspect architecture...")
model = KModel()

print("\nDECODER ARCHITECTURE:")
print(f"  Type: {type(model.decoder).__name__}")

for name, module in model.decoder.named_modules():
    if hasattr(module, 'channels') or hasattr(module, 'num_features'):
        channels = getattr(module, 'channels', getattr(module, 'num_features', 'unknown'))
        print(f"  {name}: {type(module).__name__} (channels={channels})")
    elif hasattr(module, 'style_dim') or hasattr(module, 'num_features'):
        features = getattr(module, 'style_dim', getattr(module, 'num_features', 'unknown'))
        print(f"  {name}: {type(module).__name__} (features={features})")
    elif name and hasattr(module, 'in_features'):
        print(f"  {name}: {type(module).__name__} (in={module.in_features}, out={module.out_features})")
    elif name and name.count('.') <= 3:  # Limit depth
        print(f"  {name}: {type(module).__name__}")
