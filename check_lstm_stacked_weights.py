import torch
import torch.nn as nn

# Create multi-layer LSTM to understand stacked weights
print("=== Multi-layer LSTM weight structure ===")
lstm_multi = nn.LSTM(512, 256, num_layers=2, batch_first=True)

print("\nMulti-layer LSTM parameters:")
for name, param in lstm_multi.named_parameters():
    print(f"  {name}: {list(param.shape)}")

# Now let's look at the internal state_dict to understand storage
print("\n=== PyTorch internal storage ===")
state = lstm_multi.state_dict()
for key, value in state.items():
    print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")

# Check if PyTorch internally stacks weights for efficiency
print("\n=== Checking for 'stacked' weights ===")
# In some versions, PyTorch may flatten/stack weights for efficiency
print("Note: PyTorch sometimes stacks LSTM weights for CUDA efficiency")
print("This means multiple weight matrices can be concatenated into a single tensor")

# Create a simple model and save it to see the storage format
print("\n=== Saving and examining weight storage ===")
test_lstm = nn.LSTM(10, 20, batch_first=True, bidirectional=True)
torch.save(test_lstm.state_dict(), 'test_lstm_weights.pth')

# Load and inspect
loaded = torch.load('test_lstm_weights.pth')
for k, v in loaded.items():
    print(f"  Saved weight '{k}': shape {list(v.shape)}")
    
# Check packed parameters (if they exist)
print("\n=== Checking for _flat_weights (packed parameters) ===")
if hasattr(test_lstm, '_flat_weights'):
    print(f"Number of flat weights: {len(test_lstm._flat_weights)}")
    for i, w in enumerate(test_lstm._flat_weights):
        if w is not None:
            print(f"  Flat weight {i}: shape {list(w.shape)}")
else:
    print("No _flat_weights found (normal in newer PyTorch versions)")
