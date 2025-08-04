import torch
import torch.nn as nn

# Create a bidirectional LSTM
lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

print("PyTorch Bidirectional LSTM weight structure:")
for name, param in lstm.named_parameters():
    print(f"  {name}: {list(param.shape)}")

# Show the gate ordering in PyTorch
print("\nPyTorch LSTM gate ordering in weight tensors:")
print("  Each weight has 4*hidden_size rows for gates in order:")
print("  1. Input gate")
print("  2. Forget gate") 
print("  3. Cell gate (g or candidate)")
print("  4. Output gate")

# Check if weights are stacked
print(f"\nweight_ih_l0 shape: {list(lstm.weight_ih_l0.shape)}")
print(f"Expected for non-stacked: [4*256, 512] = [1024, 512]")
print(f"Actual matches expected: {list(lstm.weight_ih_l0.shape) == [1024, 512]}")
