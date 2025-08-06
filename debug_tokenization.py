import sys
sys.path.append('kokoro')
from kokoro.model import KModel

# Load PyTorch model
model = KModel()
print("PYTORCH TOKENIZATION:")
text = "Test"
print(f"Input text: '{text}'")

# PyTorch tokenization
phonemes = list(text.lower().replace(" ", ""))
input_ids = [0]  # BOS
for char in phonemes:
    token_id = model.vocab.get(char, 1)
    input_ids.append(token_id)
    print(f"  '{char}' -> {token_id}")
input_ids.append(0)  # EOS
print(f"PyTorch input_ids: {input_ids}")
print(f"Length: {len(input_ids)}, Unique: {len(set(input_ids))}, Range: {min(input_ids)}-{max(input_ids)}")
