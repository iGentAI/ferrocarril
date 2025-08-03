# ferrocarril
Ferrocarril TTS - A Rust implementation of the Kokoro text-to-speech system with proper tensor dimension flow

## Weight Management

Ferrocarril uses real production weights from the Kokoro-82M model. See [WEIGHT_MANAGEMENT.md](WEIGHT_MANAGEMENT.md) for the complete weight conversion and loading process.

**Quick Start:**
```bash
# Download and convert real Kokoro weights (one-time setup)
python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights

# Build and test with real weights
cargo build
cargo test
```

All validation and testing uses the real 81.8M parameter production model.
