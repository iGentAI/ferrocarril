# Ferrocarril - Fast Neural Network Inference in Rust

A high-performance text-to-speech (TTS) system implemented in pure Rust, based on the Kokoro/StyleTTS2 architecture. Ferrocarril provides fast neural inference for generating natural-sounding speech from text.

## Features

- **Pure Rust Implementation**: No Python dependencies required for inference
- **Kokoro/StyleTTS2 Architecture**: State-of-the-art TTS quality with 81.8M parameters
- **Multiple Voices**: Support for 54 different voice styles
- **Real-time Performance**: Optimized for fast CPU inference
- **Simple API**: Easy-to-use interface for text-to-speech synthesis

## Quick Start

```rust
use ferrocarril_tts::FerroModel;

// Load the model
let mut model = FerroModel::from_weights_dir("./ferrocarril_weights", "./config.json")?;

// Synthesize speech
let audio = model.synthesize("Hello world", "af_heart", 1.0)?;

// Audio is f32 samples at 24kHz
println!("Generated {} audio samples", audio.len());
```

## Installation

Add Ferrocarril to your `Cargo.toml`:

```toml
[dependencies]
ferrocarril-tts = { path = "path/to/ferrocarril/ferrocarril-tts", features = ["weights"] }
```

## Model Setup

1. **Download Kokoro weights** from Hugging Face:
   ```bash
   # Download the Kokoro model (about 320MB)
   git clone https://huggingface.co/hexgrad/Kokoro-82M
   ```

2. **Convert weights** to Ferrocarril format:
   ```bash
   cd ferrocarril
   python weight_converter.py \
     --torch-model path/to/Kokoro-82M/kokoro-v1_0.pth \
     --torch-voices path/to/Kokoro-82M/voices \
     --config path/to/Kokoro-82M/config.json \
     --output-dir ./ferrocarril_weights
   ```

3. **Run inference**:
   ```rust
   cargo run --example tts_synthesis --features weights
   ```

## Architecture

Ferrocarril implements the complete Kokoro TTS pipeline:

- **G2P Processing**: IPA phoneme conversion using Phonesis
- **BERT Text Encoding**: CustomAlbert with specialized attention
- **Prosody Prediction**: Style-conditioned duration and F0 modeling  
- **Neural Vocoding**: iSTFT-based high-quality audio generation

### Components

- `ferrocarril-core`: Core tensor operations and weight management
- `ferrocarril-nn`: Neural network layers (LSTM, Conv1d, Attention)
- `ferrocarril-tts`: Complete TTS pipeline and API

## API Reference

### FerroModel

The main API interface for text-to-speech synthesis:

```rust
// Load model from weights
let model = FerroModel::from_weights_dir(weights_dir, config_path)?;

// Synthesize single text
let audio = model.synthesize(text, voice, speed)?;

// Batch synthesis
let results = model.synthesize_batch(&texts, voice, speed)?;

// List available voices
let voices = model.list_voices()?;

// Get model information
let (name, sample_rate, params) = model.info();
```

### Parameters

- `text`: Input text to synthesize
- `voice`: Voice name (e.g., "af_heart", "af_bella", "af_sarah")
- `speed`: Speech rate (0.5 to 2.0, where 1.0 is normal speed)

## Examples

See the `examples/` directory for complete usage examples:

- `tts_synthesis.rs`: Basic text-to-speech synthesis with WAV output
- More examples coming soon!

## Performance

Ferrocarril is optimized for CPU inference with:
- Efficient tensor operations
- Specialized LSTM implementations  
- Optimized convolution layers
- Smart weight loading and caching

## Development Status

✅ **Completed**:
- Full Kokoro model architecture
- Weight conversion from PyTorch
- G2P and text processing  
- Complete inference pipeline
- Multi-voice support

🚧 **Future Work**:
- SIMD optimizations
- Streaming inference
- Additional voice styles
- Fine-tuning capabilities
