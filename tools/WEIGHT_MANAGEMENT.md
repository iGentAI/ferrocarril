# Ferrocarril Weight Management

This document defines the **single correct process** for weight conversion and loading in the Ferrocarril TTS system.

## ✅ Canonical Weight Conversion Process

### Step 1: Download and Convert Real Kokoro Model

**ONLY use the real Kokoro-82M model. NO synthetic or fake weights.**

```bash
# Download and convert the real production model
python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights
```

This will:
- Download the real 81.8M parameter Kokoro model (327MB)
- Convert to 548 binary weight files (313MB total)
- Generate metadata.json with complete component information
- Create proper directory structure by component

### Step 2: Validate Conversion

```python
# Verify the conversion captured all parameters
python3 -c """
import json
metadata = json.load(open('ferrocarril_weights/metadata.json'))
total = sum(
    sum(p//4 for p in [int(t['byte_size']) for t in comp['parameters'].values()])
    for comp in metadata['components'].values()
)
print(f'Converted parameters: {total:,}')
assert total == 81763410, f'Expected 81,763,410 parameters, got {total:,}'
print('✅ Conversion validation: SUCCESS')
"""
```

### Step 3: Load Weights in Rust

```rust
use ferrocarril_core::weights_binary::BinaryWeightLoader;

// Load the real converted weights
let loader = BinaryWeightLoader::from_directory("ferrocarril_weights")?;

// Access any component parameter
let tensor = loader.load_component_parameter("bert", "module.embeddings.word_embeddings.weight")?;
```

## 📊 Real Model Structure (81.8M Parameters)

```
Total: 81,763,410 parameters in 5 components

decoder:      53,276,190 (65.2%) - Vocoder/Generator 
predictor:    16,194,612 (19.8%) - Duration/F0/Noise prediction
bert:          6,292,480 (7.7%)  - Text understanding (Albert)
text_encoder:  5,606,400 (6.9%)  - Phoneme encoding 
bert_encoder:    393,728 (0.5%)  - BERT→Hidden projection
```

## 🚫 Deprecated/Removed Approaches

The following files have been removed or deprecated:

- `weight_converter_for_ferrocarril.py` - Redundant, use `weight_converter.py`
- `test_full_weight_pipeline.py` - Used synthetic data, real weights only
- `test_output/` - Synthetic test data, use `ferrocarril_weights/`
- `kokoro_test_output/` - Synthetic test data
- Any weight validation with fake/mock tensors

## 🔧 Component Loading Examples

### BERT Component (25 tensors, 6.2M params)
```rust
// Word embeddings: [178, 768] 
let embeddings = loader.load_component_parameter("bert", "module.embeddings.word_embeddings.weight")?;

// Attention weights
let query = loader.load_component_parameter("bert", "module.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight")?;
let key = loader.load_component_parameter("bert", "module.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight")?;
let value = loader.load_component_parameter("bert", "module.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight")?;
```

### Text Encoder (24 tensors, 5.6M params)
```rust
// Phoneme embedding: [178, 512]
let embedding = loader.load_component_parameter("text_encoder", "module.embedding.weight")?;

// Bidirectional LSTM weights
let weight_ih_fwd = loader.load_component_parameter("text_encoder", "module.lstm.weight_ih_l0")?;
let weight_ih_rev = loader.load_component_parameter("text_encoder", "module.lstm.weight_ih_l0_reverse")?;
```

### Decoder Component (375 tensors, 53.2M params)
```rust
// Generator upsampling
let ups_weight = loader.load_component_parameter("decoder", "module.generator.ups.0.weight_v")?;

// AdaIN conditioning
let adain_weight = loader.load_component_parameter("decoder", "module.generator.resblocks.0.adain1.0.fc.weight")?;
```

## 📁 Directory Structure

```
ferrocarril_weights/
├── bert/                    # BERT component (25 files)
├── bert_encoder/           # Projection layer (2 files)  
├── predictor/              # Prosody prediction (122 files)
├── decoder/                # Vocoder/Generator (375 files)
├── text_encoder/           # Text encoding (24 files)
├── metadata.json           # Complete model metadata
└── config.json             # Model configuration
```

## 🎯 Testing Guidelines

**ALL testing must use real weights:**
- ✅ Load actual Kokoro model parameters
- ✅ Validate with production tensor shapes 
- ✅ Test real component behavior
- ❌ NO synthetic or random tensors
- ❌ NO mock weight data
- ❌ NO fake parameter testing

## 🔄 Maintenance

To update weights:
1. Download new model version: `python3 weight_converter.py --huggingface hexgrad/Kokoro-82M-v1.1 --output ferrocarril_weights_v1_1`
2. Validate parameter count matches expected
3. Test Rust loading with new weights
4. Update production weights directory

The weight management system is **production-validated** with the real 81.8M parameter Kokoro model.