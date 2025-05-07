# Ferrocarril G2P Integration Documentation

This document explains how the Grapheme-to-Phoneme (G2P) system is integrated into Ferrocarril's TTS pipeline.

## Architecture Overview

Ferrocarril uses a modular architecture where different components work together in the TTS pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ferrocarril TTS System                     │
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐  │
│  │           │    │           │    │           │    │        │  │
│  │  G2P      │───►│ Text      │───►│  Neural   │───►│  DSP   │  │
│  │ (Phonesis)│    │ Encoder   │    │  Decoder  │    │        │  │
│  │           │    │           │    │           │    │        │  │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The G2P component is responsible for converting input text to phonetic representations that the neural TTS model can process. This is now directly provided by the Phonesis library.

## Integration Components

### 1. Phonesis Library

The core G2P functionality is provided by the **Phonesis** library, which is now an obligate (required) dependency of Ferrocarril:

- Text normalization (numbers, symbols, abbreviations)
- Embedded pronunciation dictionary with 65,000+ English words
- Rule-based fallback for unknown words
- Support for multiple phonetic standards (ARPABET, IPA, etc.)
- Zero external dependencies, compatible with WASM

### 2. Direct Integration

Ferrocarril directly integrates Phonesis through the `PhonesisG2P` wrapper in `ferrocarril-core`:

- Provides a clean API for text-to-phoneme conversion within Ferrocarril
- Handles error conditions gracefully
- Maintains consistent phonetic representations
- Manages thread-safe access to the underlying G2P system

### 3. Core Integration

The G2P functionality is integrated into `FerroModel` as a core component:

- G2P is now an obligate dependency, not a feature flag
- The inference pipeline automatically converts text to phonemes
- Debugging and configuration options are available
- The component can be accessed directly for testing or development

## Usage

### Basic Usage

```rust
use ferrocarril_core::{FerroModel, Config};

// Load a model
let config = Config::from_json("config.json")?;
let model = FerroModel::load_binary("model_path", config)?;

// Convert text to audio (uses G2P internally)
let audio = model.infer("Hello, world!")?;
```

### Voice-Specific Inference

```rust
// Load a model and voice
let config = Config::from_json("config.json")?;
let model = FerroModel::load_binary("model_path", config)?;
let voice = model.load_voice("voice_name")?;

// Generate audio with specific voice
let audio = model.infer_with_voice("Hello, world!", &voice, 1.0)?;
```

### Direct G2P Access

For testing or development, you can access the G2P component directly:

```rust
use ferrocarril_core::PhonesisG2P;

// Create G2P converter
let g2p = PhonesisG2P::new("en-us")?;

// Convert text to phonemes
let phonemes = g2p.convert("Hello, world!")?;
```

## Configuration

### Feature Flags

- `weights`: Enables binary weight loading (enabled by default)

No feature flag is needed for G2P functionality, as it's now an obligate dependency.

### Environment Variables

Some environment variables may be available in the Phonesis implementation.

## Extension

### Adding New Languages

To add a new language to the G2P system:

1. Implement the `GraphemeToPhoneme` trait in Phonesis for the new language
2. Update the factory function in Phonesis
3. Add language-specific rules and dictionary entries to Phonesis
4. Update the `PhonesisG2P::new` method to support the new language code

Example:

```rust
// In Phonesis
pub struct FrenchG2P { /* ... */ }

impl GraphemeToPhoneme for FrenchG2P {
    // Implement required methods
}

// In ferrocarril-core
pub fn new(language: &str) -> Result<Self, FerroError> {
    match language {
        "en" | "en-us" | "en-gb" => {
            // English implementation
        },
        "fr" | "fr-fr" => {
            // French implementation
        },
        _ => Err(FerroError::new(format!("Unsupported language: {}", language))),
    }
}
```

## Performance Considerations

- **Memory Usage**: The embedded dictionary approach ensures consistent memory usage across all platforms
- **Caching**: Frequently used phrases are cached to improve performance
- **Initialization Time**: G2P component initializes instantly with no external file loading
- **Inference Overhead**: G2P conversion typically adds 5-10ms per sentence

## Troubleshooting

### Common Issues

1. **Incorrect Pronunciations**:
   - Check if the word exists in the dictionary
   - Try different fallback strategies
   - Add custom words to the embedded dictionary if necessary

2. **Slow Performance**:
   - Ensure caching is enabled
   - Pre-process long texts into smaller chunks
   - Consider batch processing for multiple inputs

3. **Memory Usage**:
   - Adjust cache size based on your use case
   - Use shared references to the G2P component when multiple instances are needed

### Diagnostic Tools

- Set `RUST_LOG=debug` to see detailed G2P conversion logs
- Use `phonesis::debug::dump_phonemes()` to inspect conversions
- Check test coverage with `cargo test`

## Future Work

- Multilingual support expansion
- Neural G2P for improved pronunciation of unknown words
- Custom vocabulary integration
- Improved caching and memory optimization
- Context-aware pronunciation selection