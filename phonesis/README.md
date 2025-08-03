
<div align="center">
  <h3>A Pure Rust Grapheme-to-Phoneme (G2P) Conversion Library</h3>
  
  <p>
    <a href="https://crates.io/crates/phonesis"><img src="https://img.shields.io/crates/v/phonesis.svg" alt="Crates.io"></a>
    <a href="https://docs.rs/phonesis"><img src="https://docs.rs/phonesis/badge.svg" alt="Documentation"></a>
    <a href="https://github.com/phonesis/phonesis/actions"><img src="https://github.com/phonesis/phonesis/workflows/CI/badge.svg" alt="Build Status"></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
  </p>
</div>

## Overview

Phonesis is a zero-dependency Rust library for converting text (graphemes) to phonetic representations (phonemes). It provides high-quality grapheme-to-phoneme conversion for English (with future plans for additional languages) while maintaining high performance and minimal memory footprint.

Phonesis is designed to work as both a standalone library and as an integrated component in text-to-speech systems such as [Ferrocarril](https://github.com/ferrocarril/ferrocarril).

## Features

- **Pure Rust implementation** with zero external dependencies
- **Embedded pronunciation dictionary** with 65,000+ English words (derived from WikiPron in ARPABET format)
- **Multi-standard phoneme output** with high-quality ARPABET-to-IPA conversion for TTS compatibility
- **WebAssembly (WASM) compatible** for browser-based applications
- **Dictionary-based conversion** with rule-based fallbacks
- **TTS-optimized output** including Kokoro/misaki-compatible IPA character streams
- **Efficient memory usage** with optimized data structures
- **Extensible architecture** for adding support for additional languages
- **Context-aware pronunciation** decisions
- **Comprehensive text normalization** for numbers, symbols, etc.

## Installation

Add Phonesis to your `Cargo.toml`:

```toml
[dependencies]
phonesis = "0.1.0"
```

Or use cargo-edit:

```bash
cargo add phonesis
```

## Quick Start

```rust
use phonesis::{GraphemeToPhoneme, english::EnglishG2P, PhonemeStandard};

fn main() {
    // Create an English G2P converter
    let g2p = EnglishG2P::new().expect("Failed to initialize G2P");
    
    // Convert text to phonemes
    let result = g2p.convert("Hello, world!").unwrap();
    
    // Print phonemes
    for phoneme in result {
        println!("{}", phoneme);
    }
    
    // Convert to a different standard
    let ipa_result = g2p.convert_to_standard("Hello", PhonemeStandard::IPA).unwrap();
    println!("IPA: {}", ipa_result.join(" "));
}
```

## Phoneme Standards Support

### ARPABET (Core Dictionary Format)
Phonesis uses ARPABET as its core dictionary format, providing:
- 65,000+ word pronunciations from WikiPron/CMU sources
- Stress notation: `EH1` (primary), `EH2` (secondary), `EH0` (unstressed)
- Multi-character phoneme symbols: `HH`, `TH`, `SH`, `NG`, etc.

### IPA (TTS Integration)
High-quality ARPABET-to-IPA conversion for TTS systems:
- **Vowel Mapping**: `AE` → `æ`, `EH` → `ɛ`, `IH` → `ɪ`, `UW` → `u`
- **Consonant Mapping**: `HH` → `h`, `TH` → `θ`, `DH` → `ð`, `NG` → `ŋ`  
- **Stress Conversion**: `EH1` → `ˈɛ`, `EH2` → `ˌɛ`, `EH0` → `ɛ`
- **TTS Compatibility**: Produces single-character IPA streams like "h ɛ l o w ɹ l d"

### Conversion Quality
The ARPABET-to-IPA conversion maintains:
- ✅ **All phonemic contrasts** (no loss of linguistic precision)
- ✅ **Stress information** (critical for prosody prediction)  
- ✅ **Dictionary completeness** (65,000+ entries → comprehensive IPA coverage)
- ✅ **TTS accuracy** (all converted IPA characters validated against target vocabularies)

## Usage Examples

### Basic Usage

```rust
use phonesis::{GraphemeToPhoneme, english::EnglishG2P};

let g2p = EnglishG2P::new().unwrap();
let phonemes = g2p.convert("pronunciation").unwrap();
println!("{}", phonemes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" "));
// Output: "P R AH0 N AH2 N S IY0 EY1 SH AH0 N"
```

### Custom Configuration

```rust
use phonesis::{
    GraphemeToPhoneme, 
    english::EnglishG2P, 
    G2POptions, 
    FallbackStrategy, 
    PhonemeStandard
};

let options = G2POptions {
    handle_stress: true,
    default_standard: PhonemeStandard::IPA,
    fallback_strategy: FallbackStrategy::UseRules,
};

let g2p = EnglishG2P::with_options(options).unwrap();
let phonemes = g2p.convert("uncommon word").unwrap();
```

### Working with Phoneme Standards

```rust
use phonesis::{
    GraphemeToPhoneme, 
    english::EnglishG2P, 
    PhonemeStandard, 
    convert_phonemes
};

let g2p = EnglishG2P::new().unwrap();

// Default is ARPABET
let arpabet = g2p.convert("hello").unwrap();

// Convert to IPA
let ipa = convert_phonemes(&arpabet, PhonemeStandard::IPA);
println!("IPA: {}", ipa.join(" "));
// Output: "IPA: h ɛ ˈl oʊ"
```

## Architecture

Phonesis uses a multi-layer approach to G2P conversion:

1. **Text Normalization**: Converts numbers, symbols, etc. to words
2. **Tokenization**: Splits text into processable tokens
3. **Dictionary Lookup**: Looks up words in the embedded pronunciation dictionary (~65,000 words)
4. **Rule-based Fallback**: Applies linguistic rules for unknown words
5. **Phoneme Generation**: Creates standardized phoneme representations

The dictionary data is embedded directly in the library's source code, with no external files needed. This makes Phonesis completely self-contained and easy to use in any environment, including WebAssembly.

For more details on the dictionary implementation, see [DICTIONARY.md](DICTIONARY.md).

## Development

### Building from Source

```bash
git clone https://github.com/phonesis/phonesis.git
cd phonesis
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License 2.0](LICENSE-APACHE)

at your option.

## Acknowledgments

Phonesis draws inspiration from several existing G2P implementations:
- [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
- [Misaki G2P library](https://github.com/hexgrad/misaki)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)

## Citation

If you use Phonesis in your research, please cite:

```bibtex
@software{phonesis2025,
  author = {Phonesis Contributors},
  title = {Phonesis: A Pure Rust G2P Library},
  year = {2025},
  url = {https://github.com/phonesis/phonesis}
}
```

## Integration with Speech Systems

Phonesis is specifically designed to integrate with text-to-speech systems. The ARPABET-to-IPA conversion ensures compatibility with modern TTS models that expect IPA character streams.

### Kokoro TTS Compatibility
Phonesis provides optimized integration for Kokoro TTS:

```rust
use phonesis::{
    english::EnglishG2P, 
    PhonemeStandard, 
    G2POptions, 
    FallbackStrategy
};

// Create G2P configured for Kokoro TTS
let options = G2POptions {
    default_standard: PhonemeStandard::IPA,  // Essential for Kokoro
    fallback_strategy: FallbackStrategy::UseRules,
    ..Default::default()
};

let g2p = EnglishG2P::with_options(options).unwrap();

// Convert to Kokoro-compatible IPA
let ipa_phonemes = g2p.convert_to_standard("hello world", PhonemeStandard::IPA).unwrap();
// Output: ["h", "ɛ", "l", "o", "w", "ɹ", "l", "d"]
```

### Ferrocarril TTS Integration
For Ferrocarril TTS, use the specialized adapter:

```rust
use phonesis::{
    ferrocarril_adapter::{FerrocarrilG2PAdapter, create_ferrocarril_g2p},
    PhonemeStandard
};

// Method 1: Direct adapter creation
let mut adapter = FerrocarrilG2PAdapter::new_english().unwrap();
adapter.set_standard(PhonemeStandard::IPA);  // Critical for vocabulary compatibility

// Method 2: Factory function  
let mut adapter = create_ferrocarril_g2p("en-us").unwrap();
adapter.set_standard(PhonemeStandard::IPA);

// Convert text to TTS-ready phoneme strings
let phonemes = adapter.convert_for_tts("hello world").unwrap();
// Output: "h ɛ l o w ɹ l d" (space-separated IPA characters)
```