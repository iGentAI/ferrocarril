# Phonesis - Rust Grapheme-to-Phoneme Conversion

A high-performance, pure Rust implementation of grapheme-to-phoneme (G2P) conversion for text-to-speech systems.

## Overview

Phonesis provides accurate phoneme conversion for English text using:
- Comprehensive dictionary with over 100,000 word pronunciations
- Rule-based fallback for out-of-vocabulary words
- Text normalization (numbers, symbols, abbreviations)
- IPA and ARPABET output formats

## Features

- **Pure Rust**: No external dependencies
- **Fast**: Optimized trie-based dictionary lookups
- **Accurate**: Based on WikiPron and CMU dictionary data
- **Flexible**: Supports multiple phoneme output formats
- **Robust**: Handles edge cases and unknown words gracefully

## Usage

```rust
use phonesis::G2P;

let g2p = G2P::new();
let phonemes = g2p.convert("Hello world");
println!("{:?}", phonemes);
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.