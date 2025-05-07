# Phonesis Binary Dictionary Setup

This document explains how to set up and use the binary dictionary system in Phonesis.

## Overview

Phonesis uses an efficient binary dictionary format for storing and loading pronunciation data. This approach provides:

- Fast loading times
- Efficient memory usage
- Compact storage format
- Easy distribution

## Creating the Binary Dictionary

1. **Prerequisites:**
   - WikiPron data file (e.g., `eng_latn_us_broad.tsv`)
   - Python 3.x
   - The `process_wikipron.py` and `generate_binary_dictionary.py` scripts

2. **Generate the Binary Dictionary:**
   ```bash
   python generate_binary_dictionary.py \
     wikipron_data/eng_latn_us_broad.tsv \
     phonesis_data/data/en_us_dictionary.bin \
     --subset-size 75000
   ```

   This creates a binary file (~4MB) containing ~79,000 pronunciations for ~65,000 words.

## Using the Binary Dictionary

### Default Setup

The Phonesis library automatically looks for the binary dictionary in several locations:

1. Environment variable `PHONESIS_DICTIONARY_PATH`
2. `data/en_us_dictionary.bin`
3. `phonesis_data/en_us_dictionary.bin`
4. `../phonesis_data/en_us_dictionary.bin`
5. `/etc/phonesis/en_us_dictionary.bin`

If not found, it falls back to a minimal built-in dictionary.

### Custom Location

Set the environment variable to specify a custom location:

```bash
export PHONESIS_DICTIONARY_PATH=/path/to/your/dictionary.bin
```

### In Rust Code

```rust
use phonesis::english::get_default_dictionary;

// Load the dictionary
let dict = get_default_dictionary().expect("Failed to load dictionary");

// Look up pronunciations
if let Some(pronunciation) = dict.lookup("hello") {
    println!("Pronunciation: {}", pronunciation);
}
```

## Binary Dictionary Format

The binary format consists of:

1. **Header (16 bytes):**
   - Magic bytes: `PHON` (4 bytes)
   - Version: 1 (2 bytes)
   - Entry count (4 bytes)
   - Metadata offset (4 bytes)
   - Checksum (2 bytes)

2. **String Table:** Contains all unique phoneme symbols

3. **Trie Structure:** Compact trie for efficient word lookup

4. **Phoneme Sequences:** Pronunciations with stress markers

5. **Metadata:** JSON metadata including source, license, etc.

## Testing the Dictionary

Run the binary dictionary test:

```bash
cargo test --features binary-dictionary binary_dictionary_test
```

## License

The WikiPron data is available under CC0 1.0 Universal (Public Domain). The binary dictionary inherits this license.