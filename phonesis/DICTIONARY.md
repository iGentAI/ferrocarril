# Phonesis Embedded Dictionary

The Phonesis G2P library includes a comprehensive English pronunciation dictionary embedded directly in the source code. This approach provides several advantages:

- **No external dependencies**: The library is completely self-contained
- **Cross-platform compatibility**: Works on any Rust target, including WebAssembly (WASM)
- **Immediate availability**: No need to load dictionary files at runtime
- **Deterministic behavior**: Same dictionary across all environments

## Dictionary Source

The embedded dictionary is derived from [WikiPron](https://github.com/CUNY-CL/wikipron), a collection of pronunciation data extracted from Wiktionary. WikiPron data is available under the CC0 1.0 Universal (Public Domain) license, making it suitable for commercial and open-source use.

The raw WikiPron data has been processed to:
1. Convert IPA (International Phonetic Alphabet) to ARPABET format
2. Filter problematic entries
3. Include the most common ~65,000 English words

## For Developers

### Using the Dictionary

The dictionary is automatically loaded when you use the `EnglishG2P` system:

```rust
use phonesis::english::{EnglishG2P, get_default_dictionary};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The dictionary is automatically loaded
    let g2p = EnglishG2P::new();
    
    // Convert a word to phonemes
    let phonemes = g2p.word_to_phonemes("hello")?;
    println!("Phonemes: {}", phonemes);
    
    // Access the dictionary directly if needed
    let dict = get_default_dictionary()?;
    if let Some(pronunciation) = dict.lookup("world") {
        println!("Pronunciation of 'world': {}", pronunciation);
    }
    
    Ok(())
}
```

### For WebAssembly (WASM) Targets

The embedded dictionary is automatically included in WebAssembly builds. No special configuration is needed.

```rust
// This works identically in WASM environments
#[wasm_bindgen]
pub fn convert_to_phonemes(text: &str) -> String {
    let g2p = EnglishG2P::new();
    match g2p.word_to_phonemes(text) {
        Ok(phonemes) => phonemes.to_string(),
        Err(_) => "Error converting text".to_string(),
    }
}
```

### Dictionary Size

The embedded dictionary:
- Contains ~65,000 English words
- Takes up ~2.3MB in source code
- Results in a minimal increase in compiled binary size due to compression

### Customizing the Dictionary

If you need to customize the dictionary, you can:

1. Modify the dictionary generation script in the Phonesis repository
2. Generate a new dictionary file
3. Replace the embedded dictionary in your fork of the library

## Implementation Details

The dictionary is embedded as a constant string in the Rust source code using a simple format similar to the CMU Pronouncing Dictionary format:

```
WORD PHONEME1 PHONEME2 ...
ANOTHER AH0 N AH0 DH ER0
```

This format allows for efficient lookup while maintaining readability for developers who need to examine or modify the embedded data.