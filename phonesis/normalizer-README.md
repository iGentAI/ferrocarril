# Phonesis Text Normalization Module

This directory contains the implementation of the text normalization module for the Phonesis G2P (Grapheme-to-Phoneme) library. Text normalization is a critical preprocessing step before G2P conversion, ensuring that non-standard text elements like numbers, symbols, and abbreviations are properly converted to their spoken forms.

## Components

### 1. Tokenizer (`tokenizer.rs`)

The tokenizer splits text into meaningful units for processing:

- Identifies words, numbers, symbols, punctuation, and whitespace
- Handles Unicode characters correctly
- Preserves case information and token positions
- Provides mechanisms for token recombination
- Detects alphanumeric tokens (e.g., "A1", "2nd")

```rust
// Example tokenization
let normalizer = TextNormalizer::new();
let tokens = normalizer.tokenize("Hello, world! 123");
// Tokens: "Hello", ",", "world", "!", "123"
```

### 2. Number Converter (`numbers.rs`)

Converts numeric values to their word representations:

- Cardinal numbers ("123" → "one hundred and twenty-three")
- Ordinal numbers ("3rd" → "third")
- Decimal numbers ("3.14" → "three point one four")
- Currency values ("$10.99" → "ten dollars and ninety-nine cents")
- Fractions ("1/2" → "one half")
- Ranges and math expressions ("1-5" → "one to five")

```rust
// Example number conversion
let converter = NumberConverter::new();
let words = converter.convert_cardinal(1234);
// Words: "one thousand two hundred and thirty-four"
```

### 3. Symbol Mapper (`symbols.rs`)

Expands symbols and abbreviations to their spoken forms:

- Punctuation marks ("?" → "question mark")
- Mathematical symbols ("+" → "plus")
- Currency symbols ("$" → "dollar")
- Special characters ("©" → "copyright")
- Abbreviations ("Dr." → "Doctor")
- Context-sensitive expansions ("-" → "to" or "dash" depending on context)

```rust
// Example symbol mapping
let mapper = SymbolMapper::new();
let expansion = mapper.convert_symbol("@");
// Expansion: "at"
```

### 4. Main Normalizer (`mod.rs`)

Combines the above components to provide complete text normalization:

- Orchestrates tokenization, number expansion, and symbol mapping
- Provides a simple API for text normalization
- Handles context-sensitive interpretations of tokens
- Configurable for different normalization styles

```rust
// Example full normalization
let normalizer = Normalizer::new();
let result = normalizer.normalize("Hello, world! I have $42.50.");
// Result: "Hello, world! I have forty-two dollars and fifty cents."
```

## Usage

```rust
use phonesis::normalizer::Normalizer;

// Create a normalizer with default settings
let normalizer = Normalizer::new();

// Normalize text for pronunciation
let text = "Dr. Smith has 123 books and lives at 42 Oak St.";
let normalized = normalizer.normalize(text).expect("Failed to normalize text");

println!("Original: {}", text);
println!("Normalized: {}", normalized);

// Output:
// Original: Dr. Smith has 123 books and lives at 42 Oak St.
// Normalized: Doctor Smith has one hundred and twenty-three books and lives at forty-two Oak Saint
```

## Customization

All components can be customized:

```rust
use phonesis::normalizer::{
    Normalizer, TextNormalizer, NumberConverter, 
    NumberConverterOptions, SymbolMapper, SymbolMapperOptions
};

// Custom number converter options
let number_options = NumberConverterOptions {
    use_and: false,
    use_hyphens: false,
    decimal_separator: "dot".to_string(),
    ..Default::default()
};

// Custom tokenizer
let tokenizer = TextNormalizer::with_options(true, false);

// Custom number converter
let number_converter = NumberConverter::with_options(number_options);

// Custom symbol mapper
let symbol_mapper = SymbolMapper::default();

// Create custom normalizer
let normalizer = Normalizer::with_components(
    tokenizer, 
    number_converter, 
    symbol_mapper
);
```

## Future Enhancements

1. Language-specific normalization rules
2. Domain-specific abbreviation sets
3. Performance optimizations for large text processing
4. Enhanced number handling (scientific notation, Roman numerals)
5. Support for more symbol types and Unicode ranges