
> Phonesis: A Pure Rust Grapheme-to-Phoneme (G2P) Library

**Version**: 0.1.0  
**Status**: In Development  
**Last Updated**: April 26, 2025

## 1. Introduction

Phonesis is a pure Rust, zero-dependency library for converting graphemes (written text) to phonemes (pronunciation units). This document outlines the design decisions, architecture, and implementation strategies for the library.

### 1.1 Motivation

Text-to-speech systems require accurate conversion from text to pronunciation. While several G2P libraries exist in other languages, the Rust ecosystem lacks a comprehensive, dependency-free solution. Phonesis aims to fill this gap, providing a high-quality G2P system that aligns with Rust's emphasis on performance, safety, and ergonomics.

### 1.2 Design Goals

1. **Zero External Dependencies**: Implement all functionality in pure Rust without relying on external libraries or bindings.
2. **High Performance**: Optimize for speed and memory efficiency, suitable for both desktop and embedded applications.
3. **Accuracy**: Provide accurate pronunciations for common words and reasonable approximations for uncommon ones.
4. **Extensibility**: Design the architecture to support multiple languages and phonetic standards.
5. **API Ergonomics**: Create an intuitive, Rust-idiomatic API that's easy to use correctly.

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Text Input     │ ──► │  Normalization  │ ──► │  Tokenization   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Postprocessing │ ◄── │  Rule Engine    │ ◄── │  Dictionary     │
│                 │     │                 │     │  Lookup         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│                 │
│  Phoneme Output │
│                 │
└─────────────────┘
```

### 2.2 Core Components

1. **Text Normalization**: Handles preprocessing including number expansion, symbol handling, and special cases.
2. **Tokenization**: Splits text into tokens (words, numbers, punctuation).
3. **Dictionary**: Provides efficient word lookup with a compressed trie.
4. **Rule Engine**: Applies linguistic rules for deriving pronunciations of unknown words.
5. **Phoneme Representation**: Manages phoneme data types and conversions between standards.

### 2.3 Data Flow

1. Input text is normalized (numbers expanded, symbols replaced, etc.)
2. Normalized text is tokenized into individual words
3. Each token is looked up in the pronunciation dictionary
4. If not found, rule-based prediction is attempted
5. The resulting phonetic representations are combined
6. Final outputs are generated in the requested format

## 3. Key Components

### 3.1 Phoneme Representation

The `Phoneme` struct represents a single phonetic unit:

```rust
pub struct Phoneme {
    pub symbol: String,
    pub stress: Option<StressLevel>,
    pub position: Option<PhonemePosition>,
    pub standard: PhonemeStandard,
    pub phoneme_type: PhonemeType,
}
```

Key features:
- Support for multiple standards (ARPABET, IPA, SAMPA, etc.)
- Stress information for vowels
- Positional data for context-aware processing
- Classification (vowel, consonant, etc.)

Optimizations:
- Compact binary representation for memory efficiency
- Bit-packed encoding of phoneme properties

### 3.2 Dictionary System

The dictionary uses a compact trie for efficient word lookup:

```rust
pub struct PronunciationDictionary {
    trie: CompactTrie,
    language: String,
    standard: PhonemeStandard,
    metadata: DictionaryMetadata,
}

struct CompactTrie {
    nodes: Vec<Node>,
    values: Vec<PhonemeSequence>,
}
```

Key features:
- Memory-efficient trie implementation
- Binary serialization format
- Support for case-insensitive lookup
- Fast prefix and exact matching
- Metadata for dictionary management

Optimizations:
- Integer indices instead of pointers
- Vector-based storage with contiguous memory
- Path compression
- Binary dictionary format

### 3.3 Rule Engine

The rule engine applies linguistic patterns to predict pronunciations:

```rust
pub struct RuleEngine {
    rules: Vec<ProductionRule>,
    patterns: PatternMatcher,
}

struct ProductionRule {
    pattern: GraphemePattern,
    output: PhonemeTemplate,
    context: RuleContext,
    priority: u32,
}
```

Key features:
- Pattern-based grapheme sequence matching
- Context-sensitive rule application
- Priority-based rule ordering
- Template-based output generation

Optimizations:
- Compiled rules for fast matching
- Early rule filtering
- Pattern precomputation

### 3.4 Text Normalization

Handles preprocessing of text:

```rust
pub struct TextNormalizer {
    number_converter: NumberConverter,
    symbol_mapper: SymbolMapper,
    tokenizer: WordTokenizer,
}
```

Key features:
- Number expansion (cardinal, ordinal, decimal)
- Symbol replacement
- Abbreviation handling
- Special case processing
- Unicode normalization

## 4. Data Structures

### 4.1 CompactTrie

The trie is optimized for memory efficiency:

```rust
struct Node {
    children: HashMap<char, usize>,
    value: Option<usize>,
}

pub struct CompactTrie {
    nodes: Vec<Node>,
    values: Vec<PhonemeSequence>,
}
```

Benefits:
- Using indices instead of pointers saves memory
- Contiguous memory layout improves cache behavior
- Integer-based indirection allows serialization
- Common prefixes are stored only once

### 4.2 Rule Representation

Rules are stored in a memory-efficient format:

```rust
struct ProductionRule {
    pattern: GraphemePattern,
    output: PhonemeTemplate,
    context: RuleContext,
    priority: u32,
}

struct GraphemePattern {
    regex: String,
    constraints: Vec<PatternConstraint>,
}

struct RuleContext {
    left_context: Option<GraphemePattern>,
    right_context: Option<GraphemePattern>,
    position: Option<WordPosition>,
}
```

Benefits:
- Explicit context representation for accurate matching
- Priority-based application order
- Constraint system for fine-grained control

### 4.3 Binary Dictionary Format

The binary dictionary format is designed for efficient loading:

```
[Header: 16 bytes]
  - Magic: 4 bytes ("PHON")
  - Version: 2 bytes
  - Entry count: 4 bytes
  - Metadata offset: 4 bytes
  - Checksum: 2 bytes

[String table: variable length]
  - Null-terminated strings

[Trie nodes: variable length]
  - Each node: 4 bytes (2 bytes children count, 2 bytes value index)
  - Child entries: 4 bytes each (1 byte char, 3 bytes child index)

[Phoneme sequences: variable length]
  - Each sequence: 2 bytes length + packed phoneme data

[Metadata: variable length]
  - Additional dictionary information
```

Benefits:
- Compact representation of dictionary data
- Fast load time with minimal parsing
- Support for incremental loading
- Option for memory mapping

## 5. Algorithms

### 5.1 Dictionary Lookup

1. Convert input word to lowercase
2. Traverse trie nodes matching each character
3. Return pronunciation if found at terminal node
4. Apply fallback strategies if no match

### 5.2 Rule-Based Prediction

1. Sort rules by specificity and priority
2. Apply each matching rule, collecting results
3. Select best prediction based on confidence score
4. Use context to disambiguate pronunciations

### 5.3 Phoneme Conversion

1. Map source phonemes to intermediate representation
2. Apply transformation rules for target standard
3. Convert stress markings to target convention
4. Format according to target standard's conventions

### 5.4 Text Normalization

1. Expand numbers to word form
2. Replace symbols with word equivalents
3. Expand abbreviations and acronyms
4. Handle special cases like dates and times

## 6. Design Decisions

### 6.1 Zero Dependencies Philosophy

**Decision**: Implement all functionality in pure Rust with no external dependencies and embed all data directly in the source code.

**Rationale**:
- Reduces compilation time and binary size
- Eliminates dependency conflicts
- Simplifies deployment in constrained environments
- Allows use in `no_std` contexts (future goal)
- Provides greater control over implementation details
- **Enables WebAssembly (WASM) compatibility** with no external file loading

**Tradeoffs**:
- Requires reimplementation of existing functionality
- May increase development time initially
- Requires more testing to ensure correctness
- Larger source files due to embedded dictionary data

### 6.2 Dictionary vs Rules Balance

**Decision**: Use a hybrid approach with dictionary-first and rule-based fallback, with the dictionary directly embedded in source code.

**Rationale**:
- Dictionary lookups are faster and more accurate for known words
- Embedding the dictionary directly in the source ensures consistent behavior across all platforms
- No external file loading or dependency on filesystem access
- Perfect for WebAssembly (WASM) targets and other restricted environments
- Rules provide reasonable predictions for unknown words
- Combined approach balances accuracy and coverage
- Common words get exact pronunciations, rare words get reasonable approximations

**Tradeoffs**:
- Dictionary requires memory
- Slightly larger source code repository
- Longer compile times when dictionary is updated 
- Rules add complexity and processing time

### 6.3 Memory Optimization

**Decision**: Use compact data structures and bit-packing for memory efficiency.

**Rationale**:
- Reduces overall memory footprint
- Improves cache behavior
- Enables larger dictionaries in constrained environments
- Faster serialization/deserialization

**Tradeoffs**:
- More complex implementation
- Slightly slower access patterns in some cases

### 6.4 Context-Aware Pronunciation

**Decision**: Include context information in the G2P conversion process.

**Rationale**:
- Many words have multiple valid pronunciations depending on context
- Provides more accurate results, especially for homographs
- Enables natural pronunciation in complete sentences

**Tradeoffs**:
- More complex API
- Additional processing overhead

### 6.5 Multiple Phoneme Standards

**Decision**: Support multiple phoneme representation standards (ARPABET, IPA, etc.).

**Rationale**:
- Different applications have different requirements
- Standards have different strengths and weaknesses
- Provides flexibility for different use cases

**Tradeoffs**:
- Additional conversion logic
- More complex API

## 7. Performance Considerations

### 7.1 Dictionary Lookups

- Trie-based lookups provide O(m) complexity (where m is word length)
- Compact trie reduces memory overhead
- Case-insensitive lookups handled efficiently

### 7.2 Rule Application

- Rules are sorted by priority and specificity
- Context checking is optimized to minimize redundant computation
- Pattern matching uses efficient algorithms

### 7.3 Memory Usage

- Bit-packed phoneme representation
- String deduplication in dictionary
- Compact trie structure
- Integer indices instead of pointers

### 7.4 Startup Time

- Embedded dictionary data for instant availability
- No file loading overhead on startup
- Lazy initialization where appropriate
- Same behavior across all platforms and environments

## 8. Future Directions

### 8.1 Additional Languages

- Japanese (kana-based system)
- Spanish (regular orthography)
- French (complex orthography)
- Chinese (pinyin-based system)

### 8.2 Advanced Features

- Neural network models for unknown words
- Morphological analysis for compound words
- Context-aware homograph disambiguation
- Dynamic stress assignment
- Dialect variations

### 8.3 Optimization Opportunities

- SIMD acceleration for pattern matching
- Custom allocators for dictionary data
- Thread pools for batch processing
- Memory-mapped dictionary loading

## 9. Testing Strategy

### 9.1 Unit Tests

- Test each component in isolation
- Property-based testing for rule engine
- Comprehensive edge case coverage

### 9.2 Integration Tests

- End-to-end pipeline testing
- Cross-component interactions
- Real-world text samples

### 9.3 Performance Tests

- Benchmark critical paths
- Memory usage monitoring
- Profile-guided optimization

### 9.4 Accuracy Tests

- Comparison with reference dictionaries (CMU)
- Validation against known pronunciations
- Coverage analysis

## 10. References

1. CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
2. ARPABET: https://en.wikipedia.org/wiki/ARPABET
3. International Phonetic Alphabet: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet
4. English Orthography: https://en.wikipedia.org/wiki/English_orthography
5. G2P Algorithms: https://aclanthology.org/N10-1078.pdf
6. Trie Data Structures: https://en.wikipedia.org/wiki/Trie

---

This design document is a living document and will be updated as the implementation progresses and new insights are gained.