
> **Project Goal**: Create a pure Rust, library for Grapheme-to-Phoneme (G2P) conversion, initially focusing on English.

**Last Updated**: April 27, 2025

## Current Implementation State

We have established the foundational structure and core components for the Phonesis library, now housed within the `phonesis` crate inside the Ferrocarril repository.

### Completed Components:

1.  **Project Setup**:
    *   `phonesis/Cargo.toml`: Defined package metadata, features, and initial dependencies.
    *   `phonesis/README.md`: Created comprehensive README file.
    *   `phonesis/DESIGN.md`: Documented architecture and design decisions.
    *   `phonesis/CONTRIBUTING.md`: Added contribution guidelines.
    *   Basic directory structure created (`src/{dictionary,rules,english,normalizer,utils}`).

2.  **Core API & Types**:
    *   `src/lib.rs`: Established the main library entry point, module structure, and the core `GraphemeToPhoneme` trait.
    *   `src/error.rs`: Implemented `G2PError` enum and `Result` type for error handling.
    *   `src/phoneme.rs`: Implemented `Phoneme`, `PhonemeSequence`, `PhonemeStandard`, `StressLevel`, and related types, including ARPABET <-> IPA conversion and compact representation logic (`to_compact`/`from_compact`).

3.  **Dictionary System**:
    *   `src/dictionary/mod.rs`: Defined the `PronunciationDictionary` struct and its public API, including loading from CMU format and case-insensitive lookup.
    *   `src/dictionary/trie.rs`: Implemented the `CompactTrie` data structure for efficient word storage and retrieval.
    *   `src/dictionary/loader.rs`: Implemented basic loading/saving logic for a custom binary dictionary format (currently a stub).
    *   `src/dictionary/compact.rs`: Defined compact representations for phoneme sequences and dictionary storage structures.

4.  **Rule Engine System**:
    *   `src/rules/mod.rs`: Defined the module structure and core traits.
    *   `src/rules/context.rs`: Implemented `RuleContext`, `WordPosition`, and `PartOfSpeech` for context-aware rules.
    *   `src/rules/patterns.rs`: Implemented `GraphemePattern` and `PatternMatcher` with basic literal and placeholder regex support.
    *   `src/rules/engine.rs`: Implemented `RuleEngine`, `ProductionRule`, and `PhonemeTemplate` for applying rules.

5.  **Example**:
    *   `examples/basic_lookup.rs`: Created a working example demonstrating dictionary usage.

### Known Issues / Areas for Improvement:

*   **Dictionary Loader**: The binary format loading (`loader.rs`) is currently a stub and needs full implementation for serialization/deserialization.
*   **Rule Pattern Matching**: The regex support in `patterns.rs` is very basic and needs enhancement (e.g., using an actual regex engine or more sophisticated parsing).
*   **Text Normalization**: Modules (`numbers.rs`, `symbols.rs`, `tokenizer.rs`) are defined but not yet implemented.
*   **English Implementation**: The `english` module exists but `EnglishG2P` and specific English rules/dictionary data are not implemented.
*   **Testing**: Unit tests exist for some components, but comprehensive integration and accuracy testing is needed.

## Next Steps Roadmap

Based on the design and task list, the next priorities are:

1.  **Implement Text Normalization (HIGH PRIORITY)**:
    *   **Objective**: Create the `normalizer` module to handle text preprocessing.
    *   **Tasks**: Implement `numbers.rs` (number to words), `symbols.rs` (symbol expansion), and `tokenizer.rs` (word/punctuation splitting).
    *   **Files to Modify**: `src/normalizer/*`

2.  **Implement English-Specific Logic (HIGH PRIORITY)**:
    *   **Objective**: Create the `EnglishG2P` struct that integrates the dictionary, rules, and normalization.
    *   **Tasks**: Define English-specific rules, integrate a default English dictionary (potentially embedded), and implement the `GraphemeToPhoneme` trait for English.
    *   **Files to Modify**: `src/english/*`, `src/lib.rs`

3.  **Refine Dictionary System (MEDIUM PRIORITY)**:
    *   **Objective**: Complete the binary dictionary format loading/saving functionality.
    *   **Tasks**: Implement full serialization/deserialization in `loader.rs`, create a build script (`build.rs`) to compile a text dictionary (e.g., CMU subset) into the binary format.
    *   **Files to Modify**: `src/dictionary/loader.rs`, add `build.rs`

4.  **Enhance Rule Engine (MEDIUM PRIORITY)**:
    *   **Objective**: Improve the sophistication of the rule engine.
    *   **Tasks**: Implement more robust regex support in `patterns.rs`, potentially use a dependency like the `regex` crate (requires evaluating the zero-dependency goal).
    *   **Files to Modify**: `src/rules/patterns.rs`

5.  **Testing & Validation (ONGOING)**:
    *   **Objective**: Ensure correctness and quality.
    *   **Tasks**: Add unit tests for normalization, rules, and English implementation. Create integration tests comparing output to known pronunciations.
    *   **Files to Modify**: Add tests within modules and in the `tests/` directory.

**Recommendation for Immediate Next Step**: Focus on **Step 1: Implement Text Normalization**, starting with the `tokenizer.rs` file.