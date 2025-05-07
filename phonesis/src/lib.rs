//!
//! Phonesis is a pure Rust library for converting graphemes (written text) to phonemes
//! (pronunciation units). It provides high-quality G2P (Grapheme-to-Phoneme) conversion
//! for English with zero external dependencies.
//!
//! ## Features
//!
//! - Pure Rust implementation with zero external dependencies
//! - Dictionary-based conversion with rule-based fallbacks
//! - Support for multiple phoneme standards (ARPABET, IPA, etc.)
//! - Efficient memory usage with compressed data structures
//! - Extensible architecture for multiple languages
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use phonesis::{GraphemeToPhoneme, english::EnglishG2P};
//!
//! let g2p = EnglishG2P::new().expect("Failed to initialize G2P");
//! let phonemes = g2p.convert("hello").expect("Failed to convert text");
//! println!("{}", phonemes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" "));
//! ```

// Re-export main types and traits
pub use error::G2PError;
pub use phoneme::{Phoneme, PhonemePosition, PhonemeStandard, StressLevel, PhonemeType, PhonemeSequence};
pub use dictionary::PronunciationDictionary;

// Feature-gated re-exports
#[cfg(feature = "english")]
pub use self::english::EnglishG2P;

// Ferrocarril adapter
pub mod ferrocarril_adapter;

// Modules
mod error;
mod phoneme;

// Dictionary module
pub mod dictionary;

// Rule engine module
pub mod rules;

// Text normalization
pub mod normalizer;

// Language-specific implementations
#[cfg(feature = "english")]
pub mod english;

// Utility functions
mod utils {
    // Placeholder for now
    // Will be implemented as needed
}

/// Core trait for grapheme-to-phoneme conversion.
///
/// This trait defines the interface for converting text (graphemes)
/// to phonetic representations (phonemes).
pub trait GraphemeToPhoneme {
    /// Convert text to phonemes.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to convert
    ///
    /// # Returns
    ///
    /// A vector of phonemes, or an error if conversion failed.
    fn convert(&self, text: &str) -> Result<Vec<Phoneme>, error::G2PError>;
    
    /// Convert text to phonemes with additional context.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to convert
    /// * `context` - Additional context for conversion
    ///
    /// # Returns
    ///
    /// A vector of phonemes, or an error if conversion failed.
    fn convert_with_context(&self, text: &str, context: &Context) -> Result<Vec<Phoneme>, error::G2PError>;
    
    /// Convert text to a specific phoneme standard.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to convert
    /// * `standard` - The target phoneme standard
    ///
    /// # Returns
    ///
    /// A vector of phonemes in the requested standard, or an error.
    fn convert_to_standard(&self, text: &str, standard: PhonemeStandard) -> Result<Vec<String>, error::G2PError> {
        let phonemes = self.convert(text)?;
        Ok(phonemes.iter().map(|p| p.to_string_in(standard)).collect())
    }
    
    /// Check if a language is supported.
    ///
    /// # Arguments
    ///
    /// * `language` - The language code (e.g., "en-us")
    ///
    /// # Returns
    ///
    /// true if the language is supported, false otherwise.
    fn supports_language(&self, language: &str) -> bool;
}

/// Context information for G2P conversion.
///
/// Provides additional context that may affect pronunciation,
/// such as part-of-speech, word position, or surrounding words.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// Part of speech tag (noun, verb, etc.)
    pub pos_tag: Option<String>,
    
    /// Position within sentence (initial, medial, final)
    pub position: Option<rules::WordPosition>,
    
    /// Previous word (if any)
    pub prev_word: Option<String>,
    
    /// Next word (if any)
    pub next_word: Option<String>,
    
    /// Is the word capitalized
    pub is_capitalized: bool,
    
    /// Is the word part of a compound
    pub is_compound: bool,
}

/// Configuration options for G2P conversion.
#[derive(Debug, Clone)]
pub struct G2POptions {
    /// Whether to handle stress in pronunciation
    pub handle_stress: bool,
    
    /// Default phoneme standard to use
    pub default_standard: PhonemeStandard,
    
    /// Strategy for handling unknown words
    pub fallback_strategy: FallbackStrategy,
}

impl Default for G2POptions {
    fn default() -> Self {
        Self {
            handle_stress: true,
            default_standard: PhonemeStandard::ARPABET,
            fallback_strategy: FallbackStrategy::UseRules,
        }
    }
}

/// Strategy for handling unknown words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackStrategy {
    /// Skip unknown words
    Skip,
    
    /// Use rule-based conversion
    UseRules,
    
    /// Make a best guess based on similar words
    GuessPhonemes,
    
    /// Return the graphemes as-is
    ReturnGraphemes,
}

/// Convert phonemes from one standard to another.
///
/// # Arguments
///
/// * `phonemes` - The phonemes to convert
/// * `to` - The target standard
///
/// # Returns
///
/// A vector of converted phonemes
pub fn convert_phonemes(
    phonemes: &[Phoneme],
    to: PhonemeStandard,
) -> Vec<String> {
    phonemes.iter().map(|p| p.to_string_in(to)).collect()
}

/// Convert text to phonemes using the specified G2P implementation.
///
/// # Arguments
///
/// * `text` - The text to convert
/// * `g2p` - The G2P implementation to use
///
/// # Returns
///
/// A vector of phonemes, or an error if conversion failed.
pub fn convert_text<T: GraphemeToPhoneme>(
    text: &str,
    g2p: &T,
) -> Result<Vec<Phoneme>, error::G2PError> {
    g2p.convert(text)
}

/// Version of the phonesis library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn verify_version() {
        assert!(!VERSION.is_empty());
    }
}