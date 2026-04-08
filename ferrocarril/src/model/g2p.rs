//! G2P integration for Ferrocarril TTS
//! 
//! This module provides proper integration between Phonesis G2P and Ferrocarril TTS,
//! ensuring compatibility with the Kokoro reference implementation.

// `original_text` and `success` are part of the public `G2PResult` API surface
// even though the binary's hot path only reads `phonemes`. Likewise
// `convert_with_chunking` is kept as a stub placeholder for future chunking
// support. Suppress dead-code warnings module-wide.
#![allow(dead_code)]

use ferrocarril_core::PhonesisG2P;
use std::error::Error;

/// Maximum number of phoneme tokens that we'll pass into the
/// downstream inference path. The BERT in Kokoro has
/// `max_position_embeddings = 512`, and `infer_with_phonemes`
/// reserves 2 of those positions for `<bos>` and `<eos>`, so the
/// effective ceiling on phoneme tokens is 510.
///
/// "Token" here means one character in the joined phoneme string
/// — the downstream tokenizer in `infer_with_phonemes` iterates
/// `phonemes.chars()` and looks each code point up in Kokoro's
/// vocab, and that includes ASCII space (`" "` = vocab id 16)
/// which acts as a word boundary token.
pub const MAX_PHONEME_TOKENS: usize = 510;

/// Legacy alias for `MAX_PHONEME_TOKENS`. Retained for backwards
/// compatibility; use `MAX_PHONEME_TOKENS` in new code.
#[deprecated(note = "Use MAX_PHONEME_TOKENS instead — the old name was misleading and measured bytes, not tokens.")]
pub const MAX_PHONEME_LENGTH: usize = MAX_PHONEME_TOKENS;

/// Result of G2P conversion
pub struct G2PResult {
    /// Original text input
    pub original_text: String,

    /// Converted phonemes as a space-separated string
    pub phonemes: String,

    /// Whether the conversion was successful
    pub success: bool,
}

/// G2P handler for Ferrocarril
pub struct G2PHandler {
    /// Inner Phonesis G2P implementation
    g2p: PhonesisG2P,
}

impl G2PHandler {
    /// Create a new G2P handler
    pub fn new(language: &str) -> Result<Self, Box<dyn Error>> {
        let g2p = PhonesisG2P::new(language)?;
        Ok(Self { g2p })
    }

    /// Convert text to phonemes
    /// 
    /// This method handles text normalization, tokenization, and conversion to phonemes.
    /// It follows the same approach as the Kokoro reference implementation.
    pub fn convert(&self, text: &str) -> G2PResult {
        let original_text = text.to_string();

        // Try to convert text to phonemes using PhonesisG2P
        // The convert method returns a Result<String, FerroError> where the String is already space-separated phonemes
        match self.g2p.convert(text) {
            Ok(phoneme_string) => {
                let phoneme_char_count = phoneme_string
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .count();
                let min_expected_phonemes = original_text.chars().count() / 4;
                let success = phoneme_char_count > 0 &&
                    (phoneme_char_count >= min_expected_phonemes
                     || original_text.chars().count() <= 4);

                let token_count = phoneme_string.chars().count();
                let final_phonemes = if token_count > MAX_PHONEME_TOKENS {
                    eprintln!(
                        "ferrocarril: warning: truncating phoneme input from {} to {} tokens",
                        token_count, MAX_PHONEME_TOKENS
                    );
                    phoneme_string
                        .chars()
                        .take(MAX_PHONEME_TOKENS)
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                } else {
                    phoneme_string
                };

                G2PResult {
                    original_text,
                    phonemes: final_phonemes,
                    success,
                }
            },
            Err(e) => {
                // Failed to convert to phonemes
                eprintln!(
                    "ferrocarril: warning: G2P conversion failed for text '{}': {} (using grapheme fallback)",
                    text, e
                );

                // Create a fallback representation - each character separated by spaces
                let fallback_phonemes = text
                    .chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");

                G2PResult {
                    original_text,
                    phonemes: fallback_phonemes,
                    success: false,
                }
            }
        }
    }

    /// Convert text to phonemes with chunking
    ///
    /// This method is closer to how Kokoro handles longer texts,
    /// breaking them into chunks and processing each separately.
    /// Currently a stub as we need to fully implement text chunking.
    pub fn convert_with_chunking(&self, text: &str) -> Vec<G2PResult> {
        // For now, just handle as a single chunk
        // TODO: Implement proper chunking based on Kokoro's logic
        vec![self.convert(text)]
    }
}

/// Test module for G2P integration
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_conversion() -> Result<(), Box<dyn Error>> {
        let handler = G2PHandler::new("en-us")?;
        let result = handler.convert("hello world");

        assert!(result.success, "Basic conversion should succeed");
        assert!(!result.phonemes.is_empty(), "Should produce non-empty phonemes");

        assert!(
            result.phonemes.contains('h'),
            "Should contain 'h' phoneme for 'hello' (got: {})",
            result.phonemes
        );
        assert!(
            result.phonemes.contains('l'),
            "Should contain 'l' phoneme for 'hello'/'world' (got: {})",
            result.phonemes
        );
        assert!(
            result.phonemes.contains('O'),
            "Should contain 'O' (Kokoro /oʊ/) phoneme for 'hello' (got: {})",
            result.phonemes
        );
        assert!(
            result.phonemes.contains('ɜ'),
            "Should contain 'ɜ' phoneme for stressed-ER of 'world' (got: {})",
            result.phonemes
        );

        println!("Original: \"{}\"", result.original_text);
        println!("Phonemes: {}", result.phonemes);

        Ok(())
    }

    #[test]
    fn test_unknown_word_handling() -> Result<(), Box<dyn Error>> {
        let handler = G2PHandler::new("en-us")?;
        let result = handler.convert("antidisestablishmentarianism");

        // This may succeed or fail depending on the dictionary
        println!("Success: {}", result.success);
        println!("Phonemes: {}", result.phonemes);

        // Should produce some output regardless
        assert!(!result.phonemes.is_empty(), "Should produce some phoneme output");

        Ok(())
    }

    #[test]
    fn test_special_case_handling() -> Result<(), Box<dyn Error>> {
        let handler = G2PHandler::new("en-us")?;

        // Test with abbreviations
        let abbrev_result = handler.convert("TTS");
        println!("TTS phonemes: {}", abbrev_result.phonemes);

        // Test with numbers
        let num_result = handler.convert("42");
        println!("42 phonemes: {}", num_result.phonemes);

        // Test with symbols
        let sym_result = handler.convert("@");
        println!("@ phonemes: {}", sym_result.phonemes);

        Ok(())
    }
}