//! G2P integration for Ferrocarril TTS
//! 
//! This module provides proper integration between Phonesis G2P and Ferrocarril TTS,
//! ensuring compatibility with the Kokoro reference implementation.

use ferrocarril_core::PhonesisG2P;
use std::error::Error;

/// Maximum sequence length for phoneme input
pub const MAX_PHONEME_LENGTH: usize = 510;

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
                // Convert Phoneme objects to a space-separated string
                let phoneme_str = phoneme_string
                    .split_whitespace()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>();
                
                // Check if we got a reasonable number of phonemes
                // A successful conversion should have more than just a few phonemes
                // for complex words
                let min_expected_phonemes = original_text.chars().count() / 4;
                let success = !phoneme_str.is_empty() && 
                    (phoneme_str.len() >= min_expected_phonemes || original_text.chars().count() <= 4);
                
                // Join with spaces
                let joined_phonemes = phoneme_str.join(" ");
                
                // Truncate if too long
                let final_phonemes = if joined_phonemes.len() > MAX_PHONEME_LENGTH {
                    println!("Warning: Truncating phonemes from {} to {} characters", 
                             joined_phonemes.len(), MAX_PHONEME_LENGTH);
                    joined_phonemes[..MAX_PHONEME_LENGTH].to_string()
                } else {
                    joined_phonemes
                };
                
                G2PResult {
                    original_text,
                    phonemes: final_phonemes,
                    success,
                }
            },
            Err(e) => {
                // Failed to convert to phonemes
                println!("G2P conversion failed: {}. Using default fallback.", e);
                println!("Input text: \"{}\"", text);
                
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
            result.phonemes.contains("ɛ"),
            "Should contain 'ɛ' phoneme for 'hello' (got: {})",
            result.phonemes
        );
        assert!(
            result.phonemes.contains('l'),
            "Should contain 'l' phoneme for 'hello'/'world' (got: {})",
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