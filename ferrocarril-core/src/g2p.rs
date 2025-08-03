//! G2P (Grapheme-to-Phoneme) integration module for Ferrocarril
//! 
//! This module provides the PhonesisG2P wrapper that integrates the Phonesis
//! library into Ferrocarril's core functionality.

use crate::FerroError;
use phonesis::{
    english::EnglishG2P,
    GraphemeToPhoneme,
    G2POptions,
    FallbackStrategy,
    PhonemeStandard,
    ferrocarril_adapter::FerrocarrilG2PAdapter
};

/// Wrapper for Phonesis G2P functionality in Ferrocarril
pub struct PhonesisG2P {
    /// Internal adapter for Ferrocarril integration
    adapter: FerrocarrilG2PAdapter,
}

impl PhonesisG2P {
    /// Create a new PhonesisG2P instance for the specified language
    pub fn new(language: &str) -> Result<Self, FerroError> {
        let mut adapter = match language {
            "en" | "en-us" | "en_us" | "english" => {
                FerrocarrilG2PAdapter::new_english()
                    .map_err(|e| FerroError::new(format!("Failed to create English G2P: {}", e)))?
            }
            _ => {
                return Err(FerroError::new(format!("Unsupported language: {}", language)));
            }
        };
        
        adapter.set_standard(PhonemeStandard::IPA);
        
        Ok(Self { adapter })
    }
    
    /// Convert text to phonemes
    /// 
    /// Returns a space-separated string of IPA phonemes matching Kokoro format
    pub fn convert(&mut self, text: &str) -> Result<String, FerroError> {
        let result = self.adapter.convert_for_tts(text)
            .map_err(|e| FerroError::new(format!("G2P conversion failed: {}", e)))?;
        
        // Verify we're getting IPA characters, not ARPABET
        if result.contains("HH") || result.contains("EH") || result.contains("OW") {
            return Err(FerroError::new(format!(
                "G2P still producing ARPABET format '{}' instead of IPA - conversion failed", result
            )));
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phonesisg2p_creation() {
        let g2p = PhonesisG2P::new("en-us");
        assert!(g2p.is_ok());
        
        let g2p_err = PhonesisG2P::new("unknown");
        assert!(g2p_err.is_err());
    }
    
    #[test]
    fn test_phonesisg2p_conversion() {
        let mut g2p = PhonesisG2P::new("en-us").unwrap();
        let result = g2p.convert("hello");
        assert!(result.is_ok());
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty());
        assert!(phonemes.contains("h"));
    }
}