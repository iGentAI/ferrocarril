//! Ferrocarril adapter for the Phonesis G2P system
//!
//! This module provides an adapter that allows Phonesis to be used
//! as the G2P component in Ferrocarril.

use crate::{
    GraphemeToPhoneme,
    english::EnglishG2P,
    G2POptions,
    FallbackStrategy,
    error::G2PError,
    PhonemeStandard,
};
use std::sync::Arc;
use std::collections::HashMap;

/// G2P adapter for Ferrocarril
///
/// This adapter provides the necessary interface for Ferrocarril
/// to use Phonesis as its G2P component.
pub struct FerrocarrilG2PAdapter {
    /// The underlying G2P implementation
    inner: Arc<dyn GraphemeToPhoneme + Send + Sync>,
    
    /// Language code for the adapter
    language: String,
    
    /// Target phoneme standard
    standard: PhonemeStandard,
    
    /// Cache for commonly used conversions
    cache: HashMap<String, String>,
    
    /// Maximum cache size
    max_cache_size: usize,
}

impl FerrocarrilG2PAdapter {
    /// Create a new adapter with the English G2P implementation
    pub fn new_english() -> Result<Self, G2PError> {
        let options = G2POptions {
            handle_stress: true,
            default_standard: PhonemeStandard::ARPABET,
            fallback_strategy: FallbackStrategy::UseRules,
        };
        
        let g2p = EnglishG2P::with_options(options)?;
        
        Ok(Self {
            inner: Arc::new(g2p),
            language: "en-us".to_string(),
            standard: PhonemeStandard::ARPABET,
            cache: HashMap::new(),
            max_cache_size: 1000, // Default cache size
        })
    }
    
    /// Create a new adapter with a custom G2P implementation
    pub fn new<G: GraphemeToPhoneme + Send + Sync + 'static>(
        g2p: G,
        language: &str,
        standard: PhonemeStandard,
        max_cache_size: usize,
    ) -> Self {
        Self {
            inner: Arc::new(g2p),
            language: language.to_string(),
            standard,
            cache: HashMap::new(),
            max_cache_size,
        }
    }
    
    /// Set the target phoneme standard
    pub fn set_standard(&mut self, standard: PhonemeStandard) {
        self.standard = standard;
        self.clear_cache();
    }
    
    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Convert text to a phonetic representation suitable for Ferrocarril
    pub fn convert_for_tts(&mut self, text: &str) -> Result<String, G2PError> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }
        
        // Use IPA standard to match Kokoro expectations
        let phonemes = self.inner.convert_to_standard(text, self.standard)?;
        
        // Join with spaces to match misaki G2P format like "h ɛ l o w ɹ l d"
        let result = phonemes.join(" ");
        
        // Cache the result if cache isn't too large
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(text.to_string(), result.clone());
        }
        
        Ok(result)
    }
    
    /// Get the language code for this adapter
    pub fn language(&self) -> &str {
        &self.language
    }
    
    /// Get the phoneme standard for this adapter
    pub fn standard(&self) -> PhonemeStandard {
        self.standard
    }
}

/// Factory function to create a G2P adapter suitable for Ferrocarril
pub fn create_ferrocarril_g2p(language: &str) -> Result<FerrocarrilG2PAdapter, G2PError> {
    match language {
        "en" | "en-us" | "en-gb" => FerrocarrilG2PAdapter::new_english(),
        _ => Err(G2PError::UnsupportedLanguage(language.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ferrocarril_adapter_english() -> Result<(), Box<dyn std::error::Error>> {
        let mut adapter = FerrocarrilG2PAdapter::new_english()?;
        
        // Verify language and standard
        assert_eq!(adapter.language(), "en-us");
        assert_eq!(adapter.standard(), PhonemeStandard::ARPABET);
        
        // Test conversion
        let result = adapter.convert_for_tts("hello world")?;
        assert!(!result.is_empty());
        assert!(result.contains("HH"));  // Should contain the 'HH' phoneme for 'hello'
        
        // Test caching
        let result2 = adapter.convert_for_tts("hello world")?;
        assert_eq!(result, result2);  // Should return the cached value
        
        Ok(())
    }
    
    #[test]
    fn test_factory_creation() -> Result<(), Box<dyn std::error::Error>> {
        let adapter = create_ferrocarril_g2p("en-us")?;
        assert_eq!(adapter.language(), "en-us");
        
        // Should fail for unsupported languages
        let result = create_ferrocarril_g2p("fr");
        assert!(result.is_err());
        
        Ok(())
    }
}