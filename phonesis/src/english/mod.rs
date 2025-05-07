//! English-specific implementation of Grapheme-to-Phoneme conversion.
//!
//! This module provides the `EnglishG2P` struct, which implements the `GraphemeToPhoneme`
//! trait for English text, using a combination of dictionary lookup and rule-based 
//! pronunciation generation.

use std::sync::Arc;

use crate::{
    GraphemeToPhoneme,
    Context,
    G2POptions,
    FallbackStrategy,
    error::{G2PError, Result},
    phoneme::{Phoneme, PhonemeSequence, PhonemeStandard},
    dictionary::PronunciationDictionary,
    rules::{RuleEngine, RuleContext},
    normalizer::Normalizer,
};

mod rules;
mod dictionary;

use rules::initialize_english_rules;

// Re-export the dictionary module's function
pub use dictionary::get_default_dictionary;

/// English implementation of the G2P system.
pub struct EnglishG2P {
    /// Dictionary for word lookups
    dictionary: Arc<PronunciationDictionary>,
    
    /// Rule engine for unknown words
    rules: RuleEngine,
    
    /// Text normalizer for preprocessing
    normalizer: Normalizer,
    
    /// Configuration options
    options: G2POptions,
}

impl EnglishG2P {
    /// Create a new English G2P converter with default settings.
    pub fn new() -> Result<Self> {
        Self::with_options(G2POptions::default())
    }
    
    /// Create a new English G2P converter with custom options.
    pub fn with_options(options: G2POptions) -> Result<Self> {
        // Get dictionary from the embedded data
        let dictionary = get_default_dictionary()?;
        
        // Create rule engine with English rules
        let mut rules = RuleEngine::new(PhonemeStandard::ARPABET);
        initialize_english_rules(&mut rules);
        
        // Create text normalizer
        let normalizer = Normalizer::new();
        
        Ok(Self {
            dictionary,
            rules,
            normalizer,
            options,
        })
    }
    
    /// Apply fallback strategies for unknown words.
    fn apply_fallback(&self, word: &str, context: Option<&Context>) -> Result<PhonemeSequence> {
        match self.options.fallback_strategy {
            FallbackStrategy::UseRules => {
                // Convert context to rule context if provided
                let rule_context = context.map(|ctx| {
                    let mut rc = RuleContext::new();
                    
                    // Set position if available
                    if let Some(pos) = &ctx.position {
                        rc = rc.with_position(*pos);
                    }
                    
                    // Set capitalization
                    rc = rc.capitalized(ctx.is_capitalized);
                    
                    // Set surrounding words as context
                    if let Some(prev) = &ctx.prev_word {
                        rc = rc.with_context(Some(prev), None);
                    }
                    
                    if let Some(next) = &ctx.next_word {
                        rc = rc.with_context(None, Some(next));
                    }
                    
                    rc
                });
                
                // Try rule-based conversion
                if let Some(pronunciation) = self.rules.apply_with_context(word, rule_context.as_ref()) {
                    // FIX: Only consider this a success if we have a reasonable number of phonemes
                    // for the word length. A very conservative measure is 1/4 of character count.
                    let min_expected_phonemes = word.chars().count() / 4;
                    
                    if pronunciation.phonemes.len() >= min_expected_phonemes {
                        return Ok(pronunciation);
                    }
                    
                    // If we got fewer phonemes than expected, this is likely a partial match
                    // and we should treat it as a failure
                }
                
                Err(G2PError::UnknownWord(word.to_string()))
            },
            FallbackStrategy::Skip => {
                // Skip unknown words
                Err(G2PError::UnknownWord(word.to_string()))
            },
            FallbackStrategy::GuessPhonemes => {
                // Make a best guess based on similar words
                // This would be a more sophisticated algorithm in a full implementation
                // For now, just use rule-based fallback
                if let Some(pronunciation) = self.rules.apply(word) {
                    // FIX: Apply the same phoneme count check here as well
                    let min_expected_phonemes = word.chars().count() / 4;
                    
                    if pronunciation.phonemes.len() >= min_expected_phonemes {
                        return Ok(pronunciation);
                    }
                }
                
                Err(G2PError::UnknownWord(word.to_string()))
            },
            FallbackStrategy::ReturnGraphemes => {
                // Convert graphemes to phonemes directly
                // This is a very naive implementation
                let phonemes = word.chars()
                    .map(|c| Phoneme::new(c.to_string(), None))
                    .collect();
                
                Ok(PhonemeSequence::new(phonemes))
            },
        }
    }
}

impl GraphemeToPhoneme for EnglishG2P {
    fn convert(&self, text: &str) -> Result<Vec<Phoneme>> {
        // Normalize the text
        let normalized = self.normalizer.normalize(text)?;
        
        // Tokenize the normalized text for word-by-word processing
        // We're reusing the tokenizer from the normalizer
        let tokens = self.normalizer.tokenizer.tokenize(&normalized);
        
        let mut result = Vec::new();
        
        // Process each token
        for token in tokens {
            // Only process word tokens; skip punctuation, whitespace, etc.
            if token.token_type == crate::normalizer::TokenType::Word {
                // Look up in dictionary
                if let Some(pronunciation) = self.dictionary.lookup(&token.text) {
                    // Add all phonemes to the result
                    for phoneme in &pronunciation.phonemes {
                        result.push(phoneme.clone());
                    }
                } else {
                    // Apply fallback strategy for unknown words
                    match self.apply_fallback(&token.text, None) {
                        Ok(pronunciation) => {
                            // Add phonemes from fallback
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(e) => {
                            // FIX: Handle Skip strategy correctly by propagating the error
                            // immediately when we encounter an unknown word
                            if self.options.fallback_strategy == FallbackStrategy::Skip {
                                // Propagate the error directly for Skip strategy
                                return Err(e);
                            } else {
                                // For other strategies, handle based on context
                                match self.options.fallback_strategy {
                                    FallbackStrategy::Skip => {
                                        // This is redundant with check above, but kept for clarity
                                        // Skip this word, continue to next
                                        continue;
                                    },
                                    _ => {
                                        // Propagate the error
                                        return Err(e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    fn convert_with_context(&self, text: &str, context: &Context) -> Result<Vec<Phoneme>> {
        // In a full implementation, this would use the context to improve pronunciation
        // For now, we'll just use the regular convert method
        self.convert(text)
    }
    
    fn supports_language(&self, language: &str) -> bool {
        // We only support English
        matches!(language.to_lowercase().as_str(), "en" | "en-us" | "en-gb")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_conversion() {
        let g2p = EnglishG2P::new().unwrap();
        
        // Test with a word in our small dictionary
        let result = g2p.convert("hello").unwrap();
        assert!(!result.is_empty());
        
        // The first phoneme should be HH
        assert_eq!(result[0].symbol, "HH");
    }
    
    #[test]
    fn test_normalization_integration() {
        let g2p = EnglishG2P::new().unwrap();
        
        // Test with number
        let result = g2p.convert("42").unwrap_or_default();
        
        // This should be normalized to "forty-two"
        // The dictionary won't have this, so it will either use rules
        // or return graphemes depending on the fallback strategy
        assert!(!result.is_empty());
    }
    
    #[test]
    fn test_language_support() {
        let g2p = EnglishG2P::new().unwrap();
        
        assert!(g2p.supports_language("en"));
        assert!(g2p.supports_language("en-us"));
        assert!(g2p.supports_language("en-gb"));
        assert!(!g2p.supports_language("fr"));
    }
    
    #[test]
    fn test_fallback_strategy() {
        // Test with ReturnGraphemes fallback
        let options = G2POptions {
            fallback_strategy: FallbackStrategy::ReturnGraphemes,
            ..Default::default()
        };
        
        let g2p = EnglishG2P::with_options(options).unwrap();
        
        // Use a word not in our dictionary
        let result = g2p.convert("antidisestablishmentarianism").unwrap();
        
        // Should have one phoneme per character
        assert_eq!(result.len(), "antidisestablishmentarianism".len());
    }
}