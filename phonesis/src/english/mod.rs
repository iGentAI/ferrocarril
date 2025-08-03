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
                    let min_expected_phonemes = word.chars().count() / 4;
                    
                    if pronunciation.phonemes.len() >= min_expected_phonemes {
                        return Ok(pronunciation);
                    }
                }
                
                // If rules fail, fall back to character-by-character conversion
                eprintln!("Warning: Rules failed for '{}', using character fallback", word);
                self.character_fallback(word)
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
                    let min_expected_phonemes = word.chars().count() / 4;
                    
                    if pronunciation.phonemes.len() >= min_expected_phonemes {
                        return Ok(pronunciation);
                    }
                }
                
                // If guessing fails, use character fallback
                eprintln!("Warning: Phoneme guessing failed for '{}', using character fallback", word);
                self.character_fallback(word)
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
    
    /// Character-level fallback that ensures we always produce phonemes.
    /// This is the last resort when all other methods fail.
    fn character_fallback(&self, word: &str) -> Result<PhonemeSequence> {
        let mut phonemes = Vec::new();
        
        for ch in word.chars() {
            // Map individual characters to basic phonemes
            let phoneme_sym = match ch.to_ascii_lowercase() {
                'a' => "EY",
                'b' => "B",
                'c' => "S",  // Soft C as default
                'd' => "D",
                'e' => "IY",
                'f' => "EH F",
                'g' => "JH IY", 
                'h' => "EY CH",
                'i' => "AY",
                'j' => "JH EY",
                'k' => "K EY",
                'l' => "EH L",
                'm' => "EH M",
                'n' => "EH N",
                'o' => "OW",
                'p' => "P IY",
                'q' => "K Y UW",
                'r' => "AA R",
                's' => "EH S",
                't' => "T IY",
                'u' => "Y UW",
                'v' => "V IY",
                'w' => "D AH B AH L Y UW",
                'x' => "EH K S",
                'y' => "W AY",
                'z' => "Z IY",
                '0' => "Z IH R OW",
                '1' => "W AH N",
                '2' => "T UW",
                '3' => "TH R IY",
                '4' => "F AO R", 
                '5' => "F AY V",
                '6' => "S IH K S",
                '7' => "S EH V AH N",
                '8' => "EY T",
                '9' => "N AY N",
                '-' => "D AE SH",
                '_' => "AH N D ER S K AO R",
                '.' => "D AA T",
                ',' => "K AA M AH",
                '!' => "IH K S K L AH M EY SH AH N",
                '?' => "K W EH S CH AH N",
                '@' => "AE T",
                '#' => "HH AE SH",
                '$' => "D AA L ER",
                '%' => "P ER S EH N T",
                '&' => "AE N D",
                
                // For any other character, use a generic phoneme
                _ => {
                    if ch.is_alphabetic() {
                        "AH"  // Schwa for unknown letters
                    } else if ch.is_numeric() {
                        "N AH M B ER"  // Generic "number"
                    } else if ch.is_whitespace() {
                        continue;  // Skip whitespace
                    } else {
                        "S IH M B AH L"  // Generic "symbol" 
                    }
                }
            };
            
            // Split multi-phoneme strings and create phoneme objects
            for p in phoneme_sym.split_whitespace() {
                phonemes.push(Phoneme::new(p.to_string(), None));
            }
        }
        
        // If we ended up with no phonemes (e.g., all whitespace), add a silence phoneme
        if phonemes.is_empty() {
            phonemes.push(Phoneme::new("SIL", None));
        }
        
        Ok(PhonemeSequence::new(phonemes))
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
            match token.token_type {
                crate::normalizer::TokenType::Word => {
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
                                if self.options.fallback_strategy == FallbackStrategy::Skip {
                                    // Propagate the error directly for Skip strategy
                                    return Err(e);
                                } else {
                                    // For other strategies, this should not happen due to character_fallback
                                    // But if it does, use emergency fallback
                                    match self.character_fallback(&token.text) {
                                        Ok(pronunciation) => {
                                            for phoneme in pronunciation.phonemes {
                                                result.push(phoneme);
                                            }
                                        },
                                        Err(_) => {
                                            // Last resort: add a silence phoneme
                                            result.push(Phoneme::new("SIL", None));
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                crate::normalizer::TokenType::Punctuation | 
                crate::normalizer::TokenType::Symbol => {
                    // Handle punctuation and symbols through fallback
                    match self.character_fallback(&token.text) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
                            // Fallback failed, skip or add silence based on strategy
                            if self.options.fallback_strategy != FallbackStrategy::Skip {
                                result.push(Phoneme::new("SIL", None));
                            }
                        }
                    }
                },
                crate::normalizer::TokenType::Number => {
                    // Numbers should have been processed in normalization
                    // If we still have a Number token, try to convert it
                    match self.apply_fallback(&token.text, None) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
                            // Use character fallback for numbers
                            match self.character_fallback(&token.text) {
                                Ok(pronunciation) => {
                                    for phoneme in pronunciation.phonemes {
                                        result.push(phoneme);
                                    }
                                },
                                Err(_) => {
                                    result.push(Phoneme::new("N AH M B ER", None));
                                }
                            }
                        }
                    }
                },
                crate::normalizer::TokenType::Alphanumeric => {
                    // Mixed alphanumeric tokens
                    match self.apply_fallback(&token.text, None) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
                            // Use character fallback
                            match self.character_fallback(&token.text) {
                                Ok(pronunciation) => {
                                    for phoneme in pronunciation.phonemes {
                                        result.push(phoneme);
                                    }
                                },
                                Err(_) => {
                                    result.push(Phoneme::new("SIL", None));
                                }
                            }
                        }
                    }
                },
                crate::normalizer::TokenType::Whitespace => {
                    // Skip whitespace tokens
                    continue;
                },
                crate::normalizer::TokenType::Unknown => {
                    // Handle unknown token types through character fallback
                    match self.character_fallback(&token.text) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
                            result.push(Phoneme::new("SIL", None));
                        }
                    }
                },
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