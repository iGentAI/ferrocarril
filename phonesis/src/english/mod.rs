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
    phoneme::{Phoneme, PhonemeSequence, PhonemeStandard, StressLevel},
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

    /// Returns true if `word` is a common English function word
    /// that should be fully destressed in misaki-style output.
    ///
    /// Function words (articles, conjunctions, most prepositions,
    /// personal pronouns, auxiliary verbs, demonstratives, modal
    /// verbs) are typically pronounced with NO stress marker in
    /// connected speech, but retain their strong vowel form
    /// (e.g. "of" → `ʌv`, not `əv`). The `convert` method
    /// implements this by converting each phoneme to IPA via
    /// `to_string_in` and then stripping the `ˈ`/`ˌ` marker
    /// characters from the output, which drops the marker
    /// without triggering the stress-dependent AH/ER mapping
    /// that would otherwise weaken the vowel to `ə`/`əɹ`.
    ///
    /// Note: **possessive pronouns** (my, your, his, her, its,
    /// our, their, mine, yours, hers, ours, theirs) are NOT in
    /// this list — they're in `is_demoted_word` because misaki
    /// renders them with secondary stress (e.g. `ˌWəɹ` for
    /// "our") rather than fully destressed.
    fn is_function_word(word: &str) -> bool {
        matches!(word.to_lowercase().as_str(),
            // Articles
            "a" | "an" | "the" |
            // Coordinating conjunctions
            "and" | "or" | "but" | "nor" | "so" | "yet" | "for" |
            // Subordinating conjunctions
            "as" | "if" | "than" | "though" | "while" | "because" |
            "although" | "since" | "until" | "unless" | "whereas" | "whether" |
            // Common prepositions
            "of" | "in" | "at" | "by" | "on" | "to" | "with" | "from" |
            "into" | "onto" | "about" | "above" | "across" | "after" |
            "against" | "along" | "among" | "around" | "before" | "behind" |
            "below" | "beneath" | "beside" | "between" | "beyond" |
            "down" | "during" | "except" | "inside" | "near" | "off" |
            "out" | "outside" | "over" | "past" | "through" | "throughout" |
            "toward" | "towards" | "under" | "underneath" | "up" |
            "upon" | "within" | "without" |
            // Personal pronouns (subject/object)
            "i" | "you" | "he" | "she" | "it" | "we" | "they" |
            "me" | "him" | "us" | "them" |
            // Demonstratives
            "this" | "that" | "these" | "those" |
            // Forms of "to be"
            "am" | "is" | "are" | "was" | "were" | "be" | "been" | "being" |
            // Forms of "to have"
            "have" | "has" | "had" | "having" |
            // Forms of "to do"
            "do" | "does" | "did" | "doing" | "done" |
            // Modal auxiliary verbs
            "will" | "would" | "shall" | "should" | "can" | "could" |
            "may" | "might" | "must" | "ought" |
            // Question / relativizing words (often unstressed)
            "what" | "who" | "whom" | "whose" | "why" | "how" |
            "where" | "when" | "which"
        )
    }

    /// Returns true if `word` is a common English word that
    /// should be **demoted** to secondary stress in connected
    /// speech (primary stress marker `ˈ` becomes secondary `ˌ`),
    /// rather than fully destressed.
    ///
    /// This applies to possessive pronouns and adjectives —
    /// words like "our", "my", "your", "her", etc. Misaki
    /// renders "our" as `ˌWəɹ` (secondary-stress W diphthong
    /// followed by rhotacized schwa) rather than `Wəɹ` (no
    /// marker) or `ˈWəɹ` (full primary stress). The secondary
    /// marker gives the initial diphthong enough prosodic
    /// weight that downstream Whisper / ASR systems correctly
    /// hear "our" instead of confusing it for "or".
    ///
    /// Reflexive pronouns ("myself", "yourself", …) are
    /// intentionally omitted — they're usually content-bearing
    /// in English sentences and keep full primary stress.
    fn is_demoted_word(word: &str) -> bool {
        matches!(word.to_lowercase().as_str(),
            // Possessive pronouns (dependent form, used with a noun)
            "my" | "your" | "his" | "her" | "its" | "our" | "their" |
            // Possessive pronouns (independent form)
            "mine" | "yours" | "hers" | "ours" | "theirs"
        )
    }

    /// Push a misaki-style word boundary marker (a single
    /// space-symbol IPA `Phoneme`) into `result`, but only if
    /// `result` is non-empty and the last phoneme isn't already
    /// a space. Used at the start of every Word/Number/
    /// Alphanumeric/Symbol token's emission to ensure the output
    /// has a single space between word groups (and not between
    /// word-internal phonemes), matching misaki's `həlˈO wˈɜɹld`
    /// format rather than the previous `h ə l ˈO w ˈɜɹ l d`
    /// format that put a separator between every phoneme.
    fn push_word_boundary(result: &mut Vec<Phoneme>) {
        if result.is_empty() {
            return;
        }
        if let Some(last) = result.last() {
            if last.symbol == " " {
                return;
            }
        }
        result.push(Phoneme::new_with_standard(
            " ".to_string(),
            None,
            PhonemeStandard::IPA,
        ));
    }
}

impl GraphemeToPhoneme for EnglishG2P {
    fn convert(&self, text: &str) -> Result<Vec<Phoneme>> {
        // Normalize the text
        let normalized = self.normalizer.normalize(text)?;
        
        // Tokenize the normalized text for word-by-word processing
        let tokens = self.normalizer.tokenizer.tokenize(&normalized);
        
        let mut result = Vec::new();

        for token in tokens {
            match token.token_type {
                crate::normalizer::TokenType::Word => {
                    Self::push_word_boundary(&mut result);

                    // Look up in dictionary first; fall back to
                    // rules / character fallback if missing.
                    let mut word_phonemes: Vec<Phoneme> = if let Some(pronunciation) = self.dictionary.lookup(&token.text) {
                        pronunciation.phonemes.clone()
                    } else {
                        match self.apply_fallback(&token.text, None) {
                            Ok(p) => p.phonemes,
                            Err(e) => {
                                if self.options.fallback_strategy == FallbackStrategy::Skip {
                                    return Err(e);
                                }
                                // Last-resort character fallback;
                                // emit a silence if even that fails.
                                self.character_fallback(&token.text)
                                    .map(|p| p.phonemes)
                                    .unwrap_or_else(|_| vec![Phoneme::new("SIL", None)])
                            }
                        }
                    };

                    if Self::is_function_word(&token.text) {
                        for p in &word_phonemes {
                            let ipa = p.to_string_in(PhonemeStandard::IPA);
                            for c in ipa.chars() {
                                if c != 'ˈ' && c != 'ˌ' {
                                    result.push(Phoneme::new_with_standard(
                                        c.to_string(),
                                        None,
                                        PhonemeStandard::IPA,
                                    ));
                                }
                            }
                        }
                    } else if Self::is_demoted_word(&token.text) {
                        for p in word_phonemes.iter_mut() {
                            if matches!(p.stress, Some(StressLevel::Primary)) {
                                p.stress = Some(StressLevel::Secondary);
                            }
                        }
                        result.extend(word_phonemes);
                    } else {
                        result.extend(word_phonemes);
                    }
                },
                crate::normalizer::TokenType::Punctuation => {
                    for c in token.text.chars() {
                        result.push(Phoneme::new_with_standard(
                            c.to_string(),
                            None,
                            PhonemeStandard::IPA,
                        ));
                    }
                },
                crate::normalizer::TokenType::Symbol => {
                    Self::push_word_boundary(&mut result);
                    match self.character_fallback(&token.text) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
                            if self.options.fallback_strategy != FallbackStrategy::Skip {
                                result.push(Phoneme::new("SIL", None));
                            }
                        }
                    }
                },
                crate::normalizer::TokenType::Number => {
                    Self::push_word_boundary(&mut result);
                    match self.apply_fallback(&token.text, None) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
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
                    Self::push_word_boundary(&mut result);
                    match self.apply_fallback(&token.text, None) {
                        Ok(pronunciation) => {
                            for phoneme in pronunciation.phonemes {
                                result.push(phoneme);
                            }
                        },
                        Err(_) => {
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
                    continue;
                },
                crate::normalizer::TokenType::Unknown => {
                    Self::push_word_boundary(&mut result);
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