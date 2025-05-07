//!
//! This module provides types and utilities for context-aware rule matching,
//! allowing rules to be applied based on the context in which a word or grapheme appears.

use std::fmt;
use std::collections::HashMap;

/// Position of a word or grapheme within a larger context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WordPosition {
    /// At the beginning (e.g., first word in a sentence)
    Initial,
    
    /// In the middle
    Medial,
    
    /// At the end (e.g., last word in a sentence)
    Final,
}

impl fmt::Display for WordPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WordPosition::Initial => write!(f, "Initial"),
            WordPosition::Medial => write!(f, "Medial"),
            WordPosition::Final => write!(f, "Final"),
        }
    }
}

/// Part-of-speech tag for a word.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PartOfSpeech {
    /// Noun
    Noun,
    
    /// Verb
    Verb,
    
    /// Adjective
    Adjective,
    
    /// Adverb
    Adverb,
    
    /// Preposition
    Preposition,
    
    /// Conjunction
    Conjunction,
    
    /// Pronoun
    Pronoun,
    
    /// Interjection
    Interjection,
    
    /// Determiner
    Determiner,
    
    /// Number
    Number,
    
    /// Other
    Other(String),
}

impl fmt::Display for PartOfSpeech {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PartOfSpeech::Noun => write!(f, "Noun"),
            PartOfSpeech::Verb => write!(f, "Verb"),
            PartOfSpeech::Adjective => write!(f, "Adjective"),
            PartOfSpeech::Adverb => write!(f, "Adverb"),
            PartOfSpeech::Preposition => write!(f, "Preposition"),
            PartOfSpeech::Conjunction => write!(f, "Conjunction"),
            PartOfSpeech::Pronoun => write!(f, "Pronoun"),
            PartOfSpeech::Interjection => write!(f, "Interjection"),
            PartOfSpeech::Determiner => write!(f, "Determiner"),
            PartOfSpeech::Number => write!(f, "Number"),
            PartOfSpeech::Other(tag) => write!(f, "{}", tag),
        }
    }
}

impl From<&str> for PartOfSpeech {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "noun" | "nn" | "nns" | "nnp" | "nnps" => Self::Noun,
            "verb" | "vb" | "vbd" | "vbg" | "vbn" | "vbp" | "vbz" => Self::Verb,
            "adj" | "adjective" | "jj" | "jjr" | "jjs" => Self::Adjective,
            "adv" | "adverb" | "rb" | "rbr" | "rbs" => Self::Adverb,
            "prep" | "preposition" | "in" => Self::Preposition,
            "conj" | "conjunction" | "cc" => Self::Conjunction,
            "pron" | "pronoun" | "prp" | "prp$" => Self::Pronoun,
            "int" | "interjection" | "uh" => Self::Interjection,
            "det" | "determiner" | "dt" => Self::Determiner,
            "num" | "number" | "cd" => Self::Number,
            _ => Self::Other(s.to_string()),
        }
    }
}

/// Context for a rule match.
///
/// This provides information about the context in which a rule is being applied,
/// such as the surrounding words, the position within a sentence, etc.
#[derive(Debug, Clone)]
pub struct RuleContext {
    /// The text or graphemes preceding the current match
    pub left_context: Option<String>,
    
    /// The text or graphemes following the current match
    pub right_context: Option<String>,
    
    /// The position within a larger context (e.g., sentence, word)
    pub position: Option<WordPosition>,
    
    /// Part-of-speech tag for the current word
    pub pos_tag: Option<PartOfSpeech>,
    
    /// Whether the matching text is capitalized
    pub is_capitalized: bool,
    
    /// Additional context information
    pub properties: HashMap<String, String>,
}

impl RuleContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self {
            left_context: None,
            right_context: None,
            position: None,
            pos_tag: None,
            is_capitalized: false,
            properties: HashMap::new(),
        }
    }
    
    /// Create a new context with left and right content.
    pub fn with_context(mut self, left: Option<&str>, right: Option<&str>) -> Self {
        self.left_context = left.map(|s| s.to_string());
        self.right_context = right.map(|s| s.to_string());
        self
    }
    
    /// Add a part-of-speech tag to this context.
    pub fn with_pos_tag(mut self, tag: impl Into<PartOfSpeech>) -> Self {
        self.pos_tag = Some(tag.into());
        self
    }
    
    /// Set the position within a larger context.
    pub fn with_position(mut self, position: WordPosition) -> Self {
        self.position = Some(position);
        self
    }
    
    /// Set capitalization flag.
    pub fn capitalized(mut self, is_capitalized: bool) -> Self {
        self.is_capitalized = is_capitalized;
        self
    }
    
    /// Add a property to this context.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
    
    /// Check if the preceding context matches a pattern.
    pub fn preceding_matches(&self, pattern: &str) -> bool {
        match &self.left_context {
            Some(context) => {
                // Simple substring match for now
                // TODO: support more complex pattern matching
                context.contains(pattern)
            }
            None => false,
        }
    }
    
    /// Check if the following context matches a pattern.
    pub fn following_matches(&self, pattern: &str) -> bool {
        match &self.right_context {
            Some(context) => {
                // Simple substring match for now
                // TODO: support more complex pattern matching
                context.contains(pattern)
            }
            None => false,
        }
    }
    
    /// Check if this context has a specific property.
    pub fn has_property(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }
    
    /// Get a property value.
    pub fn get_property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(|s| s.as_str())
    }
}

impl Default for RuleContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_part_of_speech_conversion() {
        assert_eq!(PartOfSpeech::from("noun"), PartOfSpeech::Noun);
        assert_eq!(PartOfSpeech::from("VB"), PartOfSpeech::Verb);
        assert_eq!(PartOfSpeech::from("JJ"), PartOfSpeech::Adjective);
        assert_eq!(PartOfSpeech::from("unknown"), PartOfSpeech::Other("unknown".to_string()));
    }
    
    #[test]
    fn test_context_builder_pattern() {
        let ctx = RuleContext::new()
            .with_pos_tag(PartOfSpeech::Noun)
            .with_position(WordPosition::Medial)
            .capitalized(true)
            .with_property("is_plural", "true");
        
        // Fix partial move issue by using a clone
        if let Some(pos_tag) = &ctx.pos_tag {
            assert!(matches!(pos_tag, PartOfSpeech::Noun));
        } else {
            panic!("pos_tag should be Some");
        }
        
        assert_eq!(ctx.position, Some(WordPosition::Medial));
        assert!(ctx.is_capitalized);
        assert!(ctx.has_property("is_plural"));
        assert_eq!(ctx.get_property("is_plural"), Some("true"));
    }
    
    #[test]
    fn test_context_matching() {
        let ctx = RuleContext::new().with_context(Some("the big"), Some("dog"));
        
        assert!(ctx.preceding_matches("big"));
        assert!(ctx.following_matches("dog"));
        assert!(!ctx.preceding_matches("small"));
        assert!(!ctx.following_matches("cat"));
    }
}