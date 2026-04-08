//!
//! This module provides pattern matching functionality for the rule engine,
//! allowing rules to match specific grapheme sequences based on patterns.

use std::fmt;
use crate::error::G2PError;

/// A constraint that can be applied to a pattern match.
#[derive(Debug, Clone)]
pub enum PatternConstraint {
    /// The grapheme must be a vowel
    IsVowel,
    
    /// The grapheme must be a consonant
    IsConsonant,
    
    /// The grapheme must be uppercase
    IsUppercase,
    
    /// The grapheme must be lowercase
    IsLowercase,
    
    /// The grapheme must be a digit
    IsDigit,
    
    /// The grapheme must be in a specific set
    InSet(Vec<char>),
    
    /// The grapheme must not be in a specific set
    NotInSet(Vec<char>),
    
    /// Custom constraint
    Custom(Custom),
}

/// A wrapper around a custom function to make it Debug and Clone compatible
#[derive(Clone)]
pub struct Custom {
    func: std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>
}

impl Custom {
    /// Create a new custom constraint
    pub fn new<F>(f: F) -> Self 
    where 
        F: Fn(&str) -> bool + Send + Sync + 'static 
    {
        Self { func: std::sync::Arc::new(f) }
    }
    
    /// Apply the constraint
    pub fn apply(&self, s: &str) -> bool {
        (self.func)(s)
    }
}

impl std::fmt::Debug for Custom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Custom(..)")
    }
}

impl fmt::Display for PatternConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternConstraint::IsVowel => write!(f, "IsVowel"),
            PatternConstraint::IsConsonant => write!(f, "IsConsonant"),
            PatternConstraint::IsUppercase => write!(f, "IsUppercase"),
            PatternConstraint::IsLowercase => write!(f, "IsLowercase"),
            PatternConstraint::IsDigit => write!(f, "IsDigit"),
            PatternConstraint::InSet(chars) => write!(f, "InSet({:?})", chars),
            PatternConstraint::NotInSet(chars) => write!(f, "NotInSet({:?})", chars),
            PatternConstraint::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl PatternConstraint {
    /// Check if a grapheme satisfies this constraint.
    pub fn matches(&self, grapheme: &str) -> bool {
        match self {
            PatternConstraint::IsVowel => {
                // Simple check for English vowels
                grapheme.chars().all(|c| "aeiouyAEIOUY".contains(c))
            },
            PatternConstraint::IsConsonant => {
                // Simple check for English consonants (not a vowel)
                grapheme.chars().all(|c| !"aeiouyAEIOUY".contains(c) && c.is_alphabetic())
            },
            PatternConstraint::IsUppercase => {
                grapheme.chars().all(|c| c.is_uppercase())
            },
            PatternConstraint::IsLowercase => {
                grapheme.chars().all(|c| c.is_lowercase())
            },
            PatternConstraint::IsDigit => {
                grapheme.chars().all(|c| c.is_digit(10))
            },
            PatternConstraint::InSet(chars) => {
                grapheme.chars().all(|c| chars.contains(&c))
            },
            PatternConstraint::NotInSet(chars) => {
                grapheme.chars().all(|c| !chars.contains(&c))
            },
            PatternConstraint::Custom(custom) => {
                custom.apply(grapheme)
            },
        }
    }
}

/// A pattern that can match a sequence of graphemes.
#[derive(Debug, Clone)]
pub struct GraphemePattern {
    /// The pattern to match
    pub pattern: String,
    
    /// Whether the pattern is a regex
    pub is_regex: bool,
    
    /// Constraints to apply to matched groups
    pub constraints: Vec<(usize, PatternConstraint)>,
}

impl GraphemePattern {
    /// Create a new pattern from a string literal.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            is_regex: false,
            constraints: Vec::new(),
        }
    }
    
    /// Create a new pattern from a regex pattern.
    pub fn regex(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            is_regex: true,
            constraints: Vec::new(),
        }
    }
    
    /// Add a constraint to this pattern.
    pub fn with_constraint(mut self, group: usize, constraint: PatternConstraint) -> Self {
        self.constraints.push((group, constraint));
        self
    }
    
    /// Check if this pattern matches a grapheme sequence.
    ///
    /// Returns the matched groups if the pattern matches, or None if it doesn't.
    pub fn matches(&self, text: &str) -> Option<Vec<String>> {
        // Simple substring matching for non-regex patterns
        if !self.is_regex {
            if text.contains(&self.pattern) {
                return Some(vec![self.pattern.clone()]);
            }
            return None;
        }
        
        // Handle basic regex-like patterns
        match self.pattern.as_str() {
            // Handle digit pattern \d+
            "\\d+" => {
                // Find the first sequence of digits
                let digit_sequence: String = text.chars()
                    .skip_while(|c| !c.is_digit(10))
                    .take_while(|c| c.is_digit(10))
                    .collect();
                
                if !digit_sequence.is_empty() {
                    return Some(vec![digit_sequence]);
                }
                None
            },
            // Handle word character pattern \w+
            "\\w+" => {
                // Find the first sequence of word characters
                let word_sequence: String = text.chars()
                    .skip_while(|c| !c.is_alphanumeric() && *c != '_')
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                
                if !word_sequence.is_empty() {
                    return Some(vec![word_sequence]);
                }
                None
            },
            // Handle any character pattern .+
            ".+" => {
                if !text.is_empty() {
                    return Some(vec![text.to_string()]);
                }
                None
            },
            // For other patterns, fall back to substring matching
            _ => {
                // Strip regex syntax for simple matching
                let simplified = self.pattern.replace("\\d+", "").replace("\\w+", "");
                
                if text.contains(&simplified) {
                    // Try to apply basic regex interpretation
                    if self.pattern.contains("\\d") {
                        // If pattern contains \d, check if text contains digits
                        let has_digits = text.chars().any(|c| c.is_digit(10));
                        if !has_digits {
                            return None;
                        }
                    }
                    
                    return Some(vec![simplified]);
                }
                None
            }
        }
    }
    
    /// Find all matches of this pattern in a text.
    pub fn find_all(&self, text: &str) -> Vec<(usize, Vec<String>)> {
        let mut result = Vec::new();
        
        // For literal patterns, find all occurrences
        if !self.is_regex {
            let mut start = 0;
            while let Some(pos) = text[start..].find(&self.pattern) {
                let match_pos = start + pos;
                result.push((match_pos, vec![self.pattern.clone()]));
                start = match_pos + 1;
            }
            return result;
        }
        
        // Handle basic regex-like patterns
        match self.pattern.as_str() {
            // Handle digit pattern \d+
            "\\d+" => {
                let mut start = 0;
                while start < text.len() {
                    // Find the next digit
                    if let Some(pos) = text[start..].find(|c: char| c.is_digit(10)) {
                        let match_start = start + pos;
                        let mut match_end = match_start;
                        
                        // Find the end of the digit sequence
                        while match_end < text.len() && text[match_end..].chars().next().unwrap().is_digit(10) {
                            match_end += 1;
                        }
                        
                        let matched = text[match_start..match_end].to_string();
                        result.push((match_start, vec![matched]));
                        
                        // Move past this match
                        start = match_end;
                    } else {
                        break;
                    }
                }
            },
            // For other patterns, fall back to substring matching
            _ => {
                // Strip regex syntax for simple matching
                let simplified = self.pattern.replace("\\d+", "").replace("\\w+", "");
                
                if !simplified.is_empty() {
                    let mut start = 0;
                    while let Some(pos) = text[start..].find(&simplified) {
                        let match_pos = start + pos;
                        result.push((match_pos, vec![simplified.clone()]));
                        start = match_pos + 1;
                    }
                }
            }
        }
        
        result
    }
    
    /// Compile this pattern into an optimized form.
    pub fn compile(&self) -> Result<CompiledPattern, G2PError> {
        // Validate regex patterns
        if self.is_regex {
            let open_parens = self.pattern.chars().filter(|&c| c == '(').count();
            let close_parens = self.pattern.chars().filter(|&c| c == ')').count();
            
            if open_parens != close_parens {
                return Err(G2PError::RuleError(
                    format!("Unbalanced parentheses in regex pattern: {}", self.pattern)
                ));
            }
            
            let open_brackets = self.pattern.chars().filter(|&c| c == '[').count();
            let close_brackets = self.pattern.chars().filter(|&c| c == ']').count();
            
            if open_brackets != close_brackets {
                return Err(G2PError::RuleError(
                    format!("Unbalanced square brackets in regex pattern: {}", self.pattern)
                ));
            }
        }
        
        Ok(CompiledPattern {
            original: self.clone(),
        })
    }
}

/// A compiled (optimized) pattern for efficient matching.
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    /// The original pattern
    original: GraphemePattern,
}

impl CompiledPattern {
    /// Check if this pattern matches a grapheme sequence.
    pub fn matches(&self, text: &str) -> Option<Vec<String>> {
        // Use the original pattern for matching
        // In a real implementation, we would use the optimized form
        self.original.matches(text)
    }
    
    /// Find all matches of this pattern in a text.
    pub fn find_all(&self, text: &str) -> Vec<(usize, Vec<String>)> {
        // Use the original pattern for matching
        // In a real implementation, we would use the optimized form
        self.original.find_all(text)
    }
}

/// A collection of patterns and utilities for matching grapheme sequences.
#[derive(Debug, Clone, Default)]
pub struct PatternMatcher {
    /// The compiled patterns
    patterns: Vec<CompiledPattern>,
}

impl PatternMatcher {
    /// Create a new, empty pattern matcher.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }
    
    /// Add a pattern to this matcher.
    pub fn add_pattern(&mut self, pattern: GraphemePattern) -> Result<(), G2PError> {
        let compiled = pattern.compile()?;
        self.patterns.push(compiled);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_constraint_matching() {
        let c1 = PatternConstraint::IsVowel;
        assert!(c1.matches("a"));
        assert!(c1.matches("e"));
        assert!(!c1.matches("b"));
        
        let c2 = PatternConstraint::IsConsonant;
        assert!(!c2.matches("a"));
        assert!(c2.matches("b"));
        assert!(c2.matches("c"));
        
        let c3 = PatternConstraint::IsUppercase;
        assert!(c3.matches("A"));
        assert!(!c3.matches("a"));
        
        let c4 = PatternConstraint::IsLowercase;
        assert!(!c4.matches("A"));
        assert!(c4.matches("a"));
        
        let c5 = PatternConstraint::IsDigit;
        assert!(c5.matches("1"));
        assert!(!c5.matches("a"));
        
        let c6 = PatternConstraint::InSet(vec!['a', 'b', 'c']);
        assert!(c6.matches("a"));
        assert!(!c6.matches("d"));
        
        let c7 = PatternConstraint::NotInSet(vec!['a', 'b', 'c']);
        assert!(!c7.matches("a"));
        assert!(c7.matches("d"));
    }
    
    #[test]
    fn test_pattern_matching() {
        let p1 = GraphemePattern::new("hello");
        assert!(p1.matches("hello world").is_some());
        assert!(p1.matches("world hello").is_some());
        assert!(p1.matches("world").is_none());
        
        let p2 = GraphemePattern::new("a")
            .with_constraint(0, PatternConstraint::IsVowel);
        assert!(p2.matches("a").is_some());
        
        let p3 = GraphemePattern::regex("\\d+");
        assert!(p3.matches("123").is_some());
        assert!(p3.matches("abc").is_none());
    }
    
    #[test]
    fn test_pattern_find_all() {
        let p1 = GraphemePattern::new("a");
        let matches = p1.find_all("banana");
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].0, 1);
        assert_eq!(matches[1].0, 3);
        assert_eq!(matches[2].0, 5);
        
        let p2 = GraphemePattern::new("an");
        let matches = p2.find_all("banana");
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].0, 1);
        assert_eq!(matches[1].0, 3);
    }
    
    #[test]
    fn test_pattern_compilation() {
        // Test balanced parentheses
        let p1 = GraphemePattern::regex("(a|b)c");
        assert!(p1.compile().is_ok());
        
        // Test unbalanced parentheses
        let p2 = GraphemePattern::regex("(a|b");
        assert!(p2.compile().is_err());
        
        // Test balanced brackets
        let p3 = GraphemePattern::regex("[a-z]");
        assert!(p3.compile().is_ok());
        
        // Test unbalanced brackets
        let p4 = GraphemePattern::regex("[a-z");
        assert!(p4.compile().is_err());
    }
    
    #[test]
    fn test_regex_digit_pattern() {
        let p = GraphemePattern::regex("\\d+");
        assert!(p.matches("123").is_some());
        assert_eq!(p.matches("123").unwrap()[0], "123");
        assert!(p.matches("abc").is_none());
        assert!(p.matches("abc123").is_some());
        assert_eq!(p.matches("abc123").unwrap()[0], "123");
    }
}