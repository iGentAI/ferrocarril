//! Symbol conversion module for text normalization
//! 
//! This module provides functionality to convert various symbols, special characters,
//! and abbreviations into their word representations for proper pronunciation.

use std::collections::HashMap;

/// Configuration options for symbol conversion
#[derive(Debug, Clone)]
pub struct SymbolMapperOptions {
    /// Whether to expand abbreviations
    pub expand_abbreviations: bool,
    /// Whether to use formal or informal pronunciations
    pub use_formal_style: bool,
    /// Custom symbol mappings (overrides defaults)
    pub custom_mappings: HashMap<String, String>,
    /// Context-sensitive rules for symbol expansion
    pub context_rules: HashMap<String, ContextRule>,
}

impl Default for SymbolMapperOptions {
    fn default() -> Self {
        Self {
            expand_abbreviations: true,
            use_formal_style: true,
            custom_mappings: HashMap::new(),
            context_rules: HashMap::new(),
        }
    }
}

/// Context-sensitive rule for symbol expansion
#[derive(Debug, Clone)]
pub struct ContextRule {
    /// Pattern to match before the symbol
    pub before_pattern: Option<String>,
    /// Pattern to match after the symbol
    pub after_pattern: Option<String>,
    /// Expansion to use when context matches
    pub expansion: String,
}

/// Converts symbols and special characters to their word representations
#[derive(Debug)]
pub struct SymbolMapper {
    options: SymbolMapperOptions,
    punctuation_map: HashMap<&'static str, &'static str>,
    math_map: HashMap<&'static str, &'static str>,
    currency_map: HashMap<&'static str, &'static str>,
    special_char_map: HashMap<&'static str, &'static str>,
    abbreviation_map: HashMap<&'static str, &'static str>,
}

impl SymbolMapper {
    /// Creates a new symbol mapper with default options
    pub fn new() -> Self {
        Self::with_options(SymbolMapperOptions::default())
    }
    
    /// Creates a new symbol mapper with custom options
    pub fn with_options(options: SymbolMapperOptions) -> Self {
        let mut mapper = Self {
            options,
            punctuation_map: HashMap::new(),
            math_map: HashMap::new(),
            currency_map: HashMap::new(),
            special_char_map: HashMap::new(),
            abbreviation_map: HashMap::new(),
        };
        
        mapper.initialize_default_mappings();
        mapper
    }
    
    /// Initializes default symbol mappings
    fn initialize_default_mappings(&mut self) {
        // Punctuation mappings
        self.punctuation_map.insert("&", "and");
        self.punctuation_map.insert("@", "at");
        self.punctuation_map.insert("#", "hash");
        self.punctuation_map.insert("%", "percent");
        self.punctuation_map.insert("*", "star");
        self.punctuation_map.insert("/", "slash");
        self.punctuation_map.insert("\\", "backslash");
        self.punctuation_map.insert("|", "pipe");
        self.punctuation_map.insert(":", "colon");
        self.punctuation_map.insert(";", "semicolon");
        self.punctuation_map.insert("!", "exclamation mark");
        self.punctuation_map.insert("?", "question mark");
        self.punctuation_map.insert("(", "open parenthesis");
        self.punctuation_map.insert(")", "close parenthesis");
        self.punctuation_map.insert("[", "open bracket");
        self.punctuation_map.insert("]", "close bracket");
        self.punctuation_map.insert("{", "open brace");
        self.punctuation_map.insert("}", "close brace");
        self.punctuation_map.insert("'", "apostrophe");
        self.punctuation_map.insert("\"", "quote");
        self.punctuation_map.insert(",", "comma");
        self.punctuation_map.insert(".", "period");
        self.punctuation_map.insert("...", "ellipsis");
        self.punctuation_map.insert("-", "dash");
        self.punctuation_map.insert("_", "underscore");
        self.punctuation_map.insert("~", "tilde");
        self.punctuation_map.insert("`", "backtick");
        self.punctuation_map.insert("^", "caret");
        
        // Mathematical symbols
        self.math_map.insert("+", "plus");
        self.math_map.insert("-", "minus");
        self.math_map.insert("×", "times");
        self.math_map.insert("÷", "divided by");
        self.math_map.insert("=", "equals");
        self.math_map.insert("≠", "not equals");
        self.math_map.insert("<", "less than");
        self.math_map.insert(">", "greater than");
        self.math_map.insert("≤", "less than or equal to");
        self.math_map.insert("≥", "greater than or equal to");
        self.math_map.insert("√", "square root");
        self.math_map.insert("∞", "infinity");
        self.math_map.insert("π", "pi");
        self.math_map.insert("∑", "sum");
        self.math_map.insert("∏", "product");
        self.math_map.insert("∫", "integral");
        self.math_map.insert("∂", "partial derivative");
        self.math_map.insert("±", "plus or minus");
        self.math_map.insert("°", "degrees");
        
        // Currency symbols
        self.currency_map.insert("$", "dollar");
        self.currency_map.insert("€", "euro");
        self.currency_map.insert("£", "pound");
        self.currency_map.insert("¥", "yen");
        self.currency_map.insert("₹", "rupee");
        self.currency_map.insert("¢", "cent");
        self.currency_map.insert("₣", "franc");
        self.currency_map.insert("₽", "ruble");
        self.currency_map.insert("฿", "baht");
        self.currency_map.insert("₩", "won");
        
        // Special characters
        self.special_char_map.insert("©", "copyright");
        self.special_char_map.insert("®", "registered");
        self.special_char_map.insert("™", "trademark");
        self.special_char_map.insert("§", "section");
        self.special_char_map.insert("¶", "paragraph");
        self.special_char_map.insert("†", "dagger");
        self.special_char_map.insert("‡", "double dagger");
        self.special_char_map.insert("•", "bullet");
        self.special_char_map.insert("·", "middle dot");
        self.special_char_map.insert("…", "ellipsis");
        self.special_char_map.insert("→", "right arrow");
        self.special_char_map.insert("←", "left arrow");
        self.special_char_map.insert("↑", "up arrow");
        self.special_char_map.insert("↓", "down arrow");
        self.special_char_map.insert("↔", "left right arrow");
        self.special_char_map.insert("⇒", "implies");
        self.special_char_map.insert("⇔", "if and only if");
        
        // Common abbreviations
        self.abbreviation_map.insert("Dr.", "Doctor");
        self.abbreviation_map.insert("Mr.", "Mister");
        self.abbreviation_map.insert("Mrs.", "Missus");
        self.abbreviation_map.insert("Ms.", "Ms");
        self.abbreviation_map.insert("Prof.", "Professor");
        self.abbreviation_map.insert("St.", "Saint");
        self.abbreviation_map.insert("Ave.", "Avenue");
        self.abbreviation_map.insert("Blvd.", "Boulevard");
        self.abbreviation_map.insert("Co.", "Company");
        self.abbreviation_map.insert("Corp.", "Corporation");
        self.abbreviation_map.insert("Inc.", "Incorporated");
        self.abbreviation_map.insert("Ltd.", "Limited");
        self.abbreviation_map.insert("vs.", "versus");
        self.abbreviation_map.insert("etc.", "et cetera");
        self.abbreviation_map.insert("e.g.", "for example");
        self.abbreviation_map.insert("i.e.", "that is");
        self.abbreviation_map.insert("p.m.", "P M");
        self.abbreviation_map.insert("a.m.", "A M");
        self.abbreviation_map.insert("Jr.", "Junior");
        self.abbreviation_map.insert("Sr.", "Senior");
        self.abbreviation_map.insert("No.", "Number");
        self.abbreviation_map.insert("Fig.", "Figure");
        self.abbreviation_map.insert("Rev.", "Reverend");
        self.abbreviation_map.insert("Gen.", "General");
        self.abbreviation_map.insert("Col.", "Colonel");
        self.abbreviation_map.insert("Capt.", "Captain");
        self.abbreviation_map.insert("Lt.", "Lieutenant");
    }
    
    /// Converts a symbol to its word representation
    pub fn convert_symbol(&self, symbol: &str) -> Option<String> {
        // Check custom mappings first
        if let Some(custom) = self.options.custom_mappings.get(symbol) {
            return Some(custom.clone());
        }
        
        // Check standard mappings - math symbols before punctuation for precedence
        if let Some(&word) = self.math_map.get(symbol) {
            Some(word.to_string())
        } else if let Some(&word) = self.punctuation_map.get(symbol) {
            Some(word.to_string())
        } else if let Some(&word) = self.currency_map.get(symbol) {
            Some(word.to_string())
        } else if let Some(&word) = self.special_char_map.get(symbol) {
            Some(word.to_string())
        } else if self.options.expand_abbreviations {
            self.abbreviation_map.get(symbol).map(|&s| s.to_string())
        } else {
            None
        }
    }
    
    /// Converts a symbol with context
    pub fn convert_symbol_with_context(&self, symbol: &str, before: Option<&str>, after: Option<&str>) -> Option<String> {
        // Check context rules first
        if let Some(rule) = self.options.context_rules.get(symbol) {
            if Self::matches_context(rule, before, after) {
                return Some(rule.expansion.clone());
            }
        }
        
        // Special context handling for certain symbols
        match symbol {
            // Contextual handling for slash
            "/" => {
                if let (Some(before), Some(after)) = (before, after) {
                    if before.chars().all(|c| c.is_numeric()) && after.chars().all(|c| c.is_numeric()) {
                        return Some("divided by".to_string());
                    }
                    if before.contains("/") || after.contains("/") {
                        return Some("slash".to_string());
                    }
                }
                Some("slash".to_string())
            }
            
            // Contextual handling for period
            "." => {
                if let Some(before) = before {
                    if before.chars().all(|c| c.is_uppercase()) {
                        // Likely an abbreviation
                        return Some("".to_string()); // Silent
                    }
                    if before.chars().all(|c| c.is_numeric()) && 
                        after.map_or(false, |a| a.chars().all(|c| c.is_numeric())) {
                        return Some("point".to_string());
                    }
                }
                if after.map_or(true, |a| a.trim().is_empty()) {
                    Some("period".to_string())
                } else {
                    Some("".to_string()) // Silent in middle of sentence
                }
            }
            
            // Contextual handling for dash
            "-" => {
                if let (Some(before), Some(after)) = (before, after) {
                    if before.chars().all(|c| c.is_numeric()) && after.chars().all(|c| c.is_numeric()) {
                        return Some("to".to_string()); // Ranges like 1-5
                    }
                    if before.chars().all(|c| c.is_alphabetic()) && after.chars().all(|c| c.is_alphabetic()) {
                        return Some("".to_string()); // Hyphenated words
                    }
                }
                Some("dash".to_string())
            }
            
            // Default to non-contextual conversion
            _ => self.convert_symbol(symbol),
        }
    }
    
    /// Checks if the context matches the rule
    fn matches_context(rule: &ContextRule, before: Option<&str>, after: Option<&str>) -> bool {
        let before_matches = rule.before_pattern.as_ref().map_or(true, |pattern| {
            before.map_or(false, |text| text.contains(pattern))
        });
        
        let after_matches = rule.after_pattern.as_ref().map_or(true, |pattern| {
            after.map_or(false, |text| text.contains(pattern))
        });
        
        before_matches && after_matches
    }
    
    /// Adds a custom symbol mapping
    pub fn add_custom_mapping(&mut self, symbol: &str, expansion: &str) {
        self.options.custom_mappings.insert(symbol.to_string(), expansion.to_string());
    }
    
    /// Adds a context rule for a symbol
    pub fn add_context_rule(&mut self, symbol: &str, rule: ContextRule) {
        self.options.context_rules.insert(symbol.to_string(), rule);
    }
    
    /// Expands abbreviations in text
    pub fn expand_abbreviations(&self, text: &str) -> String {
        let mut result = String::new();
        let mut current_word = String::new();
        
        for c in text.chars() {
            if c.is_whitespace() {
                if !current_word.is_empty() {
                    if let Some(expansion) = self.abbreviation_map.get(current_word.as_str()) {
                        if self.options.expand_abbreviations {
                            result.push_str(expansion);
                        } else {
                            result.push_str(&current_word);
                        }
                    } else {
                        result.push_str(&current_word);
                    }
                    current_word.clear();
                }
                result.push(c);
            } else {
                current_word.push(c);
            }
        }
        
        // Handle the last word
        if !current_word.is_empty() {
            if let Some(expansion) = self.abbreviation_map.get(current_word.as_str()) {
                if self.options.expand_abbreviations {
                    result.push_str(expansion);
                } else {
                    result.push_str(&current_word);
                }
            } else {
                result.push_str(&current_word);
            }
        }
        
        result
    }
}

impl Default for SymbolMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_punctuation_conversion() {
        let mapper = SymbolMapper::new();
        
        assert_eq!(mapper.convert_symbol("&"), Some("and".to_string()));
        assert_eq!(mapper.convert_symbol("@"), Some("at".to_string()));
        assert_eq!(mapper.convert_symbol("#"), Some("hash".to_string()));
        assert_eq!(mapper.convert_symbol("%"), Some("percent".to_string()));
        assert_eq!(mapper.convert_symbol("?"), Some("question mark".to_string()));
    }
    
    #[test]
    fn test_math_symbol_conversion() {
        let mapper = SymbolMapper::new();
        
        assert_eq!(mapper.convert_symbol("+"), Some("plus".to_string()));
        assert_eq!(mapper.convert_symbol("-"), Some("minus".to_string()));
        assert_eq!(mapper.convert_symbol("×"), Some("times".to_string()));
        assert_eq!(mapper.convert_symbol("÷"), Some("divided by".to_string()));
        assert_eq!(mapper.convert_symbol("="), Some("equals".to_string()));
        assert_eq!(mapper.convert_symbol("π"), Some("pi".to_string()));
    }
    
    #[test]
    fn test_currency_conversion() {
        let mapper = SymbolMapper::new();
        
        assert_eq!(mapper.convert_symbol("$"), Some("dollar".to_string()));
        assert_eq!(mapper.convert_symbol("€"), Some("euro".to_string()));
        assert_eq!(mapper.convert_symbol("£"), Some("pound".to_string()));
        assert_eq!(mapper.convert_symbol("¥"), Some("yen".to_string()));
    }
    
    #[test]
    fn test_abbreviation_expansion() {
        let mapper = SymbolMapper::new();
        
        assert_eq!(mapper.convert_symbol("Dr."), Some("Doctor".to_string()));
        assert_eq!(mapper.convert_symbol("Mr."), Some("Mister".to_string()));
        assert_eq!(mapper.convert_symbol("St."), Some("Saint".to_string()));
        assert_eq!(mapper.convert_symbol("etc."), Some("et cetera".to_string()));
    }
    
    #[test]
    fn test_custom_mappings() {
        let mut mapper = SymbolMapper::new();
        mapper.add_custom_mapping("&", "ampersand");
        
        assert_eq!(mapper.convert_symbol("&"), Some("ampersand".to_string()));
    }
    
    #[test]
    fn test_context_sensitive_expansion() {
        let mapper = SymbolMapper::new();
        
        // Test slash in different contexts
        assert_eq!(
            mapper.convert_symbol_with_context("/", Some("5"), Some("10")),
            Some("divided by".to_string())
        );
        assert_eq!(
            mapper.convert_symbol_with_context("/", Some("http:"), Some("www")),
            Some("slash".to_string())
        );
        
        // Test period in different contexts
        assert_eq!(
            mapper.convert_symbol_with_context(".", Some("Dr"), Some(" Smith")),
            Some("".to_string())
        );
        assert_eq!(
            mapper.convert_symbol_with_context(".", Some("3"), Some("14")),
            Some("point".to_string())
        );
        assert_eq!(
            mapper.convert_symbol_with_context(".", Some("sentence"), None),
            Some("period".to_string())
        );
        
        // Test dash in different contexts
        assert_eq!(
            mapper.convert_symbol_with_context("-", Some("1"), Some("5")),
            Some("to".to_string())
        );
        assert_eq!(
            mapper.convert_symbol_with_context("-", Some("twenty"), Some("one")),
            Some("".to_string())
        );
    }
    
    #[test]
    fn test_context_rules() {
        let mut mapper = SymbolMapper::new();
        
        // Add a custom context rule
        mapper.add_context_rule("#", ContextRule {
            before_pattern: Some("Chapter".to_string()),
            after_pattern: None,
            expansion: "number".to_string(),
        });
        
        assert_eq!(
            mapper.convert_symbol_with_context("#", Some("Chapter"), Some("5")),
            Some("number".to_string())
        );
        assert_eq!(
            mapper.convert_symbol_with_context("#", Some("Tweet"), Some("1")),
            Some("hash".to_string())
        );
    }
    
    #[test]
    fn test_expand_abbreviations_in_text() {
        let mapper = SymbolMapper::new();
        
        let text = "Dr. Smith lives on Oak St. near the Co. office.";
        let expanded = mapper.expand_abbreviations(text);
        
        assert_eq!(expanded, "Doctor Smith lives on Oak Saint near the Company office.");
    }
    
    #[test]
    fn test_disable_abbreviation_expansion() {
        let mut options = SymbolMapperOptions::default();
        options.expand_abbreviations = false;
        let mapper = SymbolMapper::with_options(options);
        
        assert_eq!(mapper.convert_symbol("Dr."), None);
        
        let text = "Dr. Smith";
        let expanded = mapper.expand_abbreviations(text);
        assert_eq!(expanded, "Dr. Smith");
    }
}