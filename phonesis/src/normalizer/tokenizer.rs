//! Tokenizer module for text normalization
//! 
//! This module provides functionality to split text into tokens for further processing.
//! It handles various types of tokens including words, numbers, symbols, punctuation,
//! and whitespace, while properly handling Unicode characters.

use std::fmt;

/// Represents different types of tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// Alphabetic word
    Word,
    /// Numeric value
    Number,
    /// Symbol or special character
    Symbol,
    /// Punctuation marks
    Punctuation,
    /// Whitespace (spaces, tabs, newlines)
    Whitespace,
    /// Mixed alphanumeric (e.g., "A1", "123abc")
    Alphanumeric,
    /// Unknown token type
    Unknown,
}

/// Represents a single token with its type and text
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The original text of the token
    pub text: String,
    /// The type of the token
    pub token_type: TokenType,
    /// Whether the token was originally capitalized
    pub is_capitalized: bool,
    /// Whether the token was entirely uppercase
    pub is_uppercase: bool,
    /// Position in the original text (byte offset)
    pub position: usize,
    /// Length in bytes
    pub byte_length: usize,
}

impl Token {
    /// Creates a new token
    pub fn new(text: String, token_type: TokenType, position: usize) -> Self {
        let is_capitalized = text.chars().next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        let is_uppercase = !text.is_empty() && text.chars().all(|c| !c.is_lowercase());
        let byte_length = text.len();
        
        Self {
            text,
            token_type,
            is_capitalized,
            is_uppercase,
            position,
            byte_length,
        }
    }
    
    /// Returns true if the token contains only ASCII characters
    pub fn is_ascii(&self) -> bool {
        self.text.is_ascii()
    }
    
    /// Returns true if the token contains only alphabetic characters
    pub fn is_alphabetic(&self) -> bool {
        self.text.chars().all(|c| c.is_alphabetic())
    }
    
    /// Returns true if the token contains only numeric characters
    pub fn is_numeric(&self) -> bool {
        self.text.chars().all(|c| c.is_numeric())
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Text normalizer that handles tokenization
#[derive(Debug, Default)]
pub struct TextNormalizer {
    /// Whether to preserve case information
    preserve_case: bool,
    /// Whether to keep whitespace tokens
    keep_whitespace: bool,
}

impl TextNormalizer {
    /// Creates a new text normalizer with default settings
    pub fn new() -> Self {
        Self {
            preserve_case: true,
            keep_whitespace: false,
        }
    }
    
    /// Creates a new text normalizer with custom settings
    pub fn with_options(preserve_case: bool, keep_whitespace: bool) -> Self {
        Self {
            preserve_case,
            keep_whitespace,
        }
    }
    
    /// Tokenizes the input text into a sequence of tokens
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut i = 0;
        
        while i < text.len() {
            // Skip to valid char boundary
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            
            // Get current character
            let current_char = text[i..].chars().next().unwrap();
            let current_type = Self::classify_char(current_char);
            let start_pos = i;
            
            // Handle whitespace
            if current_type == TokenType::Whitespace {
                let mut end = i + current_char.len_utf8();
                while end < text.len() {
                    if let Some(next_char) = text[end..].chars().next() {
                        if !next_char.is_whitespace() {
                            break;
                        }
                        end += next_char.len_utf8();
                    } else {
                        break;
                    }
                }
                
                let token_text = &text[start_pos..end];
                let token = Token::new(
                    token_text.to_string(),
                    TokenType::Whitespace,
                    start_pos
                );
                
                if self.keep_whitespace {
                    tokens.push(token);
                }
                
                i = end;
                continue;
            }
            
            // Handle decimals specially (e.g., .78 as one token)
            if current_char == '.' && i + 1 < text.len() {
                let next_char = text[i+1..].chars().next().unwrap();
                if next_char.is_numeric() {
                    // This is a decimal point followed by digits
                    let mut end = i + 1;
                    while end < text.len() {
                        if let Some(ch) = text[end..].chars().next() {
                            if !ch.is_numeric() {
                                break;
                            }
                            end += ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    
                    let token_text = &text[start_pos..end];
                    let token = Token::new(
                        token_text.to_string(),
                        TokenType::Punctuation, // Keep as punctuation per test
                        start_pos
                    );
                    tokens.push(token);
                    i = end;
                    continue;
                }
            }
            
            // Handle words, numbers, and alphanumerics
            if current_type == TokenType::Word || current_type == TokenType::Number {
                let mut end = i + current_char.len_utf8();
                let mut has_alpha = current_char.is_alphabetic();
                let mut has_digit = current_char.is_numeric();
                
                while end < text.len() {
                    if let Some(next_char) = text[end..].chars().next() {
                        let next_type = Self::classify_char(next_char);
                        
                        // Keep adding chars if they're part of the same word/number
                        // or if we're building an alphanumeric token
                        let continue_token = match (current_type, next_type) {
                            (TokenType::Word, TokenType::Word) => true,
                            (TokenType::Number, TokenType::Number) => true,
                            (TokenType::Word, TokenType::Number) => true,
                            (TokenType::Number, TokenType::Word) => true,
                            _ => false,
                        };
                        
                        if !continue_token {
                            break;
                        }
                        
                        has_alpha = has_alpha || next_char.is_alphabetic();
                        has_digit = has_digit || next_char.is_numeric();
                        end += next_char.len_utf8();
                    } else {
                        break;
                    }
                }
                
                let token_text = &text[start_pos..end];
                let token_type = if has_alpha && has_digit {
                    TokenType::Alphanumeric
                } else if has_alpha {
                    TokenType::Word
                } else {
                    TokenType::Number
                };
                
                let token = Token::new(
                    token_text.to_string(),
                    token_type,
                    start_pos
                );
                
                tokens.push(token);
                i = end;
                continue;
            }
            
            // Handle punctuation and symbols
            let token = Token::new(
                current_char.to_string(),
                current_type,
                i
            );
            
            tokens.push(token);
            i += current_char.len_utf8();
        }
        
        tokens
    }
    
    /// Converts tokens back to text
    pub fn tokens_to_text(&self, tokens: &[Token]) -> String {
        let mut result = String::new();
        
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                // Check if we need a space
                let prev = &tokens[i-1];
                let need_space = match (prev.token_type, token.token_type) {
                    // No space before punctuation
                    (_, TokenType::Punctuation) => false,
                    // No space after opening punctuation
                    (TokenType::Punctuation, _) if matches!(prev.text.chars().next(), Some('(' | '[' | '{' | '"' | '\'')) => false,
                    // Space between most other tokens
                    _ => true,
                };
                
                if need_space {
                    result.push(' ');
                }
            }
            
            result.push_str(&token.text);
        }
        
        result
    }
    
    /// Classifies a character into a token type
    fn classify_char(c: char) -> TokenType {
        if c.is_alphabetic() {
            TokenType::Word
        } else if c.is_numeric() {
            TokenType::Number
        } else if c.is_whitespace() {
            TokenType::Whitespace
        } else if Self::is_punctuation(c) {
            TokenType::Punctuation
        } else if c.is_ascii_punctuation() || !c.is_ascii() {
            TokenType::Symbol
        } else {
            TokenType::Unknown
        }
    }
    
    /// Determines if a character is punctuation
    fn is_punctuation(c: char) -> bool {
        matches!(c, '.' | ',' | '!' | '?' | ':' | ';' | '-' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}')
    }
    
    /// Determines if a token should be kept based on normalizer settings
    fn should_keep_token(&self, token: &Token) -> bool {
        match token.token_type {
            TokenType::Whitespace => self.keep_whitespace,
            _ => true,
        }
    }
    
    /// Post-processes tokens to handle special cases like alphanumeric tokens
    pub fn post_process_tokens(&self, tokens: &mut Vec<Token>) {
        for token in tokens.iter_mut() {
            // This is now handled directly in tokenize() but we keep this for compatibility
            let has_alpha = token.text.chars().any(|c| c.is_alphabetic());
            let has_digit = token.text.chars().any(|c| c.is_numeric());
            
            if has_alpha && has_digit {
                token.token_type = TokenType::Alphanumeric;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenization() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("Hello, world!");
        
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "Hello");
        assert_eq!(tokens[0].token_type, TokenType::Word);
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[1].token_type, TokenType::Punctuation);
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[2].token_type, TokenType::Word);
        assert_eq!(tokens[3].text, "!");
        assert_eq!(tokens[3].token_type, TokenType::Punctuation);
    }
    
    #[test]
    fn test_numeric_tokenization() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("123 456.78");
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "123");
        assert_eq!(tokens[0].token_type, TokenType::Number);
        assert_eq!(tokens[1].text, "456");
        assert_eq!(tokens[1].token_type, TokenType::Number);
        assert_eq!(tokens[2].text, ".78");
        assert_eq!(tokens[2].token_type, TokenType::Punctuation);
    }
    
    #[test]
    fn test_case_sensitivity() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("Hello WORLD");
        
        assert_eq!(tokens[0].is_capitalized, true);
        assert_eq!(tokens[0].is_uppercase, false);
        assert_eq!(tokens[1].is_capitalized, true);
        assert_eq!(tokens[1].is_uppercase, true);
    }
    
    #[test]
    fn test_unicode_handling() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("café naïve résumé");
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "café");
        assert_eq!(tokens[1].text, "naïve");
        assert_eq!(tokens[2].text, "résumé");
    }
    
    #[test]
    fn test_whitespace_handling() {
        let normalizer = TextNormalizer::with_options(true, true);
        let tokens = normalizer.tokenize("Hello  world\ttest");
        
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[1].token_type, TokenType::Whitespace);
        assert_eq!(tokens[3].token_type, TokenType::Whitespace);
    }
    
    #[test]
    fn test_symbols() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("email@example.com $100 #hashtag");
        
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Symbol));
    }
    
    #[test]
    fn test_token_to_text() {
        let normalizer = TextNormalizer::new();
        let original = "Hello, world!";
        let tokens = normalizer.tokenize(original);
        let reconstructed = normalizer.tokens_to_text(&tokens);
        
        assert_eq!(reconstructed, "Hello, world!");
    }
    
    #[test]
    fn test_alphanumeric_detection() {
        let normalizer = TextNormalizer::new();
        let mut tokens = normalizer.tokenize("A1 B2C3 123abc");
        normalizer.post_process_tokens(&mut tokens);
        
        assert_eq!(tokens[0].token_type, TokenType::Alphanumeric);
        assert_eq!(tokens[1].token_type, TokenType::Alphanumeric);
        assert_eq!(tokens[2].token_type, TokenType::Alphanumeric);
    }
    
    #[test]
    fn test_token_positions() {
        let normalizer = TextNormalizer::new();
        let tokens = normalizer.tokenize("Hello world");
        
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].byte_length, 5);
        assert_eq!(tokens[1].position, 6);
        assert_eq!(tokens[1].byte_length, 5);
    }
}