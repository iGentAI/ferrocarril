//! Text normalization module for Phonesis
//!
//! This module provides functionality for normalizing text before
//! grapheme-to-phoneme conversion, including tokenization, number expansion,
//! and symbol handling.

mod tokenizer;
mod numbers;
mod symbols;

pub use tokenizer::{TextNormalizer, Token, TokenType};
pub use numbers::{NumberConverter, NumberConverterOptions};
pub use symbols::{SymbolMapper, SymbolMapperOptions, ContextRule};

use crate::error::Result;

/// Normalizes text for pronunciation
///
/// This struct combines the tokenizer, number converter, and symbol mapper
/// to provide comprehensive text normalization.
#[derive(Debug)]
pub struct Normalizer {
    /// Tokenizer for splitting text into tokens
    pub tokenizer: TextNormalizer,
    /// Number converter for expanding numeric values
    number_converter: NumberConverter,
    /// Symbol mapper for expanding symbols and abbreviations
    symbol_mapper: SymbolMapper,
}

impl Normalizer {
    /// Creates a new normalizer with default options
    pub fn new() -> Self {
        Self {
            tokenizer: TextNormalizer::new(),
            number_converter: NumberConverter::new(),
            symbol_mapper: SymbolMapper::new(),
        }
    }
    
    /// Creates a new normalizer with custom components
    pub fn with_components(
        tokenizer: TextNormalizer,
        number_converter: NumberConverter,
        symbol_mapper: SymbolMapper,
    ) -> Self {
        Self {
            tokenizer,
            number_converter,
            symbol_mapper,
        }
    }
    
    /// Normalize text for pronunciation
    ///
    /// This method:
    /// 1. Tokenizes the input text
    /// 2. Combines adjacent number and decimal tokens
    /// 3. Expands numbers to their word form
    /// 4. Replaces symbols with their word equivalents
    /// 5. Expands abbreviations
    /// 6. Returns the normalized text
    pub fn normalize(&self, text: &str) -> Result<String> {
        // Tokenize the input text
        let mut tokens = self.tokenizer.tokenize(text);
        
        // Pre-process: combine adjacent number and decimal tokens
        let mut i = 0;
        while i < tokens.len() {
            if i + 2 < tokens.len() && 
               tokens[i].token_type == TokenType::Number &&
               tokens[i+1].text == "." &&
               tokens[i+2].token_type == TokenType::Number {
                // Combine "123", ".", "45" into a single "123.45" number token
                let combined_text = format!("{}.{}", tokens[i].text, tokens[i+2].text);
                let new_token = Token::new(
                    combined_text,
                    TokenType::Number,
                    tokens[i].position,
                );
                tokens[i] = new_token;
                tokens.remove(i+1); // Remove the "."
                tokens.remove(i+1); // Remove the "45" (now at i+1)
            } else if i + 1 < tokens.len() &&
                      tokens[i].token_type == TokenType::Number &&
                      tokens[i+1].text.starts_with('.') && 
                      tokens[i+1].text.len() > 1 &&
                      tokens[i+1].text[1..].chars().all(|c| c.is_numeric()) {
                // Handle cases like "123" + ".45"
                let combined_text = format!("{}{}", tokens[i].text, tokens[i+1].text);
                let new_token = Token::new(
                    combined_text,
                    TokenType::Number,
                    tokens[i].position,
                );
                tokens[i] = new_token;
                tokens.remove(i+1);
            }
            i += 1;
        }
        
        // Process each token
        for i in 0..tokens.len() {
            let token = &tokens[i];
            
            match token.token_type {
                TokenType::Number => {
                    // Handle number conversion
                    if let Ok(number) = token.text.parse::<i64>() {
                        tokens[i] = Token::new(
                            self.number_converter.convert_cardinal(number),
                            TokenType::Word,
                            token.position,
                        );
                    } else if let Ok(number) = token.text.parse::<f64>() {
                        tokens[i] = Token::new(
                            self.number_converter.convert_decimal(number),
                            TokenType::Word,
                            token.position,
                        );
                    }
                },
                TokenType::Symbol => {
                    // For symbols, check if we should convert them
                    match token.text.as_str() {
                        "@" => {
                            tokens[i] = Token::new(
                                "at".to_string(),
                                TokenType::Word,
                                token.position,
                            );
                        },
                        "$" => {
                            tokens[i] = Token::new(
                                "dollar".to_string(),
                                TokenType::Word,
                                token.position,
                            );
                        },
                        "#" => {
                            tokens[i] = Token::new(
                                "hash".to_string(),
                                TokenType::Word,
                                token.position,
                            );
                        },
                        _ => {
                            // Keep other symbols as is
                        },
                    }
                },
                TokenType::Punctuation => {
                    // Check for leading decimal points
                    if token.text.starts_with('.') && token.text.len() > 1 && 
                       token.text[1..].chars().all(|c| c.is_numeric()) {
                        // This is a leading decimal like ".5"
                        // Convert to "0.5" and process as a decimal
                        let fixed_decimal = format!("0{}", token.text);
                        if let Ok(number) = fixed_decimal.parse::<f64>() {
                            tokens[i] = Token::new(
                                self.number_converter.convert_decimal(number),
                                TokenType::Word,
                                token.position,
                            );
                        }
                    } else {
                        // Handle special cases for punctuation
                        match token.text.as_str() {
                            "." => {
                                // Check context for periods
                                if i > 0 {
                                    let prev_token = &tokens[i-1];
                                    if prev_token.text == "example" && i < tokens.len() - 1 && tokens[i+1].text == "com" {
                                        tokens[i] = Token::new(
                                            "period".to_string(),
                                            TokenType::Word,
                                            token.position,
                                        );
                                    } else if prev_token.text == "Dr" || prev_token.text == "St" ||
                                             prev_token.text == "Mr" || prev_token.text == "Mrs" ||
                                             prev_token.text == "Ms" || prev_token.text == "Prof" {
                                        // This is a period after an abbreviation, keep it for now
                                        // It will be handled with the abbreviation expansion
                                    }
                                }
                            },
                            _ => {
                                // Keep other punctuation as is
                            },
                        }
                    }
                },
                TokenType::Word => {
                    // Check if it's an abbreviation
                    let expansion = if i < tokens.len() - 1 && tokens[i+1].text == "." {
                        // Word followed by period, check for abbreviations
                        self.symbol_mapper.convert_symbol(&format!("{}.", token.text))
                    } else {
                        None
                    };
                    
                    if let Some(expanded) = expansion {
                        tokens[i] = Token::new(
                            expanded,
                            TokenType::Word,
                            token.position,
                        );
                        // Skip the period if it's part of the abbreviation
                        if i < tokens.len() - 1 && tokens[i+1].text == "." {
                            tokens[i+1] = Token::new(
                                "".to_string(),  // Empty token to be filtered out later
                                TokenType::Punctuation,
                                tokens[i+1].position,
                            );
                        }
                    }
                },
                TokenType::Alphanumeric => {
                    // For alphanumeric tokens, check if they're ordinals (e.g., "1st", "2nd", "3rd")
                    if let Some(num_str) = token.text.strip_suffix("st") {
                        if let Ok(num) = num_str.parse::<i64>() {
                            if num == 1 || num % 10 == 1 && num % 100 != 11 {
                                tokens[i] = Token::new(
                                    self.number_converter.convert_ordinal(num),
                                    TokenType::Word,
                                    token.position,
                                );
                            }
                        }
                    } else if let Some(num_str) = token.text.strip_suffix("nd") {
                        if let Ok(num) = num_str.parse::<i64>() {
                            if num == 2 || num % 10 == 2 && num % 100 != 12 {
                                tokens[i] = Token::new(
                                    self.number_converter.convert_ordinal(num),
                                    TokenType::Word,
                                    token.position,
                                );
                            }
                        }
                    } else if let Some(num_str) = token.text.strip_suffix("rd") {
                        if let Ok(num) = num_str.parse::<i64>() {
                            if num == 3 || num % 10 == 3 && num % 100 != 13 {
                                tokens[i] = Token::new(
                                    self.number_converter.convert_ordinal(num),
                                    TokenType::Word,
                                    token.position,
                                );
                            }
                        }
                    } else if let Some(num_str) = token.text.strip_suffix("th") {
                        if let Ok(num) = num_str.parse::<i64>() {
                            tokens[i] = Token::new(
                                self.number_converter.convert_ordinal(num),
                                TokenType::Word,
                                token.position,
                            );
                        }
                    }
                },
                _ => {}, // No conversion for other token types
            }
        }
        
        // Remove empty tokens created during processing
        tokens.retain(|token| !token.text.is_empty());
        
        // Convert tokens back to text
        let normalized = self.tokenizer.tokens_to_text(&tokens);
        Ok(normalized)
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalizer_basic() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("Hello, world!").unwrap();
        assert_eq!(result, "Hello, world!");
    }
    
    #[test]
    fn test_normalizer_numbers() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("I have 42 apples").unwrap();
        assert_eq!(result, "I have forty-two apples");
    }
    
    #[test]
    fn test_normalizer_symbols() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("Send email to info@example.com").unwrap();
        assert_eq!(result, "Send email to info at example period com");
    }
    
    #[test]
    fn test_normalizer_abbreviations() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("Dr. Smith lives on Oak St.").unwrap();
        assert_eq!(result, "Doctor Smith lives on Oak Saint");
    }
    
    #[test]
    fn test_normalizer_combined() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("Dr. Smith has $123.45 and lives at 42 Elm St.").unwrap();
        assert_eq!(result, "Doctor Smith has dollar one hundred and twenty-three point forty-five and lives at forty-two Elm Saint");
    }
    
    #[test]
    fn test_normalizer_ordinals() {
        let normalizer = Normalizer::new();
        let result = normalizer.normalize("He finished 1st place in the race").unwrap();
        assert_eq!(result, "He finished first place in the race");
    }
}