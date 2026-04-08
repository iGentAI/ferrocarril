//! Example usage of the Phonesis text normalization module

use phonesis::normalizer::{
    Normalizer, TextNormalizer, NumberConverter,
};
use std::io::{self, Write};

fn main() {
    println!("Phonesis Text Normalization Example");
    println!("==================================\n");
    
    // Create the normalizer
    let normalizer = Normalizer::new();
    
    // Demonstrate normalizer with some examples
    let examples = [
        "Hello, world! This is a test.",
        "I have 42 apples and 7 oranges.",
        "It costs $19.99 for the package.",
        "The meeting is at 2:30 p.m. on May 3rd, 2025.",
        "Please email me at user@example.com.",
        "Chapter #5: Special Cases & Exceptions",
        "100% of people breathe oxygen.",
        "The ratio is 3:2 or 3/2.",
        "She lives at 42 Elm St., Apt. #3."
    ];
    
    for example in examples {
        println!("Original:   {}", example);
        match normalizer.normalize(example) {
            Ok(normalized) => println!("Normalized: {}", normalized),
            Err(err) => println!("Error: {:?}", err),
        }
        println!();
    }
    
    // Interactive mode
    println!("Interactive mode:");
    println!("Enter text to normalize, or 'quit' to exit.");
    
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        
        if input.eq_ignore_ascii_case("quit") || 
           input.eq_ignore_ascii_case("exit") || 
           input.eq_ignore_ascii_case("q") {
            break;
        }
        
        // Optional: Show tokenization
        if input.starts_with("tokens ") {
            let text = &input[7..];
            let tokenizer = TextNormalizer::new();
            let tokens = tokenizer.tokenize(text);
            
            println!("Tokens for: {}", text);
            for (i, token) in tokens.iter().enumerate() {
                println!("  {}: {:?} [{:?}] at pos {}", i, token.text, token.token_type, token.position);
            }
            
            continue;
        }
        
        // Optional: Show number conversion
        if input.starts_with("number ") {
            if let Ok(number) = input[7..].parse::<i64>() {
                let converter = NumberConverter::new();
                println!("Cardinal: {}", converter.convert_cardinal(number));
                println!("Ordinal:  {}", converter.convert_ordinal(number));
                continue;
            }
            
            if let Ok(number) = input[7..].parse::<f64>() {
                let converter = NumberConverter::new();
                println!("Decimal: {}", converter.convert_decimal(number));
                
                if input[7..].starts_with("$") || input[7..].starts_with("€") {
                    let symbol = &input[7..8];
                    let amount = input[8..].parse::<f64>().unwrap_or(0.0);
                    println!("Currency: {}", converter.convert_currency(amount, symbol));
                }
                
                continue;
            }
        }
        
        // Standard normalization
        match normalizer.normalize(input) {
            Ok(normalized) => println!("Normalized: {}", normalized),
            Err(err) => println!("Error: {:?}", err),
        }
    }
    
    println!("Goodbye!");
}