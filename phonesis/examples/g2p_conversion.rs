//! Example usage of the Phonesis G2P (Grapheme-to-Phoneme) system
//!
//! This example demonstrates how to use the EnglishG2P implementation
//! to convert text to phonetic representations.

use phonesis::{
    GraphemeToPhoneme, 
    PhonemeStandard, 
    G2POptions, 
    FallbackStrategy,
    english::EnglishG2P,
};
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Phonesis G2P Conversion Example");
    println!("===============================\n");
    
    // Create a new English G2P converter with default settings
    let g2p = EnglishG2P::new()?;
    
    // Demonstrate conversion with some example phrases
    let examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Phonesis converts text to phonemes.",
        "Text-to-speech systems need phonetic transcriptions.",
        "Numbers like 42 and symbols like @ are normalized.",
        "Dr. Smith lives at 123 Oak St.",
        "How are you today?",
    ];
    
    for example in &examples {
        println!("Text: {}", example);
        
        // Convert to phonemes
        match g2p.convert(example) {
            Ok(phonemes) => {
                // Print ARPABET representation
                println!("ARPABET: {}", phonemes.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(" "));
                
                // Print IPA representation
                let ipa = g2p.convert_to_standard(example, PhonemeStandard::IPA)?;
                println!("IPA: {}", ipa.join(" "));
                
                println!();
            },
            Err(err) => {
                println!("Error: {:?}", err);
                println!();
            }
        }
    }
    
    // Interactive mode
    println!("Interactive mode:");
    println!("Enter text to convert to phonemes, or 'quit' to exit");
    
    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        
        if input.eq_ignore_ascii_case("quit") || 
           input.eq_ignore_ascii_case("exit") || 
           input.eq_ignore_ascii_case("q") {
            break;
        }
        
        // Special command to test different fallback strategies
        if input.starts_with("fallback:") {
            let parts: Vec<&str> = input.splitn(3, ':').collect();
            if parts.len() < 3 {
                println!("Usage: fallback:<strategy>:<text>");
                println!("Strategies: skip, rules, guess, graphemes");
                continue;
            }
            
            let strategy = match parts[1].trim() {
                "skip" => FallbackStrategy::Skip,
                "rules" => FallbackStrategy::UseRules,
                "guess" => FallbackStrategy::GuessPhonemes,
                "graphemes" => FallbackStrategy::ReturnGraphemes,
                _ => {
                    println!("Unknown strategy: {}", parts[1]);
                    println!("Valid strategies: skip, rules, guess, graphemes");
                    continue;
                }
            };
            
            let text = parts[2].trim();
            let options = G2POptions {
                fallback_strategy: strategy,
                ..Default::default()
            };
            
            let custom_g2p = EnglishG2P::with_options(options)?;
            match custom_g2p.convert(text) {
                Ok(phonemes) => {
                    println!("ARPABET (with {:?} fallback): {}", strategy, phonemes.iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(" "));
                },
                Err(err) => {
                    println!("Error: {:?}", err);
                }
            }
            
            continue;
        }
        
        // Standard G2P conversion
        match g2p.convert(input) {
            Ok(phonemes) => {
                println!("ARPABET: {}", phonemes.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(" "));
                
                if let Ok(ipa) = g2p.convert_to_standard(input, PhonemeStandard::IPA) {
                    println!("IPA: {}", ipa.join(" "));
                }
            },
            Err(err) => {
                println!("Error: {:?}", err);
            }
        }
    }
    
    println!("Goodbye!");
    Ok(())
}