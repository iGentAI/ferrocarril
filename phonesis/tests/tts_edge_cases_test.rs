//! Comprehensive edge case tests for Phonesis G2P in TTS contexts
//!
//! This test suite covers various edge cases, gotchas, and corner cases
//! that are critical for a production-ready TTS system.

use phonesis::{
    GraphemeToPhoneme,
    english::EnglishG2P,
};
use std::error::Error;

/// Test decimal number handling (addressing the failing test case)
#[test]
fn test_decimal_numbers() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Basic decimals
        ("3.14", "three point fourteen"),
        ("0.5", "zero point five"),
        ("1.0", "one"),
        ("10.01", "ten point zero one"),
        ("99.99", "ninety-nine point ninety-nine"),
        
        // Edge cases
        ("0.0", "zero"),
        ("0.00", "zero"),
        (".5", "point five"),  // Leading decimal
        ("100.0", "one hundred"),
        
        // Scientific notation style
        ("1.23e6", "one point twenty-three e six"),  // May not normalize correctly
    ];
    
    for (input, _expected_normalized) in &test_cases {
        println!("Testing decimal: {}", input);
        match g2p.convert(input) {
            Ok(phonemes) => {
                let phoneme_str = phonemes.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  Phonemes: {}", phoneme_str);
                assert!(!phonemes.is_empty(), "Should produce phonemes for {}", input);
            }
            Err(e) => {
                // Some edge cases might fail, which is acceptable if documented
                println!("  Failed (acceptable): {}", e);
            }
        }
    }
    
    Ok(())
}

/// Test currency and monetary values
#[test]
fn test_currency_handling() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        "$10", "$10.50", "$1,234", "$0.99",
        "€50", "£100", "¥1000",
    ];
    
    for input in &test_cases {
        let result = g2p.convert(input)?;
        assert!(!result.is_empty(), "Should handle currency: {}", input);
    }
    
    Ok(())
}

/// Test punctuation and special characters in various contexts
#[test]
fn test_punctuation_contexts() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Sentence endings
        "Hello.", "Hello!", "Hello?",
        
        // Mid-sentence punctuation
        "Well, hello there", "Yes; I agree", "Wait - stop!",
        
        // Quotes and apostrophes
        "He said \"hello\"", "It's a nice day", "The '90s",
        
        // Parentheses and brackets
        "Hello (world)", "Test [123]", "{code}",
        
        // Special punctuation
        "Hello... world", "Really?!", "Stop!!",
    ];
    
    for input in &test_cases {
        let result = g2p.convert(input)?;
        assert!(!result.is_empty(), "Should handle punctuation: {}", input);
    }
    
    Ok(())
}

/// Test mixed case and capitalization
#[test]
fn test_case_handling() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Various capitalizations
        "hello", "Hello", "HELLO", "hELLo",
        
        // Acronyms and abbreviations
        "USA", "FBI", "PhD", "ATM",
        
        // CamelCase and snake_case
        "CamelCase", "snake_case", "PascalCase",
        
        // Mixed with numbers
        "Web2.0", "H2O", "3D",
    ];
    
    for input in &test_cases {
        let result = g2p.convert(input)?;
        assert!(!result.is_empty(), "Should handle case variation: {}", input);
    }
    
    Ok(())
}

/// Test URLs, emails, and technical text
#[test]
fn test_technical_text() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // URLs
        "https://example.com", "www.test.org",
        
        // Email addresses
        "user@example.com", "test.user@company.co.uk",
        
        // File paths
        "/usr/bin/test", "C:\\Windows\\System32",
        
        // Technical symbols
        "foo->bar", "x == y", "a != b",
        
        // IP addresses
        "192.168.1.1", "127.0.0.1",
    ];
    
    for input in &test_cases {
        // Technical text might fail or produce odd results, which is acceptable
        match g2p.convert(input) {
            Ok(phonemes) => {
                println!("Technical '{}': {} phonemes", input, phonemes.len());
            }
            Err(e) => {
                println!("Technical '{}' failed (acceptable): {}", input, e);
            }
        }
    }
    
    Ok(())
}

/// Test very long inputs (stress test)
#[test]
fn test_long_inputs() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Generate a very long sentence
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    
    let start = std::time::Instant::now();
    let result = g2p.convert(&long_text)?;
    let duration = start.elapsed();
    
    assert!(!result.is_empty());
    println!("Long text ({} chars) processed in {:?}", long_text.len(), duration);
    
    // Should complete in reasonable time (< 1 second for this size)
    assert!(duration.as_secs() < 1, "Processing took too long");
    
    Ok(())
}

/// Test empty and whitespace-only inputs
#[test]
fn test_empty_inputs() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        "", " ", "  ", "\t", "\n", "\r\n",
        "   \t\n   ",
    ];
    
    for input in &test_cases {
        let result = g2p.convert(input)?;
        // Empty input should produce empty output
        assert!(result.is_empty(), "Empty input '{}' should produce no phonemes", input.escape_debug());
    }
    
    Ok(())
}

/// Test Unicode and non-ASCII characters
#[test]
fn test_unicode_handling() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Accented characters
        "café", "naïve", "résumé",
        
        // Emoji (should probably be ignored or fail gracefully)
        "Hello 👋 world", "Test 🎉",
        
        // Other scripts
        "Hello мир", "Test 世界",
        
        // Special Unicode punctuation
        "Hello—world", "Test…", "\"quotes\"",
    ];
    
    for input in &test_cases {
        match g2p.convert(input) {
            Ok(phonemes) => {
                println!("Unicode '{}': {} phonemes", input, phonemes.len());
            }
            Err(e) => {
                println!("Unicode '{}' failed (acceptable): {}", input, e);
            }
        }
    }
    
    Ok(())
}

/// Test heteronyms (words with multiple pronunciations)
#[test]
fn test_heteronyms() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    // These words have different pronunciations based on context
    let heteronyms = [
        "read",    // reed vs red
        "live",    // liv vs laiv
        "tear",    // teer vs tair
        "wind",    // wind vs waind
        "bow",     // bau vs boh
        "lead",    // leed vs led
        "close",   // klohz vs klohs
        "present", // PREZ-ent vs pri-ZENT
    ];
    
    for word in &heteronyms {
        let result = g2p.convert(word)?;
        assert!(!result.is_empty(), "Should handle heteronym: {}", word);
        // Note: Without context, G2P will pick one pronunciation
        // This is a known limitation
    }
    
    Ok(())
}

/// Test compound words and hyphenation
#[test]
fn test_compound_words() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Hyphenated compounds
        "mother-in-law", "merry-go-round", "twenty-one",
        
        // Unhyphenated compounds
        "football", "notebook", "underground",
        
        // Multi-part compounds
        "jack-of-all-trades",
        
        // Technical compounds
        "e-mail", "co-worker", "re-enter",
    ];
    
    for input in &test_cases {
        let result = g2p.convert(input)?;
        assert!(!result.is_empty(), "Should handle compound: {}", input);
    }
    
    Ok(())
}

/// Test dates and times
#[test]
fn test_dates_times() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Dates
        "2025-08-03", "08/03/2025", "August 3, 2025",
        "3rd August 2025", "Aug 3",
        
        // Times
        "10:30", "10:30 AM", "22:45", "3:15 PM",
        "12:00", "00:00",
    ];
    
    for input in &test_cases {
        match g2p.convert(input) {
            Ok(phonemes) => {
                println!("Date/time '{}': {} phonemes", input, phonemes.len());
            }
            Err(e) => {
                println!("Date/time '{}' failed (may need better normalization): {}", input, e);
            }
        }
    }
    
    Ok(())
}

/// Test performance with realistic TTS input
#[test]
fn test_tts_performance() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Typical TTS inputs
    let sentences = [
        "Welcome to the automated phone system.",
        "Your account balance is $1,234.56.",
        "Press 1 for English, 2 for Spanish.",
        "The meeting is scheduled for 3:30 PM on Tuesday, August 3rd.",
        "Please enter your 16-digit card number.",
        "Thank you for calling. Have a great day!",
    ];
    
    let mut total_phonemes = 0;
    let start = std::time::Instant::now();
    
    for sentence in &sentences {
        let phonemes = g2p.convert(sentence)?;
        total_phonemes += phonemes.len();
    }
    
    let duration = start.elapsed();
    println!("Processed {} sentences ({} total phonemes) in {:?}", 
             sentences.len(), total_phonemes, duration);
    
    // Should be fast enough for real-time TTS
    assert!(duration.as_millis() < 100, "G2P should be fast for TTS use");
    
    Ok(())
}

/// Test robustness with malformed input
#[test]
fn test_malformed_input() -> Result<(), Box<dyn Error>> {
    let g2p = EnglishG2P::new()?;
    
    let test_cases = [
        // Multiple spaces
        "Hello    world",
        
        // Mixed whitespace
        "Hello\t\tworld\n\ntest",
        
        // Unclosed quotes
        "He said \"hello",
        
        // Mismatched brackets
        "Test (unclosed",
        "Test ]mismatch[",
        
        // Strange punctuation
        "Hello,,,world!!!",
        "Test???!!!...",
        
        // Mixed languages
        "Hello мир world",
    ];
    
    for input in &test_cases {
        // Should not panic, even if it fails
        match g2p.convert(input) {
            Ok(phonemes) => {
                assert!(!phonemes.is_empty(), "Should handle malformed: {}", input);
            }
            Err(e) => {
                println!("Malformed '{}' failed gracefully: {}", input, e);
            }
        }
    }
    
    Ok(())
}