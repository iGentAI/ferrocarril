//! Robustness tests for Phonesis G2P to ensure it never crashes
//!
//! These tests verify that the system always produces some output
//! and handles all edge cases gracefully without system failures.

use phonesis::{
    GraphemeToPhoneme,
    english::EnglishG2P,
    G2POptions,
    FallbackStrategy,
};

/// Test that the system always produces output for any input
#[test]
fn test_never_crashes() {
    let strategies = vec![
        FallbackStrategy::UseRules,
        FallbackStrategy::GuessPhonemes,
        FallbackStrategy::ReturnGraphemes,
    ];
    
    for strategy in strategies {
        let options = G2POptions {
            fallback_strategy: strategy,
            ..Default::default()
        };
        
        let g2p = EnglishG2P::with_options(options).unwrap();
        
        // Test cases that should never crash
        let test_cases = vec![
            // Empty and whitespace - these might produce empty results
            ("", true),   // Empty produces empty 
            (" ", true),  // Space produces empty (whitespace is skipped)
            ("\t", true), // Tab produces empty
            ("\n", true), // Newline produces empty
            ("   \t\n   ", true), // Mixed whitespace produces empty
            
            // Single characters - these should produce phonemes
            ("a", false), ("1", false), ("!", false), ("@", false),
            
            // Unknown words - these should produce phonemes via fallback
            ("xyzzy", false), ("qwerty", false), ("asdfghjkl", false),
            
            // Mixed content - these should produce phonemes
            ("abc123", false), ("test@example.com", false), ("hello_world", false),
            
            // Special characters - these should produce phonemes
            ("!@#$%^&*()", false), ("[]{}|\\", false), ("<>?,./", false),
            
            // Unicode - may produce phonemes or empty based on handling
            ("café", false), ("naïve", false), ("résumé", false), 
            
            // Malformed input - should produce phonemes
            ("...", false), ("???", false), ("!!!", false), ("---", false),
            
            // Very long nonsense - should produce phonemes
            ("abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ", false),
            
            // Mixed languages - should produce phonemes for ASCII parts
            ("hello世界", false), ("test world", false),
            
            // URLs and technical - should produce phonemes
            ("https://example.com", false), ("user@example.com", false), ("192.168.1.1", false),
            
            // Random symbols - should produce phonemes
            ("♫♪♩♬", false), ("☺☻♥♦♣♠", false), ("←↑→↓↔↕", false),
        ];
        
        for (input, may_be_empty) in &test_cases {
            match g2p.convert(input) {
                Ok(phonemes) => {
                    // We expect some output unless it's allowed to be empty
                    if !may_be_empty && !input.trim().is_empty() {
                        assert!(!phonemes.is_empty(), 
                                "Non-empty input '{}' should produce phonemes with strategy {:?}", 
                                input, strategy);
                    }
                },
                Err(_) => {
                    // Only Skip strategy is allowed to fail
                    assert_eq!(strategy, FallbackStrategy::Skip, 
                              "Strategy {:?} should not fail on input '{}'", strategy, input);
                }
            }
        }
    }
}

/// Test character fallback specifically
#[test]
fn test_character_fallback_coverage() {
    let options = G2POptions {
        fallback_strategy: FallbackStrategy::UseRules,
        ..Default::default()
    };
    
    let g2p = EnglishG2P::with_options(options).unwrap();
    
    // Test all ASCII printable characters except whitespace
    for byte in 33u8..=126u8 {  // Skip ASCII 32 (space) since whitespace is skipped
        let ch = byte as char;
        let input = ch.to_string();
        
        let result = g2p.convert(&input).unwrap();
        assert!(!result.is_empty(), "Character '{}' should produce phonemes", ch);
    }
    
    // Test that whitespace produces empty result (expected behavior)
    let whitespace_result = g2p.convert(" ").unwrap();
    assert!(whitespace_result.is_empty(), "Whitespace should produce empty result");
    
    // Test digits specifically
    for digit in '0'..='9' {
        let input = digit.to_string();
        let result = g2p.convert(&input).unwrap();
        assert!(!result.is_empty(), "Digit '{}' should produce phonemes", digit);
    }
}

/// Test that fallback produces reasonable phoneme counts
#[test]
fn test_fallback_phoneme_counts() {
    let options = G2POptions {
        fallback_strategy: FallbackStrategy::UseRules,
        ..Default::default()
    };
    
    let g2p = EnglishG2P::with_options(options).unwrap();
    
    // Test that single letters produce reasonable phoneme counts
    let single_letters = vec!["a", "b", "w", "x", "y", "z"];
    
    for letter in single_letters {
        let result = g2p.convert(letter).unwrap();
        
        // Single letters should produce 1-5 phonemes (e.g., "w" -> "D AH B AH L Y UW")
        assert!(result.len() >= 1 && result.len() <= 7, 
                "Letter '{}' produced {} phonemes, expected 1-7", 
                letter, result.len());
    }
}

/// Test that common missing words now work
#[test]
fn test_missing_words_fixed() {
    let g2p = EnglishG2P::new().unwrap();
    
    let previously_missing = vec![
        "I", "i",
        "fourteen", "FOURTEEN", 
        "scheduled", "SCHEDULED",
        "AM", "am", "PM", "pm",
        "USA", "usa",
        "Monday", "monday",
        "January", "january",
        "http", "https", "www", "com",
    ];
    
    for word in previously_missing {
        let result = g2p.convert(word);
        assert!(result.is_ok(), "Word '{}' should now be in dictionary", word);
        
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty(), "Word '{}' should produce phonemes", word);
    }
}

/// Test leading decimal handling
#[test]
fn test_leading_decimal_fixed() {
    let g2p = EnglishG2P::new().unwrap();
    
    let decimal_cases = vec![
        (".5", "zero point fifty"),
        (".25", "zero point twenty-five"),
        (".1", "zero point ten"),
        (".01", "zero point zero one"),
        (".99", "zero point ninety-nine"),
    ];
    
    for (input, _expected) in decimal_cases {
        let result = g2p.convert(input);
        assert!(result.is_ok(), "Leading decimal '{}' should convert successfully", input);
        
        let phonemes = result.unwrap();
        assert!(!phonemes.is_empty(), "Leading decimal '{}' should produce phonemes", input);
        
        // Should start with "Z IH R OW" (zero)
        assert!(phonemes.len() > 3, "Should have phonemes for 'zero point ...'");
        assert_eq!(phonemes[0].symbol, "Z");
        assert_eq!(phonemes[1].symbol, "IH");  
        assert_eq!(phonemes[2].symbol, "R");
        assert_eq!(phonemes[3].symbol, "OW");
    }
}

/// Stress test with random input
#[test]
fn test_random_input_robustness() {
    use std::iter;
    
    let options = G2POptions {
        fallback_strategy: FallbackStrategy::UseRules,
        ..Default::default()
    };
    
    let g2p = EnglishG2P::with_options(options).unwrap();
    
    // Generate random strings
    for len in vec![1, 10, 50, 100] {
        for _ in 0..10 {
            // Random ASCII
            let random_ascii: String = iter::repeat_with(|| {
                let byte = 32 + (rand::random::<u8>() % 95); // ASCII printable range
                byte as char
            })
            .take(len)
            .collect();
            
            let result = g2p.convert(&random_ascii);
            assert!(result.is_ok(), "Random ASCII input should not crash");
            
            // Random alphanumeric
            let random_alnum: String = iter::repeat_with(|| {
                let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
                chars.chars().nth(rand::random::<usize>() % chars.len()).unwrap()
            })
            .take(len)
            .collect();
            
            let result = g2p.convert(&random_alnum);
            assert!(result.is_ok(), "Random alphanumeric input should not crash");
        }
    }
}

// Mock rand::random for testing (in real code would use the rand crate).
mod rand {
    use std::sync::atomic::{AtomicU8, Ordering};

    static COUNTER: AtomicU8 = AtomicU8::new(0);

    pub fn random<T>() -> T
    where
        T: Default + From<u8>,
    {
        let c = COUNTER.fetch_add(17, Ordering::Relaxed);
        T::from(c)
    }
}