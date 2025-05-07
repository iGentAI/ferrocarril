//! Test for handling of complex or rare words in the G2P system
//!
//! This test specifically targets the issue where complex words
//! like "antidisestablishmentarianism" were incorrectly reported
//! as successful conversions but only returned minimal phoneme outputs.

use phonesis::{
    GraphemeToPhoneme, 
    english::EnglishG2P,
    G2POptions,
    FallbackStrategy,
    PhonemeStandard,
};
use std::error::Error;

/// Test complex word handling in the G2P conversion
#[test]
fn test_complex_word_handling() -> Result<(), Box<dyn Error>> {
    // Create G2P with the default options
    let g2p = EnglishG2P::new()?;
    
    // Define some complex words to test
    let complex_words = [
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
    ];
    
    for word in &complex_words {
        println!("Testing complex word: {}", word);
        
        match g2p.convert(word) {
            Ok(phonemes) => {
                // When successfully converted, check that the phoneme count is reasonable
                // Complex words should have multiple phonemes, not just 1-2
                println!("Converted '{}' to {} phonemes: {:?}", 
                         word, 
                         phonemes.len(),
                         phonemes.iter().map(|p| p.to_string()).collect::<Vec<_>>());
                
                // A reasonable phoneme count should be at least 1/4 of the character count
                // This is a very conservative estimate - most words have more phonemes than this
                let min_expected_phonemes = word.chars().count() / 4;
                assert!(phonemes.len() >= min_expected_phonemes, 
                        "Successful conversion should return more than just a few phonemes for word '{}'. \
                         Got {} phonemes, expected at least {}",
                        word, phonemes.len(), min_expected_phonemes);
            },
            Err(e) => {
                // If failure is reported, that's acceptable, but it should be consistent
                // and not return partial results for some complex words but not others
                println!("Failed to convert '{}': {}", word, e);
                // No assertion needed - failure is an acceptable outcome
            }
        }
    }
    
    // Also test the ReturnGraphemes fallback to make sure it works correctly
    let grapheme_options = G2POptions {
        handle_stress: true,
        default_standard: PhonemeStandard::ARPABET,
        fallback_strategy: FallbackStrategy::ReturnGraphemes,
    };
    
    let g2p_with_graphemes = EnglishG2P::with_options(grapheme_options)?;
    
    for word in &complex_words {
        // ReturnGraphemes should always "succeed" but return one phoneme per character
        let result = g2p_with_graphemes.convert(word)?;
        assert_eq!(result.len(), word.chars().count(),
                   "ReturnGraphemes should return one phoneme per character");
    }
    
    Ok(())
}

/// Test that appropriate errors are returned for unknown words
#[test]
fn test_unknown_word_errors() -> Result<(), Box<dyn Error>> {
    // Create G2P with the Skip fallback strategy, which should always return errors
    let options = G2POptions {
        handle_stress: true,
        default_standard: PhonemeStandard::ARPABET,
        fallback_strategy: FallbackStrategy::Skip,
    };
    
    let g2p = EnglishG2P::with_options(options)?;
    
    // Define some nonsense words that should definitely not be in the dictionary
    let nonsense_words = [
        "xyzzy", // Classic text adventure word
        "qwerty", // Keyboard sequence
        "zzyzx", // Unusual place name
    ];
    
    for word in &nonsense_words {
        println!("Testing nonsense word with Skip strategy: {}", word);
        
        // With our implementation of Skip, we expect an error when the word appears in a sentence
        // Let's create a simple sentence containing the word
        let sentence = format!("This is a {} test", word);
        match g2p.convert(&sentence) {
            Ok(_) => {
                panic!("Sentence with unknown word '{}' should have failed conversion with FallbackStrategy::Skip", word);
            },
            Err(e) => {
                println!("Correctly failed to convert '{}': {}", sentence, e);
                // Ensure the error is of the expected type
                assert!(e.to_string().contains("Unknown word"), 
                        "Error should be 'Unknown word' type, got: {}", e);
            }
        }
    }
    
    Ok(())
}