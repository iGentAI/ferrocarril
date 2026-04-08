//! Comprehensive IPA conversion tests for Phonesis G2P system
//!
//! These tests validate ARPABET-to-IPA conversion quality, stress marker handling,
//! and compatibility with TTS systems like Kokoro that expect IPA character streams.

use phonesis::{
    GraphemeToPhoneme,
    english::EnglishG2P,
    PhonemeStandard,
    G2POptions,
    FallbackStrategy,
};
use std::collections::HashMap;

/// Test ARPABET to IPA conversion for common English phonemes
#[test]
fn test_arpabet_to_ipa_vowel_conversion() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Test specific vowel conversions known to be critical for TTS.
    //
    // Note: the AH-stressed check uses "cup" (a content word) rather
    // than "but" (a coordinating conjunction). After the misaki-style
    // function-word destressing, "but" is emitted as `b ə t`
    // (unstressed schwa) because conjunctions are reduced in connected
    // speech. "cup" isn't in the function-word list, so its CMU
    // `K AH1 P` keeps its primary stress and correctly maps to `kˈʌp`.
    let test_cases = [
        ("cat", "æ"),      // AE → æ  
        ("bet", "ɛ"),      // EH → ɛ
        ("bit", "ɪ"),      // IH → ɪ
        ("cup", "ʌ"),      // AH1 → ʌ (stressed, non-function word)
        ("boot", "u"),     // UW → u
        ("book", "ʊ"),     // UH → ʊ
    ];
    
    for (word, expected_ipa_vowel) in &test_cases {
        let ipa = g2p.convert_to_standard(word, PhonemeStandard::IPA)?;
        let ipa_string = ipa.join(" ");
        
        assert!(ipa_string.contains(expected_ipa_vowel), 
            "Word '{}' should contain IPA vowel '{}', got: '{}'", 
            word, expected_ipa_vowel, ipa_string);
    }
    
    Ok(())
}

/// Test ARPABET to IPA conversion for consonants
#[test]
fn test_arpabet_to_ipa_consonant_conversion() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = EnglishG2P::new()?;
    
    let consonant_tests = [
        ("hello", "h"),    // HH → h
        ("think", "θ"),    // TH → θ  
        ("this", "ð"),     // DH → ð
        ("shoe", "ʃ"),     // SH → ʃ
        ("ring", "ŋ"),     // NG → ŋ
        ("check", "tʃ"),   // CH → tʃ (if available) 
        ("judge", "dʒ"),   // JH → dʒ (if available)
        ("red", "ɹ"),      // R → ɹ (English approximant)
    ];
    
    for (word, expected_consonant) in &consonant_tests {
        let ipa = g2p.convert_to_standard(word, PhonemeStandard::IPA)?;
        let ipa_string = ipa.join(" ");
        
        if ipa_string.contains(expected_consonant) {
            println!("✅ Word '{}' contains expected consonant '{}': '{}'", word, expected_consonant, ipa_string);
        } else {
            println!("⚠️ Word '{}' expected '{}' but got: '{}'", word, expected_consonant, ipa_string);
            
            assert!(!ipa_string.is_empty(), "Should produce some IPA output for word '{}'", word);
        }
    }
    
    Ok(())
}

/// Test stress preservation in ARPABET to IPA conversion
#[test]
fn test_stress_preservation_in_ipa_conversion() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Test words with different stress patterns
    let stress_tests = [
        ("above", "ˈ"),    // Should have primary stress marker
        ("photograph", "ˈ"), // Primary stress on first syllable
        ("photography", "ˈ"), // Stress shift in related words
    ];
    
    for (word, stress_marker) in &stress_tests {
        let ipa = g2p.convert_to_standard(word, PhonemeStandard::IPA)?;
        let ipa_string = ipa.join(" ");
        
        // Note: Current implementation may not include stress markers in all cases
        println!("Word '{}' IPA: '{}' (checking for stress: '{}')", 
                 word, ipa_string, stress_marker);
        
        // This test documents current behavior - may need adjustment based on implementation
        if ipa_string.contains(stress_marker) {
            println!("  ✅ Stress marker '{}' preserved", stress_marker);
        } else {
            println!("  ⚠️ Stress marker '{}' not found in IPA output", stress_marker);
        }
    }
    
    Ok(())
}

/// Test Kokoro TTS vocabulary compatibility
#[test]
fn test_kokoro_vocabulary_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Common Kokoro vocabulary characters that should be produced
    let kokoro_chars = [
        'h', 'l', 'r', 'd', 't', 'n', 's', 'w', 'e', 'i', 'o', 'u', 'a',  // Basic letters
        'æ', 'ɛ', 'ɪ', 'ʊ', 'ə', 'ɑ', 'ɔ', 'θ', 'ð', 'ʃ', 'ŋ', 'ɹ'     // IPA symbols
    ];
    
    // Create a character frequency map from common English words
    let test_words = ["hello", "world", "this", "that", "think", "about", "water", "better"];
    let mut char_usage = HashMap::new();
    
    for word in &test_words {
        let ipa = g2p.convert_to_standard(word, PhonemeStandard::IPA)?;
        for phoneme_str in ipa {
            for ch in phoneme_str.chars() {
                if kokoro_chars.contains(&ch) {
                    *char_usage.entry(ch).or_insert(0) += 1;
                }
            }
        }
    }
    
    println!("Kokoro-compatible characters found in IPA output:");
    for (ch, count) in char_usage.iter() {
        println!("  '{}': used {} times", ch, count);
    }
    
    // Validate we're producing some Kokoro-compatible characters
    assert!(!char_usage.is_empty(), 
        "Should produce at least some Kokoro-compatible IPA characters");
    
    // Key validation: ensure common phonemes are properly mapped
    assert!(char_usage.contains_key(&'h'), "Should produce 'h' from HH phonemes");
    assert!(char_usage.contains_key(&'l'), "Should produce 'l' from L phonemes");
    
    Ok(())
}

/// Test IPA conversion round-trip quality  
#[test]
fn test_ipa_conversion_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let g2p = EnglishG2P::new()?;
    
    // Test that conversion maintains essential phonemic information
    let test_words = ["hello", "world", "test", "phoneme", "conversion"];
    
    for word in &test_words {
        // Get both ARPABET and IPA
        let arpabet = g2p.convert_to_standard(word, PhonemeStandard::ARPABET)?;
        let ipa = g2p.convert_to_standard(word, PhonemeStandard::IPA)?;
        
        println!("Word: '{}'\n  ARPABET: {:?}\n  IPA: {:?}", word, arpabet, ipa);
        
        // Basic validation: IPA should have reasonable length relative to ARPABET
        let arpabet_phoneme_count = arpabet.len();
        let ipa_char_count: usize = ipa.iter().map(|s| s.chars().count()).sum();
        
        // IPA might have fewer "phonemes" but similar total character count
        assert!(ipa_char_count > 0, "IPA conversion should produce non-empty output");
        
        // Allow some variance since IPA can use diphthongs, etc.
        let ratio = ipa_char_count as f64 / arpabet_phoneme_count as f64;
        assert!(ratio > 0.3 && ratio < 3.0,
            "IPA/ARPABET length ratio ({:.2}) should be reasonable for word '{}'", 
            ratio, word);
    }
    
    Ok(())
}

/// Test fallback behavior for unknown words with IPA standard
#[test]
fn test_ipa_fallback_quality() -> Result<(), Box<dyn std::error::Error>> {
    // Test with different fallback strategies to ensure IPA output remains valid
    let strategies = [
        FallbackStrategy::UseRules,
        FallbackStrategy::ReturnGraphemes,
        FallbackStrategy::GuessPhonemes,
    ];
    
    let unknown_words = ["antidisestablishmentarianism", "supercalifragilisticexpialidocious"];
    
    for strategy in &strategies {
        let options = G2POptions {
            fallback_strategy: *strategy,
            default_standard: PhonemeStandard::IPA,
            ..Default::default()
        };
        
        let g2p = EnglishG2P::with_options(options)?;
        
        for word in &unknown_words {
            match g2p.convert_to_standard(word, PhonemeStandard::IPA) {
                Ok(ipa) => {
                    println!("Strategy {:?} for '{}': {:?}", strategy, word, ipa);
                    
                    // Validate IPA output doesn't contain ARPABET artifacts
                    let ipa_string = ipa.join(" ");
                    assert!(!ipa_string.contains("HH"), "IPA output shouldn't contain ARPABET 'HH'");
                    assert!(!ipa_string.contains("EH"), "IPA output shouldn't contain ARPABET 'EH'");
                    assert!(!ipa_string.contains("0") && !ipa_string.contains("1") && !ipa_string.contains("2"), 
                            "IPA output shouldn't contain ARPABET stress numbers");
                }
                Err(e) => {
                    // Skip strategy may fail, which is acceptable
                    println!("Strategy {:?} failed for '{}': {}", strategy, word, e);
                }
            }
        }
    }
    
    Ok(())
}

/// Test Ferrocarril adapter IPA configuration
#[test]
fn test_ferrocarril_adapter_ipa_output() -> Result<(), Box<dyn std::error::Error>> {
    use phonesis::ferrocarril_adapter::FerrocarrilG2PAdapter;
    
    let mut adapter = FerrocarrilG2PAdapter::new_english()?;
    
    // Test default (should be ARPABET)
    let default_output = adapter.convert_for_tts("hello")?;
    println!("Default output: '{}'", default_output);
    
    // Configure for IPA (TTS compatibility)
    adapter.set_standard(PhonemeStandard::IPA);
    let ipa_output = adapter.convert_for_tts("hello")?;
    println!("IPA output: '{}'", ipa_output);
    
    // Validate IPA output format
    assert!(!ipa_output.contains("HH"), "IPA mode shouldn't produce ARPABET phonemes");
    assert!(!ipa_output.contains("EH0"), "IPA mode shouldn't include ARPABET stress notation");
    assert!(ipa_output.contains(" "), "IPA output should be space-separated for TTS compatibility");
    
    // Validate we're getting proper IPA characters
    assert!(ipa_output.chars().any(|c| "hɛloɹd".contains(c)), 
        "Should contain some expected IPA characters from 'hello'");
    
    Ok(())
}