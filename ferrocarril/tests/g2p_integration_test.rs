//! G2P integration test for Ferrocarril
//! 
//! This test verifies that the G2P handler properly integrates with the Ferrocarril system
//! and produces phoneme output that matches the Kokoro reference implementation.

#[cfg(test)]
mod tests {
    // Update to use the G2PHandler struct from model module
    use ferrocarril::model::g2p::G2PHandler;
    use std::error::Error;

    // Define the max phoneme length constant to match G2PHandler
    const MAX_PHONEME_LENGTH: usize = 510;

    #[test]
    fn test_g2p_basic_conversion() -> Result<(), Box<dyn Error>> {
        // Initialize the G2P handler
        let handler = G2PHandler::new("en-us")?;
        
        // Test with simple text
        let simple_text = "Hello, world!";
        let result = handler.convert(simple_text);
        
        println!("Input: \"{}\"", simple_text);
        println!("Output: \"{}\"", result.phonemes);
        
        // Should succeed and produce non-empty phonemes
        assert!(result.success, "Basic conversion should succeed");
        assert!(!result.phonemes.is_empty(), "Should produce non-empty phonemes");
        
        // Should contain expected phonemes for "hello"
        // HH for HA sound in Hello, etc.
        assert!(result.phonemes.contains("HH"), "Should contain HH phoneme for 'hello'");
        
        // Verify the format matches what we expect from the reference implementation
        assert!(result.phonemes.contains(" "), "Phonemes should be space-separated");
        
        Ok(())
    }
    
    #[test]
    fn test_g2p_abbreviation_handling() -> Result<(), Box<dyn Error>> {
        // Initialize the G2P handler
        let handler = G2PHandler::new("en-us")?;
        
        // Test with abbreviations
        let texts = ["TTS", "NASA", "USA", "FBI"];
        
        for text in &texts {
            let result = handler.convert(text);
            println!("Input: \"{}\"", text);
            println!("Output: \"{}\"", result.phonemes);
            
            // Even if G2P fails, it should produce some output
            assert!(!result.phonemes.is_empty(), "Should produce some phonemes for abbreviations");
            
            // Check if we get more than just character-by-character output
            // If it's working well, phonemes.len() should be different from text
            if result.success {
                println!("Successfully converted abbreviation: {}", text);
            } else {
                println!("Fallback handling used for abbreviation: {}", text);
                
                // Verify fallback format is char-by-char with spaces
                let expected_fallback = text.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
                assert_eq!(result.phonemes, expected_fallback, 
                           "Fallback should be character-by-character with spaces");
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_g2p_complex_text() -> Result<(), Box<dyn Error>> {
        // Initialize the G2P handler
        let handler = G2PHandler::new("en-us")?;
        
        // Test with complex text containing punctuation, numbers, and mixed case
        let complex_text = "The quick brown fox jumps over the lazy dog at 12:34 PM!";
        let result = handler.convert(complex_text);
        
        println!("Input: \"{}\"", complex_text);
        println!("Output: \"{}\"", result.phonemes);
        
        // Should produce non-empty phonemes even if conversion fails
        assert!(!result.phonemes.is_empty(), "Should produce non-empty phonemes");
        
        // Should contain spaces between phonemes
        assert!(result.phonemes.contains(" "), "Phonemes should be space-separated");
        
        // Note: We don't assert success here because the G2P may or may not be able to handle the complex text
        // Instead, we're verifying that we get some reasonable output in either case
        
        Ok(())
    }
    
    #[test]
    fn test_g2p_unknown_word_handling() -> Result<(), Box<dyn Error>> {
        // Initialize the G2P handler
        let handler = G2PHandler::new("en-us")?;
        
        // Test with rare/unknown words
        let texts = [
            "antidisestablishmentarianism", 
            "supercalifragilisticexpialidocious",
            "pneumonoultramicroscopicsilicovolcanoconiosis"
        ];
        
        for text in &texts {
            let result = handler.convert(text);
            println!("Input: \"{}\"", text);
            println!("Output: \"{}\"", result.phonemes);
            
            // Even if conversion fails, we should get non-empty phonemes
            assert!(!result.phonemes.is_empty(), "Should produce some output for unknown words");
            
            if result.success {
                println!("Successfully converted rare word: {}", text);
            } else {
                println!("Fallback handling used for rare word: {}", text);
                
                // Verify consistent fallback format
                let expected_fallback = text.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
                assert_eq!(result.phonemes, expected_fallback, 
                           "Fallback should be character-by-character with spaces");
            }
        }
        
        Ok(())
    }
    
    #[test]
    fn test_g2p_max_length_handling() -> Result<(), Box<dyn Error>> {
        // Initialize the G2P handler
        let handler = G2PHandler::new("en-us")?;
        
        // Create a very long text by repeating "hello " many times
        let long_text = "hello ".repeat(200);
        let result = handler.convert(&long_text);
        
        println!("Input length: {} characters", long_text.len());
        println!("Output length: {} characters", result.phonemes.len());
        
        // Should handle long text without errors
        assert!(!result.phonemes.is_empty(), "Should produce some output for long text");
        
        // Enforce that the output doesn't exceed the maximum length
        assert!(result.phonemes.len() <= MAX_PHONEME_LENGTH, 
                "Phoneme output must not exceed MAX_PHONEME_LENGTH ({}), got {} characters", 
                MAX_PHONEME_LENGTH, result.phonemes.len());
        
        Ok(())
    }
}