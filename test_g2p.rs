use phonesis::{GraphemeToPhoneme, english::EnglishG2P};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the G2P system
    println!("Initializing Phonesis G2P...");
    let g2p = EnglishG2P::new()?;
    
    // Test cases
    let test_cases = [
        "Hello, world!",
        "Text to speech",
        "Ferrocarril",
        "TTS",
        "Kokoro is a Text-to-Speech system.",
        "Unknown words like antidisestablishmentarianism should be handled."
    ];
    
    for test in &test_cases {
        println!("\nInput: \"{}\"", test);
        
        match g2p.convert(test) {
            Ok(phonemes) => {
                println!("Success! Phonemes:");
                let phoneme_strings: Vec<String> = phonemes.iter().map(|p| p.to_string()).collect();
                println!("{}", phoneme_strings.join(" "));
            },
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    
    Ok(())
}