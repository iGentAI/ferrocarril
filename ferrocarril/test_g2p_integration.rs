use ferrocarril_core::PhonesisG2P;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Ferrocarril G2P Integration");
    
    let mut g2p = PhonesisG2P::new("en-us")?;
    println!("Created PhonesisG2P successfully");
    
    let test_phrases = vec![
        "Hello world",
        "This is a test",
        "The quick brown fox jumps", 
        "3.14 is pi",
        "I am scheduled to meet at 3 PM";
    ];
    
    for phrase in test_phrases {
        println!("\nConverting: '{}'", phrase);
        match g2p.convert(phrase) {
            Ok(phonemes) => {
                println!("Phonemes: {}", phonemes);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    
    println!("\nG2P integration test completed successfully!");
    Ok(())
}
