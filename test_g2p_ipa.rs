//! Test Phonesis G2P for proper IPA output

use ferrocarril_core::PhonesisG2P;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🧪 TESTING PHONESIS G2P - IPA OUTPUT VERIFICATION");
    println!("=" * 50);
    
    let g2p = PhonesisG2P::new("en-us")?;
    
    let test_cases = vec![
        "hello",
        "world", 
        "hello world",
        "TTS",
        "Kokoro",
        "the quick brown fox"
    ];
    
    println!("\n📝 Testing G2P conversion to IPA phonemes:");
    
    for text in test_cases {
        println!("\nInput: '{}'", text);
        
        match g2p.convert(text) {
            Ok(phonemes) => {
                println!("  Phonemes: '{}'", phonemes);
                
                // Check if output looks like IPA
                let has_ipa_chars = phonemes.chars().any(|c| {
                    matches!(c, 'ʌ' | 'ə' | 'ɛ' | 'ɪ' | 'ʊ' | 'ɑ' | 'ɔ' | 'æ' | 'ɝ' | 'ɚ' | 'ŋ' | 'θ' | 'ð' | 'ʃ' | 'ʒ')
                });
                
                if has_ipa_chars {
                    println!("  ✅ Contains IPA characters");
                } else {
                    println!("  ⚠️  No obvious IPA characters detected");
                }
                
                // Check phoneme count vs input length  
                let phoneme_count = phonemes.split_whitespace().count();
                let char_count = text.chars().filter(|c| c.is_alphabetic()).count();
                println!("  Phonemes: {} tokens, Input chars: {}", phoneme_count, char_count);
                
                // Detailed phoneme breakdown
                println!("  Individual phonemes: {:?}", 
                    phonemes.split_whitespace().collect::<Vec<_>>());
                    
            }
            Err(e) => {
                println!("  ❌ Error: {}", e);
            }
        }
    }
    
    println!("\n🎯 G2P IPA OUTPUT TEST COMPLETE");
    
    Ok(())
}
