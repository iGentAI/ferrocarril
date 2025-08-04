use phonesis::{english::EnglishG2P, GraphemeToPhoneme, PhonemeStandard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 PHONESIS G2P IPA OUTPUT TEST");
    let g2p = EnglishG2P::new()?;
    
    let test_cases = ["hello", "world", "hello world", "TTS", "Kokoro"];
    
    for word in &test_cases {
        println!("\nInput: '{}'", word);
        
        // Test ARPABET output
        match g2p.convert_to_standard(word, PhonemeStandard::ARPABET) {
            Ok(phonemes) => {
                println!("  ARPABET: {:?}", phonemes);
                println!("  ARPABET joined: {}", phonemes.join(" "));
            }
            Err(e) => println!("  ARPABET error: {}", e)
        }
        
        // Test IPA output
        match g2p.convert_to_standard(word, PhonemeStandard::IPA) {
            Ok(phonemes) => {
                println!("  IPA: {:?}", phonemes);
                println!("  IPA joined: {}", phonemes.join(" "));
                
                // Check for IPA characters
                let ipa_str = phonemes.join(" ");
                let has_ipa = ipa_str.chars().any(|c| matches!(c, 'ʌ' | 'ə' | 'ɛ' | 'ɪ' | 'ʊ' | 'ɑ' | 'ɔ' | 'æ' | 'ɝ'));
                if has_ipa {
                    println!("  ✅ Contains IPA symbols");
                } else {
                    println!("  ⚠️  No IPA symbols detected");
                }
            }
            Err(e) => println!("  IPA error: {}", e)
        }
    }
    
    Ok(())
}
