use phonesis::{english::EnglishG2P, GraphemeToPhoneme, PhonemeStandard};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 PHONESIS COMPREHENSIVE IPA RANGE TEST");
    let g2p = EnglishG2P::new()?;
    
    // Test diverse English words to see IPA range
    let test_words = [
        "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "think", "this", "that", "there", "these", "those", 
        "phoneme", "through", "tough", "plough", "cough", "enough",
        "church", "judge", "ship", "measure", "ring", "sing"
    ];
    
    let mut all_phonemes_found = std::collections::HashSet::new();
    
    for word in &test_words {
        match g2p.convert_to_standard(word, PhonemeStandard::IPA) {
            Ok(phonemes) => {
                for phoneme in &phonemes {
                    all_phonemes_found.insert(phoneme.clone());
                }
                println!("{}\t→\t{}", word, phonemes.join(" "));
            }
            Err(e) => println!("{}: {}", word, e)
        }
    }
    
    println!("\n📊 PHONESIS IPA RANGE SUMMARY:");
    println!("Total unique IPA symbols produced: {}", all_phonemes_found.len());
    println!("\nAll IPA symbols found:");
    let mut sorted_phonemes: Vec<_> = all_phonemes_found.into_iter().collect();
    sorted_phonemes.sort();
    for (i, phoneme) in sorted_phonemes.iter().enumerate() {
        print!("{} ", phoneme);
        if (i + 1) % 10 == 0 {
            println!();
        }
    }
    println!();
    
    Ok(())
}
