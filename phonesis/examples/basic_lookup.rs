// of the crate they're running in - they're compiling against the published API
use phonesis::{
    PronunciationDictionary,
    Phoneme, 
    PhonemeSequence, 
    PhonemeStandard, 
    StressLevel, 
    PhonemeType
};
use std::env;

fn main() {
    println!("Phonesis - Basic Dictionary Lookup Example");
    println!("===========================================\n");

    // Create a new dictionary
    let mut dict = PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET);
    
    // Add some words to the dictionary
    add_example_words(&mut dict);
    
    // Get the word to look up from command line args or use default
    let word = env::args().nth(1).unwrap_or_else(|| "hello".to_string());
    
    // Look up the word
    println!("Looking up pronunciation for: '{}'", word);
    match dict.lookup(&word) {
        Some(pronunciation) => {
            println!("Found pronunciation: {}", pronunciation);
            
            println!("\nIndividual phonemes:");
            for (i, phoneme) in pronunciation.phonemes.iter().enumerate() {
                println!("  {}: {} ({})", 
                    i + 1,
                    phoneme,
                    match phoneme.phoneme_type {
                        PhonemeType::Vowel => "vowel",
                        PhonemeType::Consonant => "consonant",
                        PhonemeType::Diphthong => "diphthong",
                        PhonemeType::Special => "special",
                    }
                );
            }
            
            // Convert to IPA
            println!("\nIPA representation:");
            for phoneme in &pronunciation.phonemes {
                print!("{} ", phoneme.to_string_in(PhonemeStandard::IPA));
            }
            println!();
        },
        None => {
            println!("Word not found in dictionary.");
        },
    }
    
    // Print dictionary statistics
    println!("\nDictionary statistics:");
    println!("Language: {}", dict.language());
    println!("Standard: {:?}", dict.standard());
    println!("Entries: {}", dict.len());
    println!("Empty: {}", dict.is_empty());
}

fn add_example_words(dict: &mut PronunciationDictionary) {
    // hello
    let hello = PhonemeSequence::new(vec![
        Phoneme::new("HH", None),
        Phoneme::new("AH", Some(StressLevel::Unstressed)),
        Phoneme::new("L", None),
        Phoneme::new("OW", Some(StressLevel::Primary)),
    ]);
    dict.insert("hello", hello);
    
    // world
    let world = PhonemeSequence::new(vec![
        Phoneme::new("W", None),
        Phoneme::new("ER", Some(StressLevel::Primary)),
        Phoneme::new("L", None),
        Phoneme::new("D", None),
    ]);
    dict.insert("world", world);
    
    // example
    let example = PhonemeSequence::new(vec![
        Phoneme::new("IH", Some(StressLevel::Secondary)),
        Phoneme::new("G", None),
        Phoneme::new("Z", None),
        Phoneme::new("AE", Some(StressLevel::Primary)),
        Phoneme::new("M", None),
        Phoneme::new("P", None),
        Phoneme::new("AH", Some(StressLevel::Unstressed)),
        Phoneme::new("L", None),
    ]);
    dict.insert("example", example);
    
    // pronunciation
    let pronunciation = PhonemeSequence::new(vec![
        Phoneme::new("P", None),
        Phoneme::new("R", None),
        Phoneme::new("AH", Some(StressLevel::Unstressed)),
        Phoneme::new("N", None),
        Phoneme::new("AH", Some(StressLevel::Secondary)),
        Phoneme::new("N", None),
        Phoneme::new("S", None),
        Phoneme::new("IY", Some(StressLevel::Unstressed)),
        Phoneme::new("EY", Some(StressLevel::Primary)),
        Phoneme::new("SH", None),
        Phoneme::new("AH", Some(StressLevel::Unstressed)),
        Phoneme::new("N", None),
    ]);
    dict.insert("pronunciation", pronunciation);
}