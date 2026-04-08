//! English dictionary data for the Phonesis G2P system
//!
//! Provides the default embedded dictionary for English words, built
//! from the CMU Pronouncing Dictionary (public domain). The dictionary
//! is lazily loaded once per process via `OnceLock`.

use std::sync::{Arc, OnceLock};

use crate::{
    dictionary::PronunciationDictionary,
    phoneme::PhonemeStandard,
    error::Result,
};

// Lazily-initialised process-global dictionary. `OnceLock` replaces the
// legacy `static mut DEFAULT_DICTIONARY + Once + unsafe` pattern from
// pre-Rust-1.70 code; it gives us the same "initialise exactly once"
// semantics with no unsafe and no shared-mutable-static warning.
static DEFAULT_DICTIONARY: OnceLock<Arc<PronunciationDictionary>> = OnceLock::new();

// Include the entire dictionary data as a constant string so the
// dictionary is embedded in the binary at compile time.
include!("../../embedded_dictionary_data.rs");

/// Get the default English pronunciation dictionary.
pub fn get_default_dictionary() -> Result<Arc<PronunciationDictionary>> {
    let dict = DEFAULT_DICTIONARY.get_or_init(|| {
        match PronunciationDictionary::from_cmu_str(EMBEDDED_CMU_DICTIONARY, "en-us") {
            Ok(mut dict) => {
                // Inject number words, days of the week, common
                // abbreviations, and a handful of tech terms that the
                // raw CMU dict either misses or mis-stresses for TTS.
                add_missing_essentials(&mut dict);
                Arc::new(dict)
            }
            Err(e) => {
                eprintln!("Failed to load embedded dictionary: {}", e);
                // Fall back to an empty dictionary so callers still get
                // a valid Arc and can rely on rule-based + character
                // fallback paths.
                Arc::new(PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET))
            }
        }
    });
    Ok(Arc::clone(dict))
}

/// Add missing essential words to the dictionary
fn add_missing_essentials(dict: &mut PronunciationDictionary) {
    use crate::phoneme::{Phoneme, PhonemeSequence, StressLevel};
    
    // Add missing number words (11-19)
    let numbers = vec![
        ("eleven", vec!["IH0", "L", "EH1", "V", "AH0", "N"]),
        ("twelve", vec!["T", "W", "EH1", "L", "V"]),
        ("thirteen", vec!["TH", "ER1", "T", "IY2", "N"]),
        ("fourteen", vec!["F", "AO1", "R", "T", "IY2", "N"]),
        ("fifteen", vec!["F", "IH1", "F", "T", "IY2", "N"]),
        ("sixteen", vec!["S", "IH1", "K", "S", "T", "IY2", "N"]),
        ("seventeen", vec!["S", "EH1", "V", "AH0", "N", "T", "IY2", "N"]),
        ("eighteen", vec!["EY1", "T", "IY2", "N"]),
        ("nineteen", vec!["N", "AY1", "N", "T", "IY2", "N"]),
        
        // Add tens
        ("twenty", vec!["T", "W", "EH1", "N", "T", "IY0"]),
        ("thirty", vec!["TH", "ER1", "D", "IY0"]),
        ("forty", vec!["F", "AO1", "R", "D", "IY0"]),
        ("fifty", vec!["F", "IH1", "F", "T", "IY0"]),
        ("sixty", vec!["S", "IH1", "K", "S", "T", "IY0"]),
        ("seventy", vec!["S", "EH1", "V", "AH0", "N", "T", "IY0"]),
        ("eighty", vec!["EY1", "D", "IY0"]),
        ("ninety", vec!["N", "AY1", "N", "D", "IY0"]),
        
        // Common words that might be missing
        ("hundred", vec!["HH", "AH1", "N", "D", "R", "AH0", "D"]),
        ("thousand", vec!["TH", "AW1", "Z", "AH0", "N", "D"]),
        ("million", vec!["M", "IH1", "L", "Y", "AH0", "N"]),
        ("billion", vec!["B", "IH1", "L", "Y", "AH0", "N"]),
        
        // Add "point" for decimal numbers
        ("point", vec!["P", "OY1", "N", "T"]),
        
        // Common pronouns that were missing
        ("i", vec!["AY1"]),
        ("me", vec!["M", "IY1"]),
        ("my", vec!["M", "AY1"]),
        ("mine", vec!["M", "AY1", "N"]),
        ("myself", vec!["M", "AY0", "S", "EH1", "L", "F"]),
        ("us", vec!["AH1", "S"]),
        ("our", vec!["AW1", "ER0"]),
        ("ours", vec!["AW1", "ER0", "Z"]),
        ("ourselves", vec!["AW0", "ER0", "S", "EH1", "L", "V", "Z"]),
        
        // Common abbreviations
        ("am", vec!["EY1", "EH1", "M"]),  // A.M.
        ("pm", vec!["P", "IY1", "EH1", "M"]),  // P.M.
        ("usa", vec!["Y", "UW1", "EH1", "S", "EY1"]),  // U.S.A.
        ("uk", vec!["Y", "UW1", "K", "EY1"]),  // U.K.
        ("eu", vec!["IY1", "Y", "UW1"]),  // E.U.
        ("fbi", vec!["EH1", "F", "B", "IY1", "AY1"]),  // F.B.I.
        ("cia", vec!["S", "IY1", "AY1", "EY1"]),  // C.I.A.
        ("phd", vec!["P", "IY1", "EY1", "CH", "D", "IY1"]),  // Ph.D.
        ("mr", vec!["M", "IH1", "S", "T", "ER0"]),  // Mr.
        ("mrs", vec!["M", "IH1", "S", "IH0", "Z"]),  // Mrs.
        ("ms", vec!["M", "IH1", "Z"]),  // Ms.
        ("dr", vec!["D", "AA1", "K", "T", "ER0"]),  // Dr.
        ("st", vec!["S", "T", "R", "IY1", "T"]),  // St. (street)
        ("ave", vec!["AE1", "V", "AH0", "N", "UW2"]),  // Ave. (avenue)
        ("blvd", vec!["B", "UH1", "L", "AH0", "V", "AA2", "R", "D"]),  // Blvd.
        ("jr", vec!["JH", "UW1", "N", "Y", "ER0"]),  // Jr.
        ("sr", vec!["S", "IY1", "N", "Y", "ER0"]),  // Sr.
        
        // Common verbs that might be missing
        ("scheduled", vec!["S", "K", "EH1", "JH", "UW0", "L", "D"]),
        ("scheduling", vec!["S", "K", "EH1", "JH", "UW0", "L", "IH0", "NG"]),
        ("schedule", vec!["S", "K", "EH1", "JH", "UW0", "L"]),
        
        // Days of the week  
        ("monday", vec!["M", "AH1", "N", "D", "EY2"]),
        ("tuesday", vec!["T", "UW1", "Z", "D", "EY2"]),
        ("wednesday", vec!["W", "EH1", "N", "Z", "D", "EY2"]),
        ("thursday", vec!["TH", "ER1", "Z", "D", "EY2"]),
        ("friday", vec!["F", "R", "AY1", "D", "EY2"]),
        ("saturday", vec!["S", "AE1", "T", "ER0", "D", "EY2"]),
        ("sunday", vec!["S", "AH1", "N", "D", "EY2"]),
        
        // Months (in case they're missing)
        ("january", vec!["JH", "AE1", "N", "Y", "UW0", "EH2", "R", "IY0"]),
        ("february", vec!["F", "EH1", "B", "R", "UW0", "EH2", "R", "IY0"]),
        ("march", vec!["M", "AA1", "R", "CH"]),
        ("april", vec!["EY1", "P", "R", "AH0", "L"]),
        ("may", vec!["M", "EY1"]),
        ("june", vec!["JH", "UW1", "N"]),
        ("july", vec!["JH", "UW0", "L", "AY1"]),
        ("august", vec!["AO1", "G", "AH0", "S", "T"]),
        ("september", vec!["S", "EH0", "P", "T", "EH1", "M", "B", "ER0"]),
        ("october", vec!["AA0", "K", "T", "OW1", "B", "ER0"]),
        ("november", vec!["N", "OW0", "V", "EH1", "M", "B", "ER0"]),
        ("december", vec!["D", "IH0", "S", "EH1", "M", "B", "ER0"]),
        
        // Tech abbreviations
        ("http", vec!["EY1", "CH", "T", "IY1", "T", "IY1", "P", "IY1"]),
        ("https", vec!["EY1", "CH", "T", "IY1", "T", "IY1", "P", "IY1", "EH1", "S"]),
        ("www", vec!["D", "AH1", "B", "AH0", "L", "Y", "UW2", "D", "AH1", "B", "AH0", "L", "Y", "UW2", "D", "AH1", "B", "AH0", "L", "Y", "UW2"]),
        ("com", vec!["K", "AA1", "M"]),
        ("org", vec!["AO1", "R", "G"]),
        ("net", vec!["N", "EH1", "T"]),
        ("edu", vec!["EH1", "D", "UW2"]),
        ("gov", vec!["G", "AH1", "V"]),
    ];
    
    for (word, phoneme_strs) in numbers {
        // Check if word already exists
        if !dict.contains(word) {
            let mut phonemes = Vec::new();
            for p_str in phoneme_strs {
                let (symbol, stress) = if p_str.ends_with('0') {
                    (p_str[..p_str.len()-1].to_string(), Some(StressLevel::Unstressed))
                } else if p_str.ends_with('1') {
                    (p_str[..p_str.len()-1].to_string(), Some(StressLevel::Primary))
                } else if p_str.ends_with('2') {
                    (p_str[..p_str.len()-1].to_string(), Some(StressLevel::Secondary))
                } else {
                    (p_str.to_string(), None)
                };
                phonemes.push(Phoneme::new(symbol, stress));
            }
            dict.insert(word, PhonemeSequence::new(phonemes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_dictionary() {
        let dict = get_default_dictionary().unwrap();
        
        // Check that the dictionary contains common words
        assert!(dict.contains("hello"));
        assert!(dict.contains("HELLO")); // Case-insensitive
        
        // Check pronunciation of key words
        let hello = dict.lookup("hello").unwrap();
        assert!(hello.phonemes.len() > 2);
        
        let world = dict.lookup("world").unwrap();
        assert!(world.phonemes.len() > 2);
    }
    
    #[test]
    fn test_dictionary_size() {
        let dict = get_default_dictionary().unwrap();
        
        // Should have loaded many words
        assert!(dict.len() > 50000);
        
        println!("Dictionary contains {} words", dict.len());
    }
}