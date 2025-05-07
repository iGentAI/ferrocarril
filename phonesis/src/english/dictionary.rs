//! English dictionary data for the Phonesis G2P system
//!
//! This module provides a default embedded dictionary for English words,
//! using data converted from WikiPron (CC0/public domain license).

use std::sync::Once;
use std::sync::Arc;

use crate::{
    dictionary::PronunciationDictionary,
    phoneme::PhonemeStandard,
    error::{G2PError, Result},
};

// Static dictionary initialization
static INIT_DICTIONARY: Once = Once::new();
static mut DEFAULT_DICTIONARY: Option<Arc<PronunciationDictionary>> = None;

// Include the entire dictionary data as a constant string
// This embeds the dictionary into the binary at compile time
include!("../../embedded_dictionary_data.rs");

/// Get the default English pronunciation dictionary.
pub fn get_default_dictionary() -> Result<Arc<PronunciationDictionary>> {
    unsafe {
        INIT_DICTIONARY.call_once(|| {
            // Initialize dictionary with embedded data
            match PronunciationDictionary::from_cmu_str(EMBEDDED_WIKIPRON_DICTIONARY, "en-us") {
                Ok(dict) => {
                    DEFAULT_DICTIONARY = Some(Arc::new(dict));
                },
                Err(e) => {
                    eprintln!("Failed to load embedded dictionary: {}", e);
                    // Initialize with empty dictionary as fallback
                    let empty_dict = PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET);
                    DEFAULT_DICTIONARY = Some(Arc::new(empty_dict));
                }
            }
        });
        
        match &DEFAULT_DICTIONARY {
            Some(dict) => Ok(Arc::clone(dict)),
            None => Err(G2PError::DictionaryError("Default dictionary not initialized".to_string())),
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