//!
//! This module provides the dictionary functionality for the Phonesis library,
//! including the `PronunciationDictionary` struct for efficient word lookup.

mod trie;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{G2PError, Result};
use crate::phoneme::{Phoneme, PhonemeSequence, PhonemeStandard};

pub use trie::CompactTrie;

/// Dictionary for pronunciation lookups.
///
/// Provides efficient lookup of word pronunciations using a compressed trie data structure.
#[derive(Debug, Clone)]
pub struct PronunciationDictionary {
    /// The underlying trie structure
    trie: CompactTrie,
    
    /// Language of this dictionary
    language: String,
    
    /// Phoneme standard used in this dictionary
    standard: PhonemeStandard,
    
    /// Additional metadata
    metadata: DictionaryMetadata,
}

/// Metadata for a pronunciation dictionary.
#[derive(Debug, Clone)]
pub struct DictionaryMetadata {
    /// Name of the dictionary
    pub name: String,
    
    /// Version of the dictionary
    pub version: String,
    
    /// Description of the dictionary
    pub description: String,
    
    /// Number of entries in the dictionary
    pub entry_count: usize,
    
    /// Source of the dictionary
    pub source: String,
    
    /// License of the dictionary data
    pub license: Option<String>,
    
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl Default for DictionaryMetadata {
    fn default() -> Self {
        Self {
            name: "Default Dictionary".to_string(),
            version: "0.1.0".to_string(),
            description: "A pronunciation dictionary".to_string(),
            entry_count: 0,
            source: "Unknown".to_string(),
            license: None,
            properties: HashMap::new(),
        }
    }
}

impl PronunciationDictionary {
    /// Create a new empty dictionary.
    pub fn new(language: &str, standard: PhonemeStandard) -> Self {
        Self {
            trie: CompactTrie::new(),
            language: language.to_string(),
            standard,
            metadata: DictionaryMetadata::default(),
        }
    }
    
    
    /// Create a dictionary from a text file in CMU format.
    pub fn from_cmu_file<P: AsRef<Path> + Clone>(path: P, language: &str) -> Result<Self> {
        let file = File::open(path.clone()).map_err(|e| {
            G2PError::IoError(format!("Failed to open CMU dictionary file: {}", e))
        })?;
        
        let reader = BufReader::new(file);
        let mut dict = Self::new(language, PhonemeStandard::ARPABET);
        
        for line in reader.lines() {
            let line = line.map_err(|e| {
                G2PError::IoError(format!("Error reading line: {}", e))
            })?;
            
            // Skip comments and empty lines
            if line.starts_with(";;;") || line.trim().is_empty() {
                continue;
            }
            
            // Parse line: WORD  P1 P2 P3 ...
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            // Handle duplicate markers in CMU dict (WORD(1), WORD(2), etc.)
            let mut word = parts[0].to_string();
            if word.ends_with(')') {
                if let Some(idx) = word.rfind('(') {
                    word = word[..idx].to_string();
                }
            }
            
            // Parse phonemes
            let mut phonemes = Vec::new();
            for p in &parts[1..] {
                let mut p_str = p.to_string();
                let stress = if p_str.ends_with('0') || p_str.ends_with('1') || p_str.ends_with('2') {
                    let stress_char = p_str.pop().unwrap();
                    match stress_char {
                        '1' => Some(crate::phoneme::StressLevel::Primary),
                        '2' => Some(crate::phoneme::StressLevel::Secondary),
                        '0' => Some(crate::phoneme::StressLevel::Unstressed),
                        _ => None,
                    }
                } else {
                    None
                };
                
                phonemes.push(Phoneme::new(p_str, stress));
            }
            
            // Insert into dictionary (first-wins: skip if already present)
            let lc_word = word.to_lowercase();
            if !dict.contains(&lc_word) {
                dict.insert(&lc_word, PhonemeSequence::new(phonemes));
            }
        }
        
        // Update metadata
        dict.metadata.entry_count = dict.len();
        dict.metadata.source = format!("CMU dictionary from {}", path.as_ref().display());
        
        Ok(dict)
    }

    /// Create a dictionary from CMU format text.
    pub fn from_cmu_str(text: &str, language: &str) -> Result<Self> {
        let mut dict = Self::new(language, PhonemeStandard::ARPABET);
        
        for line in text.lines() {
            // Skip comments and empty lines
            if line.starts_with(";;;") || line.trim().is_empty() {
                continue;
            }
            
            // Parse line: WORD  P1 P2 P3 ...
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            // Handle duplicate markers in CMU dict (WORD(1), WORD(2), etc.)
            let mut word = parts[0].to_string();
            if word.ends_with(')') {
                if let Some(idx) = word.rfind('(') {
                    word = word[..idx].to_string();
                }
            }
            
            // Parse phonemes
            let mut phonemes = Vec::new();
            for p in &parts[1..] {
                let mut p_str = p.to_string();
                let stress = if p_str.ends_with('0') || p_str.ends_with('1') || p_str.ends_with('2') {
                    let stress_char = p_str.pop().unwrap();
                    match stress_char {
                        '1' => Some(crate::phoneme::StressLevel::Primary),
                        '2' => Some(crate::phoneme::StressLevel::Secondary),
                        '0' => Some(crate::phoneme::StressLevel::Unstressed),
                        _ => None,
                    }
                } else {
                    None
                };
                
                phonemes.push(Phoneme::new(p_str, stress));
            }
            
            // Insert into dictionary (first-wins: skip if already present)
            let lc_word = word.to_lowercase();
            if !dict.contains(&lc_word) {
                dict.insert(&lc_word, PhonemeSequence::new(phonemes));
            }
        }
        
        // Update metadata
        dict.metadata.entry_count = dict.len();
        dict.metadata.source = "Embedded dictionary".to_string();
        
        Ok(dict)
    }
    
    /// Look up the pronunciation of a word.
    pub fn lookup(&self, word: &str) -> Option<&PhonemeSequence> {
        self.trie.get(&word.to_lowercase())
    }
    
    /// Insert a word and its pronunciation into the dictionary.
    pub fn insert(&mut self, word: &str, pronunciation: PhonemeSequence) {
        self.trie.insert(&word.to_lowercase(), pronunciation);
        self.metadata.entry_count = self.trie.len();
    }
    
    /// Check if the dictionary contains a word.
    pub fn contains(&self, word: &str) -> bool {
        self.trie.contains(&word.to_lowercase())
    }
    
    /// Get the number of entries in the dictionary.
    pub fn len(&self) -> usize {
        self.trie.len()
    }
    
    /// Returns true if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }
    
    /// Get the language of this dictionary.
    pub fn language(&self) -> &str {
        &self.language
    }
    
    /// Get the phoneme standard used in this dictionary.
    pub fn standard(&self) -> PhonemeStandard {
        self.standard
    }
    
    /// Get a reference to the metadata for this dictionary.
    pub fn metadata(&self) -> &DictionaryMetadata {
        &self.metadata
    }
    
    /// Set the metadata for this dictionary.
    pub fn set_metadata(&mut self, metadata: DictionaryMetadata) {
        self.metadata = metadata;
    }
    
}

/// Create a small in-memory dictionary with common English words.
pub fn create_small_dict() -> PronunciationDictionary {
    // Include a small set of common English words
    let cmu_data = r#"
THE DH AH0
A AH0
AND AE1 N D
TO T UW1
OF AH1 V
IN IH0 N
THAT DH AE1 T
IS IH1 Z
IT IH1 T
FOR F AO1 R
YOU Y UW1
HE HH IY1
HAVE HH AE1 V
WITH W IH1 DH
ON AA1 N
THIS DH IH1 S
BE B IY1
AT AE1 T
BUT B AH1 T
NOT N AA1 T
BY B AY1
FROM F R AH1 M
THEY DH EY1
WE W IY1
SAY S EY1
HER HH ER0
SHE SH IY1
OR AO1 R
AN AE1 N
WILL W IH1 L
MY M AY1
ONE W AH1 N
OUT AW1 T
IF IH1 F
ABOUT AH0 B AW1 T
WHO HH UW1
GET G EH1 T
WHICH W IH1 CH
GO G OW1
HELLO HH EH1 L OW2
WORLD W ER1 L D
"#;

    PronunciationDictionary::from_cmu_str(cmu_data, "en-us").unwrap_or_else(|_| {
        PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_dictionary() {
        let dict = PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET);
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert!(!dict.contains("hello"));
    }
    
    #[test]
    fn test_insert_and_lookup() {
        let mut dict = PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET);
        
        // Create a phoneme sequence
        let phonemes = vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(crate::phoneme::StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", None),
        ];
        
        dict.insert("hello", PhonemeSequence::new(phonemes));
        
        // Check dictionary state
        assert!(!dict.is_empty());
        assert_eq!(dict.len(), 1);
        assert!(dict.contains("hello"));
        
        // Look up pronunciation
        let result = dict.lookup("hello").unwrap();
        assert_eq!(result.phonemes[0].symbol, "HH");
        assert_eq!(result.phonemes[1].symbol, "EH");
        assert_eq!(result.phonemes[1].stress, Some(crate::phoneme::StressLevel::Primary));
    }
    
    #[test]
    fn test_from_cmu_str() {
        let cmu_data = r#"
;;; Sample CMU dictionary
HELLO  HH EH1 L OW0
WORLD  W ER1 L D
TEST  T EH1 S T
"#;

        let dict = PronunciationDictionary::from_cmu_str(cmu_data, "en-us").unwrap();
        
        assert_eq!(dict.len(), 3);
        assert!(dict.contains("HELLO"));
        assert!(dict.contains("WORLD"));
        assert!(dict.contains("TEST"));
    }
    
    #[test]
    fn test_create_small_dict() {
        let dict = create_small_dict();
        
        assert!(!dict.is_empty());
        assert!(dict.contains("HELLO"));
        assert!(dict.contains("WORLD"));
        assert!(dict.contains("THE"));
        assert!(dict.contains("AND"));
    }
    
    #[test]
    fn test_case_insensitivity() {
        let mut dict = PronunciationDictionary::new("en-us", PhonemeStandard::ARPABET);
        
        // Create a phoneme sequence
        let phonemes = vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(crate::phoneme::StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", None),
        ];
        
        // Insert with mixed case
        dict.insert("Hello", PhonemeSequence::new(phonemes));
        
        // Check with different cases
        assert!(dict.contains("hello"));
        assert!(dict.contains("Hello"));
        assert!(dict.contains("HELLO"));
        assert!(dict.contains("HeLLo"));
    }
}