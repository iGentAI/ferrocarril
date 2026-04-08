//!
//! This module defines the core `Phoneme` struct and related types for representing
//! phonetic units in different standards (ARPABET, IPA, etc.).

use std::fmt;
use std::collections::HashMap;
use crate::error::G2PError;

lazy_static::lazy_static! {
    static ref ARPABET_TO_IPA: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        // Vowels (stress-context-free; AH and ER are stress-dependent
        // and handled inline in `Phoneme::to_string_in`)
        m.insert("AA", "ɑ");
        m.insert("AE", "æ");
        m.insert("AO", "ɔ");
        m.insert("AX", "ə");
        m.insert("EH", "ɛ");
        m.insert("IH", "ɪ");
        m.insert("IY", "i");
        m.insert("UH", "ʊ");
        m.insert("UW", "u");

        // Diphthongs (Kokoro single-char uppercase tokens)
        m.insert("AW", "W");   // /aʊ/
        m.insert("AY", "I");   // /aɪ/
        m.insert("EY", "A");   // /eɪ/
        m.insert("OW", "O");   // /oʊ/
        m.insert("OY", "Y");   // /ɔɪ/

        // Consonants
        m.insert("B", "b");
        m.insert("CH", "ʧ");
        m.insert("D", "d");
        m.insert("DH", "ð");
        m.insert("F", "f");
        m.insert("G", "ɡ");
        m.insert("HH", "h");
        m.insert("JH", "ʤ");
        m.insert("K", "k");
        m.insert("L", "l");
        m.insert("M", "m");
        m.insert("N", "n");
        m.insert("NG", "ŋ");
        m.insert("P", "p");
        m.insert("R", "ɹ");
        m.insert("S", "s");
        m.insert("SH", "ʃ");
        m.insert("T", "t");
        m.insert("TH", "θ");
        m.insert("V", "v");
        m.insert("W", "w");
        m.insert("Y", "j");
        m.insert("Z", "z");
        m.insert("ZH", "ʒ");
        
        m
    };
    
    static ref IPA_TO_ARPABET: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        for (arpabet, ipa) in ARPABET_TO_IPA.iter() {
            m.insert(*ipa, *arpabet);
        }
        m
    };
}

/// Phoneme standard (notation system) for representing sounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonemeStandard {
    /// International Phonetic Alphabet
    IPA, 
    
    /// ARPABET (used by CMU Pronouncing Dictionary)
    ARPABET, 
    
    /// Speech Assessment Methods Phonetic Alphabet
    SAMPA, 
    
    /// Extended SAMPA
    XSAMPA,
    
    /// Custom standard
    Custom(u32),
}

impl Default for PhonemeStandard {
    fn default() -> Self {
        PhonemeStandard::ARPABET
    }
}

/// Stress level for a phoneme (primarily vowels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StressLevel {
    /// Primary stress
    Primary = 1,
    
    /// Secondary stress
    Secondary = 2,
    
    /// Unstressed
    Unstressed = 0,
}

impl From<u8> for StressLevel {
    fn from(value: u8) -> Self {
        match value {
            1 => StressLevel::Primary,
            2 => StressLevel::Secondary,
            _ => StressLevel::Unstressed,
        }
    }
}

impl From<StressLevel> for u8 {
    fn from(value: StressLevel) -> Self {
        match value {
            StressLevel::Primary => 1,
            StressLevel::Secondary => 2,
            StressLevel::Unstressed => 0,
        }
    }
}

/// Position of a phoneme in a word or syllable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonemePosition {
    /// Word or syllable initial
    Initial,
    
    /// Word or syllable medial
    Medial,
    
    /// Word or syllable final
    Final,
}

/// Type of phoneme (vowel, consonant, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonemeType {
    /// Vowel
    Vowel,
    
    /// Consonant
    Consonant,
    
    /// Diphthong (vowel combination)
    Diphthong,
    
    /// Special symbol (stress, syllable boundary, etc.)
    Special,
}

/// Representation of a single phonetic unit (phoneme).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Phoneme {
    /// The phoneme symbol
    pub symbol: String, 
    
    /// Stress level (primarily for vowels)
    pub stress: Option<StressLevel>, 
    
    /// Position within word or syllable
    pub position: Option<PhonemePosition>, 
    
    /// Phoneme standard
    pub standard: PhonemeStandard,
    
    /// Phoneme type
    pub phoneme_type: PhonemeType,
}

impl Phoneme {
    /// Create a new phoneme.
    pub fn new<S: Into<String>>(
        symbol: S, 
        stress: Option<StressLevel>
    ) -> Self {
        let symbol = symbol.into();
        let phoneme_type = Self::determine_type(&symbol);
        
        Self {
            symbol,
            stress,
            position: None,
            standard: PhonemeStandard::ARPABET,
            phoneme_type,
        }
    }
    
    /// Create a new phoneme with specific standard.
    pub fn new_with_standard<S: Into<String>>(
        symbol: S, 
        stress: Option<StressLevel>,
        standard: PhonemeStandard
    ) -> Self {
        let symbol = symbol.into();
        let phoneme_type = Self::determine_type(&symbol);
        
        Self {
            symbol,
            stress,
            position: None,
            standard,
            phoneme_type,
        }
    }
    
    /// Create a fully specified phoneme.
    pub fn new_full<S: Into<String>>(
        symbol: S, 
        stress: Option<StressLevel>,
        position: Option<PhonemePosition>,
        standard: PhonemeStandard,
        phoneme_type: PhonemeType
    ) -> Self {
        Self {
            symbol: symbol.into(),
            stress,
            position,
            standard,
            phoneme_type,
        }
    }
    
    /// Convert this phoneme to a string in the specified standard.
    pub fn to_string_in(&self, target_standard: PhonemeStandard) -> String {
        if self.standard == target_standard {
            return self.to_string();
        }
        
        match (self.standard, target_standard) {
            (PhonemeStandard::ARPABET, PhonemeStandard::IPA) => {
                let stressed = matches!(
                    self.stress,
                    Some(StressLevel::Primary) | Some(StressLevel::Secondary)
                );
                let base_ipa: String = match (self.symbol.as_str(), stressed) {
                    ("AH", false) => "ə".to_string(),
                    ("AH", true)  => "ʌ".to_string(),
                    ("ER", false) => "əɹ".to_string(),
                    ("ER", true)  => "ɜɹ".to_string(),
                    (sym, _) => ARPABET_TO_IPA
                        .get(sym)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| self.symbol.clone()),
                };
                
                // Add stress mark if needed
                if let Some(stress) = self.stress {
                    match stress {
                        StressLevel::Primary => format!("ˈ{}", base_ipa),
                        StressLevel::Secondary => format!("ˌ{}", base_ipa),
                        StressLevel::Unstressed => base_ipa,
                    }
                } else {
                    base_ipa
                }
            },
            (PhonemeStandard::IPA, PhonemeStandard::ARPABET) => {
                // Find the ARPABET equivalent
                let base_arpabet = IPA_TO_ARPABET.get(self.symbol.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| self.symbol.clone());
                
                // Add stress number if needed
                if let Some(stress) = self.stress {
                    match stress {
                        StressLevel::Primary => format!("{}1", base_arpabet),
                        StressLevel::Secondary => format!("{}2", base_arpabet),
                        StressLevel::Unstressed => format!("{}0", base_arpabet),
                    }
                } else {
                    base_arpabet
                }
            },
            // Add other conversions as needed
            _ => self.to_string(), // Default to original if conversion not implemented
        }
    }
    
    /// Returns true if this phoneme is a vowel
    pub fn is_vowel(&self) -> bool {
        self.phoneme_type == PhonemeType::Vowel
    }
    
    /// Returns true if this phoneme is a consonant
    pub fn is_consonant(&self) -> bool {
        self.phoneme_type == PhonemeType::Consonant
    }
    
    /// Returns true if this phoneme is a diphthong
    pub fn is_diphthong(&self) -> bool {
        self.phoneme_type == PhonemeType::Diphthong
    }
    
    /// Determine the type of phoneme from its symbol
    fn determine_type(symbol: &str) -> PhonemeType {
        // This is a simplified version - would be expanded with full phoneme sets
        match symbol {
            "AA" | "AE" | "AH" | "AO" | "EH" | "ER" | "IH" | "IY" | "UH" | "UW" | "AX" => {
                PhonemeType::Vowel
            },
            "AW" | "AY" | "EY" | "OW" | "OY" => {
                PhonemeType::Diphthong
            },
            "B" | "CH" | "D" | "DH" | "F" | "G" | "HH" | "JH" | "K" | "L" | "M" | "N" | 
            "NG" | "P" | "R" | "S" | "SH" | "T" | "TH" | "V" | "W" | "Y" | "Z" | "ZH" => {
                PhonemeType::Consonant
            },
            _ => {
                // If it's IPA, try to classify
                if symbol.chars().count() == 1 {
                    match symbol.chars().next().unwrap() {
                        'ɑ' | 'æ' | 'ʌ' | 'ɔ' | 'ə' | 'ɛ' | 'ɝ' | 'ɜ'
                        | 'ɪ' | 'i' | 'ʊ' | 'u' => PhonemeType::Vowel,
                        'A' | 'I' | 'O' | 'W' | 'Y' => PhonemeType::Diphthong,
                        'b' | 'd' | 'ð' | 'f' | 'ɡ' | 'g' | 'h' | 'k'
                        | 'l' | 'm' | 'n' | 'ŋ' | 'p' | 'ɹ' | 'r'
                        | 's' | 'ʃ' | 't' | 'θ' | 'v' | 'w' | 'j'
                        | 'z' | 'ʒ' | 'ʧ' | 'ʤ' => PhonemeType::Consonant,
                        _ => PhonemeType::Special,
                    }
                } else if symbol.chars().count() > 1 {
                    if symbol.contains(|c: char| {
                        "aeiouæɑəɛɝɜɪɔʊʌAIOWY".contains(c)
                    }) {
                        PhonemeType::Vowel
                    } else {
                        PhonemeType::Consonant
                    }
                } else {
                    PhonemeType::Special
                }
            }
        }
    }
    
    /// Create a compact binary representation of this phoneme
    pub fn to_compact(&self) -> Result<u8, G2PError> {
        // This is a simplified implementation - a real one would use a proper mapping table
        let phoneme_index = match self.standard {
            PhonemeStandard::ARPABET => {
                match self.symbol.as_str() {
                    "AA" => 0,
                    "AE" => 1,
                    "AH" => 2,
                    "AO" => 3,
                    "AW" => 4,
                    "AY" => 5,
                    "EH" => 6,
                    "ER" => 7,
                    "EY" => 8,
                    "IH" => 9,
                    "IY" => 10,
                    "OW" => 11,
                    "OY" => 12,
                    "UH" => 13,
                    "UW" => 14,
                    "B" => 15,
                    "CH" => 16,
                    "D" => 17,
                    "DH" => 18,
                    "F" => 19,
                    "G" => 20,
                    "HH" => 21,
                    "JH" => 22,
                    "K" => 23,
                    "L" => 24,
                    "M" => 25,
                    "N" => 26,
                    "NG" => 27,
                    "P" => 28,
                    "R" => 29,
                    "S" => 30,
                    "SH" => 31,
                    "T" => 32,
                    "TH" => 33,
                    "V" => 34,
                    "W" => 35,
                    "Y" => 36,
                    "Z" => 37,
                    "ZH" => 38,
                    _ => return Err(G2PError::PhonemeError(
                        format!("Unknown ARPABET phoneme: {}", self.symbol)
                    )),
                }
            },
            _ => return Err(G2PError::PhonemeError(
                format!("Compact encoding not implemented for {:?}", self.standard)
            )),
        };
        
        // Encode stress level in top 2 bits
        let stress_bits = match self.stress {
            Some(StressLevel::Primary) => 0b01_000000,
            Some(StressLevel::Secondary) => 0b10_000000,
            Some(StressLevel::Unstressed) => 0b11_000000,
            None => 0b00_000000,
        };
        
        // Combine phoneme index (6 bits) and stress (2 bits)
        Ok((phoneme_index & 0b00_111111) | stress_bits)
    }
    
    /// Create a phoneme from compact representation
    pub fn from_compact(compact: u8, standard: PhonemeStandard) -> Result<Self, G2PError> {
        if standard != PhonemeStandard::ARPABET {
            return Err(G2PError::PhonemeError(
                format!("Compact decoding not implemented for {:?}", standard)
            ));
        }
        
        // Extract stress from top 2 bits
        let stress = match (compact >> 6) & 0b11 {
            0b01 => Some(StressLevel::Primary),
            0b10 => Some(StressLevel::Secondary),
            0b11 => Some(StressLevel::Unstressed),
            _ => None,
        };
        
        // Extract phoneme index from bottom 6 bits
        let phoneme_index = compact & 0b00_111111;
        let symbol = match phoneme_index {
            0 => "AA",
            1 => "AE",
            2 => "AH",
            3 => "AO",
            4 => "AW",
            5 => "AY",
            6 => "EH",
            7 => "ER",
            8 => "EY",
            9 => "IH",
            10 => "IY",
            11 => "OW",
            12 => "OY",
            13 => "UH",
            14 => "UW",
            15 => "B",
            16 => "CH",
            17 => "D",
            18 => "DH",
            19 => "F",
            20 => "G",
            21 => "HH",
            22 => "JH",
            23 => "K",
            24 => "L",
            25 => "M",
            26 => "N",
            27 => "NG",
            28 => "P",
            29 => "R",
            30 => "S",
            31 => "SH",
            32 => "T",
            33 => "TH",
            34 => "V",
            35 => "W",
            36 => "Y",
            37 => "Z",
            38 => "ZH",
            _ => return Err(G2PError::PhonemeError(
                format!("Invalid phoneme index: {}", phoneme_index)
            )),
        };
        
        let phoneme_type = Self::determine_type(symbol);
        
        Ok(Self {
            symbol: symbol.to_string(),
            stress,
            position: None,
            standard: PhonemeStandard::ARPABET,
            phoneme_type,
        })
    }
}

impl fmt::Display for Phoneme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ARPABET style with stress number
        if self.standard == PhonemeStandard::ARPABET {
            match self.stress {
                Some(StressLevel::Primary) => write!(f, "{}1", self.symbol),
                Some(StressLevel::Secondary) => write!(f, "{}2", self.symbol),
                Some(StressLevel::Unstressed) => write!(f, "{}0", self.symbol),
                None => write!(f, "{}", self.symbol),
            }
        } else {
            // Otherwise just use the symbol
            write!(f, "{}", self.symbol)
        }
    }
}

/// A sequence of phonemes (pronunciation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeSequence {
    /// The phonemes in this sequence
    pub phonemes: Vec<Phoneme>,
    
    /// The standard used for this sequence
    pub standard: PhonemeStandard,
}

impl PhonemeSequence {
    /// Create a new phoneme sequence.
    pub fn new(phonemes: Vec<Phoneme>) -> Self {
        let standard = if !phonemes.is_empty() {
            phonemes[0].standard
        } else {
            PhonemeStandard::ARPABET
        };
        
        Self { phonemes, standard }
    }
    
    /// Create a new phoneme sequence with a specific standard.
    pub fn new_with_standard(phonemes: Vec<Phoneme>, standard: PhonemeStandard) -> Self {
        Self { phonemes, standard }
    }
    
    /// Get the number of phonemes in this sequence.
    pub fn len(&self) -> usize {
        self.phonemes.len()
    }
    
    /// Returns true if this sequence has no phonemes.
    pub fn is_empty(&self) -> bool {
        self.phonemes.is_empty()
    }
    
    /// Convert this phoneme sequence to a different standard.
    pub fn to_standard(&self, standard: PhonemeStandard) -> Self {
        if self.standard == standard {
            return self.clone();
        }
        
        let phonemes = self.phonemes.iter()
            .map(|p| {
                let symbol = p.to_string_in(standard);
                Phoneme::new_full(
                    symbol,
                    p.stress,
                    p.position,
                    standard,
                    p.phoneme_type,
                )
            })
            .collect();
        
        Self { phonemes, standard }
    }
    
    /// Get an iterator over the phonemes in this sequence.
    pub fn iter(&self) -> impl Iterator<Item = &Phoneme> {
        self.phonemes.iter()
    }
}

impl fmt::Display for PhonemeSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, 
            "{}",
            self.phonemes.iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}

impl From<Vec<Phoneme>> for PhonemeSequence {
    fn from(phonemes: Vec<Phoneme>) -> Self {
        Self::new(phonemes)
    }
}

impl From<PhonemeSequence> for Vec<Phoneme> {
    fn from(seq: PhonemeSequence) -> Self {
        seq.phonemes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phoneme_creation() {
        let p1 = Phoneme::new("AA", Some(StressLevel::Primary));
        assert_eq!(p1.symbol, "AA");
        assert_eq!(p1.stress, Some(StressLevel::Primary));
        assert_eq!(p1.standard, PhonemeStandard::ARPABET);
        assert_eq!(p1.phoneme_type, PhonemeType::Vowel);
        
        let p2 = Phoneme::new_with_standard("k", None, PhonemeStandard::IPA);
        assert_eq!(p2.symbol, "k");
        assert_eq!(p2.standard, PhonemeStandard::IPA);
        assert_eq!(p2.phoneme_type, PhonemeType::Consonant);
    }
    
    #[test]
    fn test_phoneme_display() {
        let p1 = Phoneme::new("AA", Some(StressLevel::Primary));
        assert_eq!(p1.to_string(), "AA1");
        
        let p2 = Phoneme::new("K", None);
        assert_eq!(p2.to_string(), "K");
    }
    
    #[test]
    fn test_phoneme_to_string_in() {
        let p1 = Phoneme::new("AA", Some(StressLevel::Primary));
        assert_eq!(p1.to_string_in(PhonemeStandard::IPA), "ˈɑ");
        
        let p2 = Phoneme::new_with_standard("ɑ", Some(StressLevel::Primary), PhonemeStandard::IPA);
        assert_eq!(p2.to_string_in(PhonemeStandard::ARPABET), "AA1");
    }
    
    #[test]
    fn test_phoneme_sequence() {
        let phonemes = vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", None),
        ];
        
        let seq = PhonemeSequence::new(phonemes);
        assert_eq!(seq.len(), 4);
        assert_eq!(seq.to_string(), "HH EH1 L OW");
    }
    
    #[test]
    fn test_compact_representation() {
        let p1 = Phoneme::new("AA", Some(StressLevel::Primary));
        let compact = p1.to_compact().unwrap();
        let p2 = Phoneme::from_compact(compact, PhonemeStandard::ARPABET).unwrap();
        assert_eq!(p1.symbol, p2.symbol);
        assert_eq!(p1.stress, p2.stress);
    }
}