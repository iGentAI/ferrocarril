//!
//! This module provides memory-efficient implementations of the data structures
//! needed for the pronunciation dictionary, including a compact trie node
//! representation and efficient phoneme storage.

use std::mem;
use crate::phoneme::{Phoneme, PhonemeSequence, StressLevel, PhonemeStandard, PhonemeType};
use crate::error::{G2PError, Result};

/// A compact representation of a phoneme.
///
/// Stores a phoneme using a single byte:
/// - Bits 0-5: Phoneme index (0-63)
/// - Bits 6-7: Stress level (0=none, 1=primary, 2=secondary, 3=unstressed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactPhoneme(pub(crate) u8);

impl CompactPhoneme {
    /// Create a new compact phoneme.
    pub fn new(phoneme_index: u8, stress: Option<StressLevel>) -> Self {
        // Ensure phoneme index fits in 6 bits
        assert!(phoneme_index < 64, "Phoneme index must be less than 64");
        
        // Encode stress in top 2 bits
        let stress_bits = match stress {
            Some(StressLevel::Primary) => 1 << 6,    // 0b01_000000
            Some(StressLevel::Secondary) => 2 << 6,  // 0b10_000000
            Some(StressLevel::Unstressed) => 3 << 6, // 0b11_000000
            None => 0,
        };
        
        Self((phoneme_index & 0b00_111111) | stress_bits)
    }
    
    /// Get the phoneme index.
    pub fn phoneme_index(&self) -> u8 {
        self.0 & 0b00_111111
    }
    
    /// Get the stress level.
    pub fn stress(&self) -> Option<StressLevel> {
        let stress_bits = (self.0 & 0b11_000000) >> 6;
        match stress_bits {
            0 => None,
            x if x == 1 => Some(StressLevel::Primary),
            x if x == 2 => Some(StressLevel::Secondary),
            x if x == 3 => Some(StressLevel::Unstressed),
            _ => unreachable!("Invalid stress bits: {}", stress_bits),
        }
    }
}

/// A compact representation of a phoneme sequence.
///
/// Stores a sequence of phonemes in a memory-efficient format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompactPhonemeSequence {
    /// The phonemes in this sequence
    pub(crate) phonemes: Vec<CompactPhoneme>,
    
    /// The standard used for this sequence
    pub(crate) standard: PhonemeStandard,
}

/// A compact node in the trie data structure.
///
/// This representation is much more memory-efficient than using
/// standard Rust data structures, as it uses integer indices
/// instead of pointers and bit-packing for flags.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompactNode {
    /// Offset to first child in children array
    pub(crate) children_offset: u32,
    
    /// Number of children
    pub(crate) children_count: u16,
    
    /// Index of the value in values array, if any
    pub(crate) value_index: Option<u32>,
}

/// A child entry in the compact trie.
///
/// Each entry consists of a character and the index of the child node.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChildEntry {
    /// The character for this edge
    pub(crate) character: char,
    
    /// Index of the child node
    pub(crate) node_index: u32,
}

/// A compact trie data structure for memory-efficient dictionary storage.
///
/// This implementation uses integer indices instead of pointers,
/// which reduces memory usage significantly.
#[derive(Debug, Clone)]
pub(crate) struct CompactTrieStorage {
    /// All nodes in the trie
    pub(crate) nodes: Vec<CompactNode>,
    
    /// All child entries, indexed by offset
    pub(crate) children: Vec<ChildEntry>,
    
    /// All values in the trie
    pub(crate) values: Vec<CompactValue>,
    
    /// String table for word storage
    pub(crate) strings: Vec<u8>,
    
    /// String offsets in the string table
    pub(crate) string_offsets: Vec<u32>,
}

/// A compact value in the trie.
///
/// Stores a value (pronunciation) in a memory-efficient format.
#[derive(Debug, Clone)]
pub(crate) struct CompactValue {
    /// The phoneme sequence for this value
    pub(crate) phoneme_sequence: CompactPhonemeSequence,
}

impl CompactPhonemeSequence {
    /// Create a new compact phoneme sequence.
    pub fn new(phoneme_sequence: &PhonemeSequence) -> Result<Self> {
        let mut phonemes = Vec::with_capacity(phoneme_sequence.phonemes.len());
        
        // Only support ARPABET for now
        if phoneme_sequence.standard != PhonemeStandard::ARPABET {
            return Err(G2PError::PhonemeError(
                format!("Compact encoding only supports ARPABET phonemes, got {:?}", 
                        phoneme_sequence.standard)
            ));
        }
        
        for phoneme in &phoneme_sequence.phonemes {
            let compact = CompactPhoneme::from_phoneme(phoneme)?;
            phonemes.push(compact);
        }
        
        Ok(Self {
            phonemes,
            standard: phoneme_sequence.standard,
        })
    }
    
    /// Convert to a normal phoneme sequence.
    pub fn to_phoneme_sequence(&self) -> Result<PhonemeSequence> {
        let mut phonemes = Vec::with_capacity(self.phonemes.len());
        
        for compact in &self.phonemes {
            let phoneme = compact.to_phoneme(self.standard)?;
            phonemes.push(phoneme);
        }
        
        Ok(PhonemeSequence::new_with_standard(phonemes, self.standard))
    }
    
    /// Get the memory usage of this sequence in bytes.
    pub fn memory_usage(&self) -> usize {
        // Vec overhead + phonemes
        mem::size_of::<Vec<CompactPhoneme>>() + 
            self.phonemes.len() * mem::size_of::<CompactPhoneme>() +
            mem::size_of::<PhonemeStandard>()
    }
    
    /// Serialize to a byte array.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.phonemes.len() + 2);
        
        // Write phoneme count (u16)
        let count = self.phonemes.len() as u16;
        bytes.extend_from_slice(&count.to_le_bytes());
        
        // Write standard (u8)
        match self.standard {
            PhonemeStandard::IPA => bytes.push(0),
            PhonemeStandard::ARPABET => bytes.push(1),
            PhonemeStandard::SAMPA => bytes.push(2),
            PhonemeStandard::XSAMPA => bytes.push(3),
            PhonemeStandard::Custom(id) => {
                bytes.push(255); // special value for custom
                bytes.extend_from_slice(&id.to_le_bytes());
            },
        }
        
        // Write phonemes
        for phoneme in &self.phonemes {
            bytes.push(phoneme.0);
        }
        
        bytes
    }
    
    /// Deserialize from a byte array.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 3 {
            return Err(G2PError::DictionaryError(
                "Invalid compact phoneme sequence".to_string()
            ));
        }
        
        // Read phoneme count (u16)
        let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        
        // Read standard (u8)
        let standard_byte = bytes[2];
        let (standard, offset) = match standard_byte {
            0 => (PhonemeStandard::IPA, 3),
            1 => (PhonemeStandard::ARPABET, 3),
            2 => (PhonemeStandard::SAMPA, 3),
            3 => (PhonemeStandard::XSAMPA, 3),
            255 => {
                // Custom standard with ID
                if bytes.len() < 7 { // 3 + 4 bytes for u32
                    return Err(G2PError::DictionaryError(
                        "Invalid compact phoneme sequence - missing custom ID".to_string()
                    ));
                }
                let id = u32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
                (PhonemeStandard::Custom(id), 7)
            },
            _ => return Err(G2PError::PhonemeError(
                format!("Invalid phoneme standard: {}", bytes[2])
            )),
        };
        
        // Read phonemes
        if bytes.len() < offset + count {
            return Err(G2PError::DictionaryError(
                "Incomplete compact phoneme sequence".to_string()
            ));
        }
        
        let phonemes = bytes[offset..offset+count]
            .iter()
            .map(|&b| CompactPhoneme(b))
            .collect();
        
        Ok(Self {
            phonemes,
            standard,
        })
    }
}

impl CompactPhoneme {
    /// Convert a phoneme to compact representation.
    pub fn from_phoneme(phoneme: &Phoneme) -> Result<Self> {
        // Only support ARPABET for now
        if phoneme.standard != PhonemeStandard::ARPABET {
            return Err(G2PError::PhonemeError(
                format!("Compact encoding only supports ARPABET phonemes, got {:?}", 
                        phoneme.standard)
            ));
        }
        
        // Get phoneme index based on symbol
        let phoneme_index = match phoneme.symbol.as_str() {
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
                format!("Unknown ARPABET phoneme: {}", phoneme.symbol)
            )),
        };
        
        Ok(Self::new(phoneme_index, phoneme.stress))
    }
    
    /// Convert to a phoneme.
    pub fn to_phoneme(&self, standard: PhonemeStandard) -> Result<Phoneme> {
        // Only support ARPABET for now
        if standard != PhonemeStandard::ARPABET {
            return Err(G2PError::PhonemeError(
                format!("Compact encoding only supports ARPABET phonemes, got {:?}", 
                        standard)
            ));
        }
        
        // Get symbol based on phoneme index
        let symbol = match self.phoneme_index() {
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
                format!("Invalid phoneme index: {}", self.phoneme_index())
            )),
        };
        
        // Determine phoneme type
        let phoneme_type = match self.phoneme_index() {
            0..=14 => PhonemeType::Vowel,
            _ => PhonemeType::Consonant,
        };
        
        Ok(Phoneme::new_full(
            symbol.to_string(),
            self.stress(),
            None,
            standard,
            phoneme_type,
        ))
    }
}

impl Default for CompactNode {
    fn default() -> Self {
        Self {
            children_offset: 0,
            children_count: 0,
            value_index: None,
        }
    }
}

impl CompactTrieStorage {
    /// Create a new empty compact trie storage.
    pub fn new() -> Self {
        Self {
            nodes: vec![CompactNode::default()], // Root node
            children: Vec::new(),
            values: Vec::new(),
            strings: Vec::new(),
            string_offsets: Vec::new(),
        }
    }
    
    /// Add a string to the string table.
    pub fn add_string(&mut self, s: &str) -> u32 {
        let offset = self.strings.len() as u32;
        self.string_offsets.push(offset);
        
        // Add null-terminated string
        self.strings.extend_from_slice(s.as_bytes());
        self.strings.push(0);
        
        self.string_offsets.len() as u32 - 1
    }
    
    /// Get a string from the string table.
    pub fn get_string(&self, index: u32) -> Result<&str> {
        let offset = self.string_offsets.get(index as usize)
            .ok_or_else(|| G2PError::DictionaryError(
                format!("Invalid string index: {}", index)
            ))?;
        
        let offset = *offset as usize;
        
        // Find null terminator
        let mut end = offset;
        while end < self.strings.len() && self.strings[end] != 0 {
            end += 1;
        }
        
        if end >= self.strings.len() {
            return Err(G2PError::DictionaryError(
                "Missing null terminator in string table".to_string()
            ));
        }
        
        let bytes = &self.strings[offset..end];
        std::str::from_utf8(bytes)
            .map_err(|e| G2PError::DictionaryError(
                format!("Invalid UTF-8 in string table: {}", e)
            ))
    }
    
    /// Add a child entry.
    pub fn add_child(&mut self, character: char, node_index: u32) -> u32 {
        let offset = self.children.len() as u32;
        self.children.push(ChildEntry {
            character,
            node_index,
        });
        
        offset
    }
    
    /// Add a value.
    pub fn add_value(&mut self, phoneme_sequence: CompactPhonemeSequence) -> u32 {
        let index = self.values.len() as u32;
        self.values.push(CompactValue {
            phoneme_sequence,
        });
        
        index
    }
    
    /// Get the memory usage of this trie in bytes.
    pub fn memory_usage(&self) -> MemoryUsage {
        let nodes_size = self.nodes.len() * mem::size_of::<CompactNode>();
        let children_size = self.children.len() * mem::size_of::<ChildEntry>();
        let values_size = self.values.iter()
            .map(|v| mem::size_of::<CompactValue>() + v.phoneme_sequence.memory_usage())
            .sum();
        let strings_size = self.strings.len() + self.string_offsets.len() * mem::size_of::<u32>();
        
        MemoryUsage {
            nodes_size,
            values_size,
            string_table_size: strings_size,
            metadata_size: 0,
            total_size: nodes_size + children_size + values_size + strings_size,
        }
    }
}

impl CompactValue {
    /// Convert to a regular phoneme sequence.
    pub fn to_phoneme_sequence(&self) -> Result<PhonemeSequence> {
        self.phoneme_sequence.to_phoneme_sequence()
    }
    
    /// Create from a regular phoneme sequence.
    pub fn from_phoneme_sequence(phoneme_sequence: &PhonemeSequence) -> Result<Self> {
        Ok(Self {
            phoneme_sequence: CompactPhonemeSequence::new(phoneme_sequence)?,
        })
    }
}

/// Memory usage statistics for a dictionary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryUsage {
    /// Size of nodes in bytes
    pub nodes_size: usize,
    
    /// Size of values in bytes
    pub values_size: usize,
    
    /// Size of string table in bytes
    pub string_table_size: usize,
    
    /// Size of metadata in bytes
    pub metadata_size: usize,
    
    /// Total size in bytes
    pub total_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compact_phoneme() {
        let phoneme = Phoneme::new_full(
            "AA", 
            Some(StressLevel::Primary),
            None,
            PhonemeStandard::ARPABET,
            PhonemeType::Vowel
        );
        
        let compact = CompactPhoneme::from_phoneme(&phoneme).unwrap();
        assert_eq!(compact.phoneme_index(), 0);
        assert_eq!(compact.stress(), Some(StressLevel::Primary));
        
        let roundtrip = compact.to_phoneme(PhonemeStandard::ARPABET).unwrap();
        assert_eq!(roundtrip.symbol, "AA");
        assert_eq!(roundtrip.stress, Some(StressLevel::Primary));
    }
    
    #[test]
    fn test_compact_phoneme_sequence() {
        let phonemes = vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ];
        
        let seq = PhonemeSequence::new(phonemes);
        let compact = CompactPhonemeSequence::new(&seq).unwrap();
        
        assert_eq!(compact.phonemes.len(), 4);
        assert_eq!(compact.standard, PhonemeStandard::ARPABET);
        
        let roundtrip = compact.to_phoneme_sequence().unwrap();
        assert_eq!(roundtrip.phonemes.len(), 4);
        assert_eq!(roundtrip.phonemes[0].symbol, "HH");
        assert_eq!(roundtrip.phonemes[1].symbol, "EH");
        assert_eq!(roundtrip.phonemes[1].stress, Some(StressLevel::Primary));
    }
    
    #[test]
    fn test_serialization() {
        let phonemes = vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ];
        
        let seq = PhonemeSequence::new(phonemes);
        let compact = CompactPhonemeSequence::new(&seq).unwrap();
        
        let bytes = compact.to_bytes();
        let roundtrip = CompactPhonemeSequence::from_bytes(&bytes).unwrap();
        
        assert_eq!(compact.phonemes.len(), roundtrip.phonemes.len());
        assert_eq!(compact.standard, roundtrip.standard);
        
        for (a, b) in compact.phonemes.iter().zip(roundtrip.phonemes.iter()) {
            assert_eq!(a.0, b.0);
        }
    }
    
    #[test]
    fn test_compact_trie_storage() {
        let mut storage = CompactTrieStorage::new();
        
        // Add some strings
        let hello_idx = storage.add_string("hello");
        let world_idx = storage.add_string("world");
        
        // Add some nodes
        storage.nodes.push(CompactNode {
            children_offset: 0,
            children_count: 1,
            value_index: None,
        });
        
        storage.nodes.push(CompactNode {
            children_offset: 1,
            children_count: 1,
            value_index: Some(0),
        });
        
        // Add some children
        storage.add_child('h', 1);
        storage.add_child('e', 2);
        
        // Add some values
        let phonemes = vec![Phoneme::new("HH", None)];
        let seq = PhonemeSequence::new(phonemes);
        let compact_seq = CompactPhonemeSequence::new(&seq).unwrap();
        
        storage.add_value(compact_seq);
        
        // Check string retrieval
        assert_eq!(storage.get_string(hello_idx).unwrap(), "hello");
        assert_eq!(storage.get_string(world_idx).unwrap(), "world");
        
        // Check memory usage
        let usage = storage.memory_usage();
        assert!(usage.total_size > 0);
    }
}