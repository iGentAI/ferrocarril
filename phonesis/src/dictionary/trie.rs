//!
//! This module provides a memory-efficient trie data structure for
//! storing and retrieving word pronunciations.

use std::collections::HashMap;
use crate::phoneme::PhonemeSequence;

/// A node in the trie.
#[derive(Debug, Default, Clone)]
struct Node {
    /// Children of this node, indexed by character
    children: HashMap<char, usize>,
    
    /// Value (pronunciation) at this node, if any
    value: Option<usize>,
}

/// A compact trie implementation for efficient word lookup.
///
/// This trie uses integer indices instead of pointers, which allows
/// for better memory usage and potential serialization.
#[derive(Debug, Clone)]
pub struct CompactTrie {
    /// The nodes in the trie
    nodes: Vec<Node>,
    
    /// The values (pronunciations) stored in the trie
    values: Vec<PhonemeSequence>,
}

impl Default for CompactTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactTrie {
    /// Create a new empty trie.
    pub fn new() -> Self {
        // Always create a root node
        Self {
            nodes: vec![Node::default()],
            values: Vec::new(),
        }
    }
    
    /// Insert a word and its pronunciation into the trie.
    pub fn insert(&mut self, word: &str, pronunciation: PhonemeSequence) {
        let mut node_idx = 0; // Start at the root
        
        // Navigate (and create) the path for this word
        for c in word.to_lowercase().chars() {
            let next_idx = match self.nodes[node_idx].children.get(&c) {
                Some(&idx) => idx,
                None => {
                    // Create a new node
                    let new_idx = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[node_idx].children.insert(c, new_idx);
                    new_idx
                }
            };
            
            node_idx = next_idx;
        }
        
        // Store the pronunciation
        let _value_idx = match self.nodes[node_idx].value {
            Some(idx) => {
                // Replace the existing pronunciation
                self.values[idx] = pronunciation;
                idx
            }
            None => {
                // Add a new pronunciation
                let idx = self.values.len();
                self.values.push(pronunciation);
                self.nodes[node_idx].value = Some(idx);
                idx
            }
        };
    }
    
    /// Get the pronunciation for a word.
    pub fn get(&self, word: &str) -> Option<&PhonemeSequence> {
        let mut node_idx = 0; // Start at the root
        
        // Navigate the path for this word
        for c in word.to_lowercase().chars() {
            node_idx = *self.nodes[node_idx].children.get(&c)?;
        }
        
        // Return the pronunciation, if any
        self.nodes[node_idx].value.map(|idx| &self.values[idx])
    }
    
    /// Check if the trie contains a word.
    pub fn contains(&self, word: &str) -> bool {
        self.get(word).is_some()
    }
    
    /// Get the number of words in the trie.
    pub fn len(&self) -> usize {
        self.values.len()
    }
    
    /// Check if the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    
    /// Get an iterator over all words and pronunciations in the trie.
    pub fn iter(&self) -> impl Iterator<Item = (String, &PhonemeSequence)> {
        // This is a simple implementation that builds all words upfront.
        // A more efficient implementation would use a custom iterator.
        let mut result = Vec::new();
        let mut prefix = String::new();
        self.collect_words(0, &mut prefix, &mut result);
        result.into_iter()
    }
    
    /// Helper to recursively collect all words in the trie.
    fn collect_words<'a>(&'a self, node_idx: usize, prefix: &mut String, result: &mut Vec<(String, &'a PhonemeSequence)>) {
        // Add this node's value, if any
        if let Some(value_idx) = self.nodes[node_idx].value {
            result.push((prefix.clone(), &self.values[value_idx]));
        }
        
        // Recurse to children
        for (&c, &child_idx) in &self.nodes[node_idx].children {
            prefix.push(c);
            self.collect_words(child_idx, prefix, result);
            prefix.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoneme::{Phoneme, StressLevel};
    
    #[test]
    fn test_empty_trie() {
        let trie = CompactTrie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
        assert!(!trie.contains("hello"));
    }
    
    #[test]
    fn test_insert_and_get() {
        let mut trie = CompactTrie::new();
        
        // Create phoneme sequences
        let hello = PhonemeSequence::new(vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ]);
        
        let world = PhonemeSequence::new(vec![
            Phoneme::new("W", None),
            Phoneme::new("ER", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("D", None),
        ]);
        
        // Insert into trie
        trie.insert("hello", hello);
        trie.insert("world", world);
        
        // Check trie state
        assert!(!trie.is_empty());
        assert_eq!(trie.len(), 2);
        assert!(trie.contains("hello"));
        assert!(trie.contains("world"));
        assert!(!trie.contains("test"));
        
        // Lookup and check values
        let hello_result = trie.get("hello").unwrap();
        assert_eq!(hello_result.phonemes[0].symbol, "HH");
        assert_eq!(hello_result.phonemes[1].symbol, "EH");
        assert_eq!(hello_result.phonemes[1].stress, Some(StressLevel::Primary));
        
        let world_result = trie.get("world").unwrap();
        assert_eq!(world_result.phonemes[0].symbol, "W");
        assert_eq!(world_result.phonemes[1].symbol, "ER");
        assert_eq!(world_result.phonemes[1].stress, Some(StressLevel::Primary));
    }
    
    #[test]
    fn test_case_insensitivity() {
        let mut trie = CompactTrie::new();
        
        // Create a phoneme sequence
        let hello = PhonemeSequence::new(vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ]);
        
        // Insert with lowercase
        trie.insert("hello", hello);
        
        // Check with different cases
        assert!(trie.contains("hello"));
        assert!(trie.contains("Hello"));
        assert!(trie.contains("HELLO"));
        assert!(trie.contains("HeLLo"));
    }
    
    #[test]
    fn test_update_existing() {
        let mut trie = CompactTrie::new();
        
        // Create initial phoneme sequence
        let hello1 = PhonemeSequence::new(vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ]);
        
        // Create updated phoneme sequence
        let hello2 = PhonemeSequence::new(vec![
            Phoneme::new("HH", None),
            Phoneme::new("AH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", None),
        ]);
        
        // Insert then update
        trie.insert("hello", hello1);
        assert_eq!(trie.len(), 1);
        
        trie.insert("hello", hello2);
        assert_eq!(trie.len(), 1); // Should still have only one entry
        
        // Check updated value
        let result = trie.get("hello").unwrap();
        assert_eq!(result.phonemes[1].symbol, "AH");
    }
    
    #[test]
    fn test_iteration() {
        let mut trie = CompactTrie::new();
        
        // Create phoneme sequences
        let hello = PhonemeSequence::new(vec![
            Phoneme::new("HH", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("OW", Some(StressLevel::Unstressed)),
        ]);
        
        let world = PhonemeSequence::new(vec![
            Phoneme::new("W", None),
            Phoneme::new("ER", Some(StressLevel::Primary)),
            Phoneme::new("L", None),
            Phoneme::new("D", None),
        ]);
        
        let test = PhonemeSequence::new(vec![
            Phoneme::new("T", None),
            Phoneme::new("EH", Some(StressLevel::Primary)),
            Phoneme::new("S", None),
            Phoneme::new("T", None),
        ]);
        
        // Insert into trie
        trie.insert("hello", hello);
        trie.insert("world", world);
        trie.insert("test", test);
        
        // Collect all entries
        let entries: Vec<_> = trie.iter().collect();
        
        // Check results
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().any(|(w, _)| w == "hello"));
        assert!(entries.iter().any(|(w, _)| w == "world"));
        assert!(entries.iter().any(|(w, _)| w == "test"));
    }
    
    #[test]
    fn test_prefix_sharing() {
        let mut trie = CompactTrie::new();
        
        // Create simple phoneme sequences
        let test1 = PhonemeSequence::new(vec![Phoneme::new("T1", None)]);
        let test2 = PhonemeSequence::new(vec![Phoneme::new("T2", None)]);
        let testing = PhonemeSequence::new(vec![Phoneme::new("T3", None)]);
        
        // Insert words with shared prefix
        trie.insert("test", test1);
        trie.insert("tests", test2);
        trie.insert("testing", testing);
        
        // Check trie state
        assert_eq!(trie.len(), 3);
        
        // Ensure each word is correctly stored
        assert!(trie.contains("test"));
        assert!(trie.contains("tests"));
        assert!(trie.contains("testing"));
        
        // Check the retrieved values
        assert_eq!(trie.get("test").unwrap().phonemes[0].symbol, "T1");
        assert_eq!(trie.get("tests").unwrap().phonemes[0].symbol, "T2");
        assert_eq!(trie.get("testing").unwrap().phonemes[0].symbol, "T3");
    }
}