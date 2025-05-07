//! English-specific pronunciation rules
//!
//! This module defines a set of pronunciation rules specific to English
//! for converting unknown words to phonemes when dictionary lookup fails.

use crate::{
    phoneme::{Phoneme, StressLevel},
    rules::{RuleEngine, ProductionRule, GraphemePattern},
};

/// Initialize the English rule engine with pronunciation rules.
pub fn initialize_english_rules(rule_engine: &mut RuleEngine) {
    // Add consonant rules
    add_consonant_rules(rule_engine);
    
    // Add vowel rules
    add_vowel_rules(rule_engine);
    
    // Add digraph rules
    add_digraph_rules(rule_engine);
    
    // Add syllable stress rules
    add_stress_rules(rule_engine);
}

/// Add basic consonant pronunciation rules.
fn add_consonant_rules(rule_engine: &mut RuleEngine) {
    // Basic consonant mappings
    let consonant_mappings = [
        ("b", vec!["B"]),
        ("c", vec!["K"]),
        ("d", vec!["D"]),
        ("f", vec!["F"]),
        ("g", vec!["G"]),
        ("h", vec!["HH"]),
        ("j", vec!["JH"]),
        ("k", vec!["K"]),
        ("l", vec!["L"]),
        ("m", vec!["M"]),
        ("n", vec!["N"]),
        ("p", vec!["P"]),
        ("q", vec!["K"]),
        ("r", vec!["R"]),
        ("s", vec!["S"]),
        ("t", vec!["T"]),
        ("v", vec!["V"]),
        ("w", vec!["W"]),
        ("x", vec!["K", "S"]),
        ("y", vec!["Y"]),
        ("z", vec!["Z"]),
    ];
    
    // Add rules with appropriate priorities
    // Higher priority = applied first
    for (idx, (grapheme, phonemes)) in consonant_mappings.iter().enumerate() {
        let rule = create_basic_rule(
            format!("consonant_{}", grapheme),
            grapheme,
            phonemes.to_vec(),
            100 - idx as u32, // Decreasing priority based on order
        );
        
        // Add rule to engine
        match rule_engine.add_rule(rule) {
            Ok(_) => {},
            Err(e) => {
                // Log error but continue
                eprintln!("Failed to add rule for consonant '{}': {}", grapheme, e);
            }
        }
    }
    
    // Context-specific consonant rules
    // Examples:
    // - 'c' before 'e', 'i', 'y' is pronounced as 'S'
    // - 'g' before 'e', 'i', 'y' can be soft 'JH'
    
    // Add more complex rules here
}

/// Add basic vowel pronunciation rules.
fn add_vowel_rules(rule_engine: &mut RuleEngine) {
    // Basic vowel mappings
    let vowel_mappings = [
        ("a", vec!["AE"]),
        ("e", vec!["EH"]),
        ("i", vec!["IH"]),
        ("o", vec!["AA"]),
        ("u", vec!["AH"]),
    ];
    
    // Add rules with appropriate priorities
    for (idx, (grapheme, phonemes)) in vowel_mappings.iter().enumerate() {
        let rule = create_basic_rule(
            format!("vowel_{}", grapheme),
            grapheme,
            phonemes.to_vec(),
            200 - idx as u32, // Higher priority than consonants
        );
        
        // Add rule to engine
        match rule_engine.add_rule(rule) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Failed to add rule for vowel '{}': {}", grapheme, e);
            }
        }
    }
    
    // Long vowel patterns (vowel-consonant-e)
    // Examples:
    // - 'a' in 'cake' is pronounced 'EY'
    // - 'i' in 'time' is pronounced 'AY'
    
    // Add more complex vowel rules here
}

/// Add digraph and trigraph pronunciation rules.
fn add_digraph_rules(rule_engine: &mut RuleEngine) {
    // Common English digraphs
    let digraph_mappings = [
        ("th", vec!["TH"]),
        ("ch", vec!["CH"]),
        ("sh", vec!["SH"]),
        ("ph", vec!["F"]),
        ("wh", vec!["W"]),
        ("ng", vec!["NG"]),
        ("qu", vec!["K", "W"]),
        ("ck", vec!["K"]),
        ("gh", vec!["G"]),
        ("kn", vec!["N"]),
        ("wr", vec!["R"]),
        ("mb", vec!["M"]),
    ];
    
    // Add rules with appropriate priorities
    for (idx, (grapheme, phonemes)) in digraph_mappings.iter().enumerate() {
        let rule = create_basic_rule(
            format!("digraph_{}", grapheme),
            grapheme,
            phonemes.to_vec(),
            300 - idx as u32, // Higher priority than single letters
        );
        
        // Add rule to engine
        match rule_engine.add_rule(rule) {
            Ok(_) => {},
            Err(e) => {
                eprintln!("Failed to add rule for digraph '{}': {}", grapheme, e);
            }
        }
    }
    
    // Add more complex digraph/trigraph rules here
}

/// Add syllable stress rules.
fn add_stress_rules(rule_engine: &mut RuleEngine) {
    // Simple stress rules for English words
    // In a full implementation, this would use more sophisticated analysis
    // For now, we'll just use a few simple heuristics:
    
    // 1. Default stress on first syllable for two-syllable words
    // 2. For longer words, handle common suffixes
    
    // Add more complex stress rules here
}

/// Create a basic rule for mapping a grapheme to phonemes.
fn create_basic_rule(
    id: String,
    grapheme: &str,
    phonemes: Vec<&str>,
    priority: u32,
) -> ProductionRule {
    // Create a pattern to match the grapheme
    let pattern = GraphemePattern::new(grapheme);
    
    // Create the output template
    let mut template_phonemes = Vec::new();
    for &p in &phonemes {
        template_phonemes.push(p.to_string());
    }
    
    // Create a production rule
    ProductionRule::new(
        id,
        pattern,
        crate::rules::PhonemeTemplate::new(template_phonemes),
        priority,
    )
}

/// Create English-specific stress rules for words.
pub fn apply_english_stress(phonemes: &mut Vec<Phoneme>) -> Result<(), crate::error::G2PError> {
    // Count vowels (potential syllable nuclei)
    let vowels: Vec<usize> = phonemes.iter()
        .enumerate()
        .filter(|(_, p)| p.is_vowel())
        .map(|(i, _)| i)
        .collect();
    
    // Add stress based on the number of syllables
    match vowels.len() {
        0 => {} // No vowels to stress
        1 => {
            // Monosyllabic words - primary stress on the only vowel
            if let Some(&idx) = vowels.first() {
                phonemes[idx].stress = Some(StressLevel::Primary);
            }
        }
        2 => {
            // Bisyllabic words - primary stress typically on the first syllable
            if let Some(&idx) = vowels.first() {
                phonemes[idx].stress = Some(StressLevel::Primary);
            }
            // Unless it's a suffix that pulls stress to the second syllable
            // (Simplified heuristic for this implementation)
        }
        _ => {
            // For longer words, a simplified approach:
            // Put primary stress on the antepenultimate syllable (3rd from end)
            let stress_idx = if vowels.len() >= 3 {
                vowels[vowels.len() - 3]
            } else {
                vowels[0]
            };
            
            phonemes[stress_idx].stress = Some(StressLevel::Primary);
            
            // Add secondary stress to the first syllable if it's not the primary
            if vowels[0] != stress_idx && vowels.len() > 3 {
                phonemes[vowels[0]].stress = Some(StressLevel::Secondary);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhonemeStandard;
    
    #[test]
    fn test_rule_creation() {
        // Create a rule engine
        let mut rule_engine = RuleEngine::new(PhonemeStandard::ARPABET);
        
        // Add a basic consonant rule
        add_consonant_rules(&mut rule_engine);
        
        // Check that rules were added
        assert!(rule_engine.rule_count() > 0);
    }
    
    #[test]
    fn test_stress_application() {
        // Create a set of phonemes for a word
        let mut phonemes = vec![
            Phoneme::new("K", None),
            Phoneme::new("AE", None),  // vowel[0] at index 1
            Phoneme::new("T", None),
            Phoneme::new("AH", None),  // vowel[1] at index 3  
            Phoneme::new("G", None),
            Phoneme::new("AO", None),  // vowel[2] at index 5
            Phoneme::new("R", None),
            Phoneme::new("IY", None),  // vowel[3] at index 7
        ];
        
        // Apply English stress rules
        apply_english_stress(&mut phonemes).unwrap();
        
        // Check that stress was applied correctly
        // The word resembles "category" with 4 syllables
        // Primary stress should be on antepenultimate (3rd from end) = vowel[1] = index 3
        // Secondary stress should be on first syllable = vowel[0] = index 1  
        assert_eq!(phonemes[3].stress, Some(StressLevel::Primary));
        assert_eq!(phonemes[1].stress, Some(StressLevel::Secondary));
    }
}