//! English-specific pronunciation rules
//!
//! This module defines a set of pronunciation rules specific to English
//! for converting unknown words to phonemes when dictionary lookup fails.

use crate::rules::{RuleEngine, ProductionRule, GraphemePattern};

/// Initialize the English rule engine with pronunciation rules.
pub fn initialize_english_rules(rule_engine: &mut RuleEngine) {
    // Add consonant rules
    add_consonant_rules(rule_engine);
    
    // Add vowel rules
    add_vowel_rules(rule_engine);
    
    // Add digraph rules
    add_digraph_rules(rule_engine);
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
        if let Err(e) = rule_engine.add_rule(rule) {
            eprintln!("Failed to add rule for consonant '{}': {}", grapheme, e);
        }
    }
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
        if let Err(e) = rule_engine.add_rule(rule) {
            eprintln!("Failed to add rule for vowel '{}': {}", grapheme, e);
        }
    }
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
        if let Err(e) = rule_engine.add_rule(rule) {
            eprintln!("Failed to add rule for digraph '{}': {}", grapheme, e);
        }
    }
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
}