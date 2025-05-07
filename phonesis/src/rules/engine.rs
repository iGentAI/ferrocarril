//!
//! This module provides the core rule engine for applying production rules
//! to convert graphemes to phonemes.

use std::fmt;
use crate::error::G2PError;
use crate::phoneme::{Phoneme, PhonemeSequence, PhonemeStandard};
use super::context::RuleContext;
use super::patterns::{GraphemePattern, PatternMatcher};

/// A template for generating phoneme output.
#[derive(Debug, Clone)]
pub struct PhonemeTemplate {
    /// The phoneme symbols to generate
    pub phonemes: Vec<String>,
    
    /// The stress pattern to apply
    pub stress: Option<StressPattern>,
}

impl PhonemeTemplate {
    /// Create a new phoneme template.
    pub fn new(phonemes: Vec<String>) -> Self {
        Self {
            phonemes,
            stress: None,
        }
    }
    
    /// Create a phoneme template with specific stress pattern.
    pub fn with_stress(phonemes: Vec<String>, stress: StressPattern) -> Self {
        Self {
            phonemes,
            stress: Some(stress),
        }
    }
    
    /// Apply this template to generate phonemes.
    pub fn apply(
        &self,
        standard: PhonemeStandard,
        matched_groups: &[String],
    ) -> Result<PhonemeSequence, G2PError> {
        // Expand any placeholders in the template
        let mut phonemes = Vec::new();
        
        for symbol in &self.phonemes {
            // Check for group references (e.g., "$1", "$2")
            if symbol.starts_with('$') && symbol.len() > 1 {
                if let Ok(group) = symbol[1..].parse::<usize>() {
                    if group < matched_groups.len() {
                        // Replace with the matched group
                        phonemes.push(Phoneme::new(
                            matched_groups[group].clone(),
                            None, // Stress will be applied later
                        ));
                        continue;
                    }
                }
            }
            
            // Regular phoneme symbol
            phonemes.push(Phoneme::new(
                symbol.clone(),
                None, // Stress will be applied later
            ));
        }
        
        // Apply stress pattern if specified
        if let Some(stress) = &self.stress {
            stress.apply(&mut phonemes)?;
        }
        
        Ok(PhonemeSequence::new_with_standard(phonemes, standard))
    }
}

/// A pattern for applying stress to phonemes.
#[derive(Debug, Clone)]
pub enum StressPattern {
    /// Primary stress on the first vowel
    PrimaryFirst,
    
    /// Primary stress on the last vowel
    PrimaryLast,
    
    /// Primary stress on a specific phoneme index
    PrimaryAt(usize),
    
    /// Secondary stress on the first vowel
    SecondaryFirst,
    
    /// Secondary stress on the last vowel
    SecondaryLast,
    
    /// Secondary stress on a specific phoneme index
    SecondaryAt(usize),
    
    /// Custom stress pattern
    Custom(CustomStressPattern),
}

/// A wrapper for custom stress patterns
#[derive(Clone)]
pub struct CustomStressPattern {
    func: std::sync::Arc<dyn Fn(&mut [Phoneme]) -> Result<(), G2PError> + Send + Sync>
}

impl CustomStressPattern {
    /// Create a new custom stress pattern
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&mut [Phoneme]) -> Result<(), G2PError> + Send + Sync + 'static
    {
        Self { func: std::sync::Arc::new(f) }
    }
    
    /// Apply the stress pattern
    pub fn apply(&self, phonemes: &mut [Phoneme]) -> Result<(), G2PError> {
        (self.func)(phonemes)
    }
}

impl std::fmt::Debug for CustomStressPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CustomStressPattern(..)")
    }
}

impl StressPattern {
    /// Apply this stress pattern to a sequence of phonemes.
    pub fn apply(&self, phonemes: &mut [Phoneme]) -> Result<(), G2PError> {
        match self {
            StressPattern::PrimaryFirst => {
                // Find the first vowel and apply primary stress
                if let Some(pos) = phonemes.iter().position(|p| p.is_vowel()) {
                    phonemes[pos].stress = Some(crate::phoneme::StressLevel::Primary);
                }
            },
            StressPattern::PrimaryLast => {
                // Find the last vowel and apply primary stress
                if let Some(pos) = phonemes.iter().rposition(|p| p.is_vowel()) {
                    phonemes[pos].stress = Some(crate::phoneme::StressLevel::Primary);
                }
            },
            StressPattern::PrimaryAt(idx) => {
                // Apply primary stress to the specified phoneme
                if *idx < phonemes.len() {
                    phonemes[*idx].stress = Some(crate::phoneme::StressLevel::Primary);
                } else {
                    return Err(G2PError::RuleError(
                        format!("Stress index out of range: {} >= {}", idx, phonemes.len())
                    ));
                }
            },
            StressPattern::SecondaryFirst => {
                // Find the first vowel and apply secondary stress
                if let Some(pos) = phonemes.iter().position(|p| p.is_vowel()) {
                    phonemes[pos].stress = Some(crate::phoneme::StressLevel::Secondary);
                }
            },
            StressPattern::SecondaryLast => {
                // Find the last vowel and apply secondary stress
                if let Some(pos) = phonemes.iter().rposition(|p| p.is_vowel()) {
                    phonemes[pos].stress = Some(crate::phoneme::StressLevel::Secondary);
                }
            },
            StressPattern::SecondaryAt(idx) => {
                // Apply secondary stress to the specified phoneme
                if *idx < phonemes.len() {
                    phonemes[*idx].stress = Some(crate::phoneme::StressLevel::Secondary);
                } else {
                    return Err(G2PError::RuleError(
                        format!("Stress index out of range: {} >= {}", idx, phonemes.len())
                    ));
                }
            },
            StressPattern::Custom(custom) => {
                custom.apply(phonemes)?;
            },
        }
        
        Ok(())
    }
}

/// A rule for converting graphemes to phonemes.
#[derive(Debug, Clone)]
pub struct ProductionRule {
    /// A unique identifier for this rule
    pub id: String,
    
    /// The pattern to match
    pub pattern: GraphemePattern,
    
    /// The template to apply when the pattern matches
    pub output: PhonemeTemplate,
    
    /// The context in which this rule applies
    pub context: Option<RuleContext>,
    
    /// The priority of this rule (higher values = higher priority)
    pub priority: u32,
}

impl ProductionRule {
    /// Create a new production rule.
    pub fn new(
        id: impl Into<String>,
        pattern: GraphemePattern,
        output: PhonemeTemplate,
        priority: u32,
    ) -> Self {
        Self {
            id: id.into(),
            pattern,
            output,
            context: None,
            priority,
        }
    }
    
    /// Create a rule with context.
    pub fn with_context(
        id: impl Into<String>,
        pattern: GraphemePattern,
        output: PhonemeTemplate,
        context: RuleContext,
        priority: u32,
    ) -> Self {
        Self {
            id: id.into(),
            pattern,
            output,
            context: Some(context),
            priority,
        }
    }
    
    /// Check if this rule is applicable in the given context.
    pub fn is_applicable(&self, context: Option<&RuleContext>) -> bool {
        // If this rule has no context requirements, it's always applicable
        if self.context.is_none() {
            return true;
        }
        
        // If no context is provided but this rule requires one, it's not applicable
        if let Some(rule_context) = &self.context {
            match context {
                Some(provided_context) => {
                    // Check position
                    if let Some(position) = rule_context.position {
                        if let Some(provided_position) = provided_context.position {
                            if position != provided_position {
                                return false;
                            }
                        } else {
                            // Required position but none provided
                            return false;
                        }
                    }
                    
                    // Check if context is capitalized
                    if rule_context.is_capitalized != provided_context.is_capitalized {
                        return false;
                    }
                    
                    // Check preceding context
                    if let Some(left_context) = &rule_context.left_context {
                        if let Some(provided_left) = &provided_context.left_context {
                            if !provided_left.contains(left_context) {
                                return false;
                            }
                        } else {
                            // Required left context but none provided
                            return false;
                        }
                    }
                    
                    // Check following context
                    if let Some(right_context) = &rule_context.right_context {
                        if let Some(provided_right) = &provided_context.right_context {
                            if !provided_right.contains(right_context) {
                                return false;
                            }
                        } else {
                            // Required right context but none provided
                            return false;
                        }
                    }
                    
                    // Check properties
                    for (key, value) in &rule_context.properties {
                        if let Some(provided_value) = provided_context.properties.get(key) {
                            if value != provided_value {
                                return false;
                            }
                        } else {
                            // Required property but none provided
                            return false;
                        }
                    }
                    
                    // All context requirements satisfied
                    true
                },
                None => false, // Rule requires context but none provided
            }
        } else {
            true
        }
    }
    
    /// Apply this rule to a grapheme sequence.
    pub fn apply(
        &self,
        text: &str,
        standard: PhonemeStandard,
        context: Option<&RuleContext>,
    ) -> Option<PhonemeSequence> {
        // Check if this rule is applicable in the given context
        if !self.is_applicable(context) {
            return None;
        }
        
        // Check if the pattern matches
        let matched = self.pattern.matches(text)?;
        
        // Apply the template to generate phonemes
        match self.output.apply(standard, &matched) {
            Ok(phonemes) => Some(phonemes),
            Err(_) => None,
        }
    }
}

impl fmt::Display for ProductionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} -> {:?}",
            self.id,
            self.pattern.pattern,
            self.output.phonemes
        )
    }
}

/// Trait for rules that can be applied to input.
pub trait RuleApplication<T, U> {
    /// Apply this rule to input.
    fn apply(
        &self,
        input: &T,
        context: Option<&RuleContext>,
    ) -> Option<U>;
    
    /// Get the priority of this rule.
    fn priority(&self) -> u32;
    
    /// Get the identifier for this rule.
    fn id(&self) -> &str;
}

impl RuleApplication<String, PhonemeSequence> for ProductionRule {
    fn apply(
        &self,
        input: &String,
        context: Option<&RuleContext>,
    ) -> Option<PhonemeSequence> {
        self.apply(input, PhonemeStandard::ARPABET, context)
    }
    
    fn priority(&self) -> u32 {
        self.priority
    }
    
    fn id(&self) -> &str {
        &self.id
    }
}

/// The rule engine for applying production rules.
#[derive(Debug, Clone)]
pub struct RuleEngine {
    /// The rules in this engine
    rules: Vec<ProductionRule>,
    
    /// The pattern matcher
    patterns: PatternMatcher,
    
    /// The default phoneme standard
    standard: PhonemeStandard,
}

impl RuleEngine {
    /// Create a new rule engine.
    pub fn new(standard: PhonemeStandard) -> Self {
        Self {
            rules: Vec::new(),
            patterns: PatternMatcher::new(),
            standard,
        }
    }
    
    /// Add a rule to this engine.
    pub fn add_rule(&mut self, rule: ProductionRule) -> Result<(), G2PError> {
        // Add the rule's pattern to the pattern matcher
        self.patterns.add_pattern(rule.pattern.clone())?;
        
        // Add the rule to the engine
        self.rules.push(rule);
        
        // Sort rules by priority (descending)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        Ok(())
    }
    
    /// Apply rules to convert graphemes to phonemes.
    pub fn apply(&self, text: &str) -> Option<PhonemeSequence> {
        self.apply_with_context(text, None)
    }
    
    /// Apply rules with context.
    pub fn apply_with_context(
        &self,
        text: &str,
        context: Option<&RuleContext>,
    ) -> Option<PhonemeSequence> {
        // Try each rule in order of priority
        for rule in &self.rules {
            if let Some(result) = rule.apply(text, self.standard, context) {
                return Some(result);
            }
        }
        
        // No rules matched
        None
    }
    
    /// Get all applicable rules for a given text.
    pub fn find_applicable_rules(
        &self,
        text: &str,
        context: Option<&RuleContext>,
    ) -> Vec<&ProductionRule> {
        self.rules
            .iter()
            .filter(|rule| rule.is_applicable(context) && rule.pattern.matches(text).is_some())
            .collect()
    }
    
    /// Get the number of rules in this engine.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
    
    /// Get the default phoneme standard.
    pub fn standard(&self) -> PhonemeStandard {
        self.standard
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new(PhonemeStandard::ARPABET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoneme::StressLevel;
    
    #[test]
    fn test_phoneme_template() {
        let template = PhonemeTemplate::new(vec!["T".into(), "EH".into(), "S".into(), "T".into()]);
        let result = template.apply(PhonemeStandard::ARPABET, &["TEST".into()]).unwrap();
        
        assert_eq!(result.phonemes.len(), 4);
        assert_eq!(result.phonemes[0].symbol, "T");
        assert_eq!(result.phonemes[1].symbol, "EH");
        assert_eq!(result.phonemes[2].symbol, "S");
        assert_eq!(result.phonemes[3].symbol, "T");
    }
    
    #[test]
    fn test_phoneme_template_with_groups() {
        let template = PhonemeTemplate::new(vec!["$1".into()]);
        let result = template.apply(PhonemeStandard::ARPABET, &["".into(), "TEST".into()]).unwrap();
        
        assert_eq!(result.phonemes.len(), 1);
        assert_eq!(result.phonemes[0].symbol, "TEST");
    }
    
    #[test]
    fn test_stress_pattern() {
        // Create some phonemes
        let mut phonemes = vec![
            Phoneme::new("T", None),
            Phoneme::new("EH", None),
            Phoneme::new("S", None),
            Phoneme::new("T", None),
        ];
        
        // Apply stress to the first vowel
        StressPattern::PrimaryFirst.apply(&mut phonemes).unwrap();
        
        // Check that the stress was applied correctly
        assert_eq!(phonemes[0].stress, None); // T is not a vowel
        assert_eq!(phonemes[1].stress, Some(StressLevel::Primary)); // EH is a vowel
        assert_eq!(phonemes[2].stress, None); // S is not a vowel
        assert_eq!(phonemes[3].stress, None); // T is not a vowel
    }
    
    #[test]
    fn test_rule_application() {
        // Create a simple rule
        let rule = ProductionRule::new(
            "test_rule",
            GraphemePattern::new("test"),
            PhonemeTemplate::new(vec!["T".into(), "EH".into(), "S".into(), "T".into()]),
            10,
        );
        
        // Apply the rule
        let result = rule.apply("This is a test", PhonemeStandard::ARPABET, None);
        assert!(result.is_some());
        
        let phonemes = result.unwrap();
        assert_eq!(phonemes.phonemes.len(), 4);
        assert_eq!(phonemes.phonemes[0].symbol, "T");
        assert_eq!(phonemes.phonemes[1].symbol, "EH");
        assert_eq!(phonemes.phonemes[2].symbol, "S");
        assert_eq!(phonemes.phonemes[3].symbol, "T");
    }
    
    #[test]
    fn test_rule_context() {
        // Create a context-sensitive rule
        let context = RuleContext::new()
            .with_context(Some("is a"), Some("of"))
            .capitalized(false);
        
        let rule = ProductionRule::with_context(
            "context_rule",
            GraphemePattern::new("test"),
            PhonemeTemplate::new(vec!["T".into(), "EH".into(), "S".into(), "T".into()]),
            context.clone(),
            10,
        );
        
        // Test with matching context
        let result = rule.apply(
            "test",
            PhonemeStandard::ARPABET,
            Some(&context),
        );
        assert!(result.is_some());
        
        // Test with non-matching context
        let wrong_context = RuleContext::new()
            .with_context(Some("wrong"), Some("context"))
            .capitalized(true);
        
        let result = rule.apply(
            "test",
            PhonemeStandard::ARPABET,
            Some(&wrong_context),
        );
        assert!(result.is_none());
    }
    
    #[test]
    fn test_rule_engine() {
        let mut engine = RuleEngine::new(PhonemeStandard::ARPABET);
        
        // Add some rules
        engine.add_rule(ProductionRule::new(
            "rule1",
            GraphemePattern::new("hello"),
            PhonemeTemplate::new(vec!["HH".into(), "EH".into(), "L".into(), "OW".into()]),
            10,
        )).unwrap();
        
        engine.add_rule(ProductionRule::new(
            "rule2",
            GraphemePattern::new("world"),
            PhonemeTemplate::new(vec!["W".into(), "ER".into(), "L".into(), "D".into()]),
            5,
        )).unwrap();
        
        // Apply rules
        let result1 = engine.apply("hello");
        assert!(result1.is_some());
        let phonemes1 = result1.unwrap();
        assert_eq!(phonemes1.phonemes.len(), 4);
        assert_eq!(phonemes1.phonemes[0].symbol, "HH");
        
        let result2 = engine.apply("world");
        assert!(result2.is_some());
        let phonemes2 = result2.unwrap();
        assert_eq!(phonemes2.phonemes.len(), 4);
        assert_eq!(phonemes2.phonemes[0].symbol, "W");
        
        // Test with no matching rules
        let result3 = engine.apply("test");
        assert!(result3.is_none());
    }
}