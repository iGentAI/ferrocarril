//!
//! This module provides a rule-based system for converting graphemes (text) to phonemes
//! (pronunciation) when dictionary lookup fails. It uses pattern matching, context-sensitive
//! rules, and priority-based application to generate plausible pronunciations.

mod context;
mod engine;
mod patterns;

pub use context::{RuleContext, WordPosition};
pub use engine::{RuleEngine, ProductionRule, PhonemeTemplate, StressPattern};
pub use patterns::{GraphemePattern, PatternConstraint};

// Re-export traits for rule engine extensions
pub use engine::RuleApplication;

/// A generic trait for rule-based processing.
pub trait ApplyRules<T, U> {
    /// Apply rules to input and produce output.
    fn apply(&self, input: &T) -> Option<U>;
    
    /// Apply rules with context.
    fn apply_with_context(&self, input: &T, context: &RuleContext) -> Option<U>;
}