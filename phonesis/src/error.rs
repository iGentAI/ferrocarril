//!
//! This module defines the error types used throughout the Phonesis library.
//! The main error type is `G2PError`, which encompasses all possible errors
//! that can occur during grapheme-to-phoneme conversion.

/// Error type for the Phonesis library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum G2PError {
    /// Error related to dictionary operations
    DictionaryError(String),
    
    /// Error related to rule application
    RuleError(String),
    
    /// Error related to phoneme operations
    PhonemeError(String),
    
    /// Error related to text normalization
    NormalizationError(String),
    
    /// Error related to language support
    UnsupportedLanguage(String),
    
    /// Invalid input provided
    InvalidInput(String),
    
    /// Unknown or unhandled word
    UnknownWord(String),
    
    /// IO error during dictionary loading
    IoError(String),
    
    /// General error
    Other(String),
}

impl std::error::Error for G2PError {}

impl std::fmt::Display for G2PError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G2PError::DictionaryError(msg) => write!(f, "Dictionary error: {}", msg),
            G2PError::RuleError(msg) => write!(f, "Rule error: {}", msg),
            G2PError::PhonemeError(msg) => write!(f, "Phoneme error: {}", msg),
            G2PError::NormalizationError(msg) => write!(f, "Normalization error: {}", msg),
            G2PError::UnsupportedLanguage(msg) => write!(f, "Unsupported language: {}", msg),
            G2PError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            G2PError::UnknownWord(msg) => write!(f, "Unknown word: {}", msg),
            G2PError::IoError(msg) => write!(f, "IO error: {}", msg),
            G2PError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl From<std::io::Error> for G2PError {
    fn from(error: std::io::Error) -> Self {
        G2PError::IoError(error.to_string())
    }
}

/// Result type for the Phonesis library.
pub type Result<T> = std::result::Result<T, G2PError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let error = G2PError::DictionaryError("failed to load".into());
        assert_eq!(error.to_string(), "Dictionary error: failed to load");
    }
    
    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: G2PError = io_error.into();
        assert!(matches!(error, G2PError::IoError(_)));
    }
}