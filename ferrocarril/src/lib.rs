//! Ferrocarril TTS Engine library
//! 
//! A pure Rust implementation of the Kokoro TTS inference engine

// Re-export core components
pub use ferrocarril_core::Config;
pub use ferrocarril_core::tensor::Tensor;

// Expose our modules
pub mod model;

// Re-export main model implementation
pub use self::model::FerroModel;