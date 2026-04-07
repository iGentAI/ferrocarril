//! Ferrocarril TTS Engine library
//! 
//! A pure Rust implementation of the Kokoro TTS inference engine.
//!
//! This crate re-exports the core model, tensor, and weight-loading
//! types from the workspace sub-crates so downstream consumers only
//! need to depend on `ferrocarril` itself.

// Re-export core components
pub use ferrocarril_core::{Config, FerroError, Parameter};
pub use ferrocarril_core::tensor::Tensor;

#[cfg(feature = "weights")]
pub use ferrocarril_core::weights_binary::{
    BinaryWeightLoader,
    FilesystemBlobProvider,
    MapBlobProvider,
    WeightBlobProvider,
};

// Expose our modules
pub mod model;

// Re-export main model implementation
pub use self::model::FerroModel;