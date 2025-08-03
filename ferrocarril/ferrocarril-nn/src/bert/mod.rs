//! BERT (ALBERT) implementation for Ferrocarril TTS
//! 
//! This module provides a pure Rust implementation of the ALBERT architecture 
//! used in the Kokoro TTS model. The implementation is based on the ALBERT paper
//! with parameter sharing optimizations for efficiency.

mod embeddings;
mod attention;
mod feed_forward;
mod layer_norm;
mod transformer;

pub use embeddings::BertEmbeddings;
pub use attention::MultiHeadAttention;
pub use feed_forward::FeedForward;
pub use layer_norm::LayerNorm;
pub use transformer::AlbertLayer;
pub use transformer::AlbertLayerGroup;
pub use transformer::CustomBert;

// Re-export the main component for easier access
pub use transformer::CustomBert as Bert;