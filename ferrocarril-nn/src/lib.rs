//! Neural network components for Ferrocarril

pub mod linear;
pub mod activation;
pub mod lstm;
pub mod adain;
pub mod conv;
pub mod text_encoder;
pub mod prosody;
pub mod conv_transpose;
pub mod vocoder;
pub mod bert;

use ferrocarril_core::tensor::Tensor;
// Re-export Parameter from ferrocarril-core
pub use ferrocarril_core::Parameter;

/// Forward trait for neural network layers
pub trait Forward {
    type Output;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output;
}