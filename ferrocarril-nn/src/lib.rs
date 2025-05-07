//! Neural network components for Ferrocarril

pub mod linear;
pub mod activation;
pub mod conv;
pub mod lstm; 
pub mod adain;
pub mod text_encoder;
pub mod prosody;
pub mod vocoder;
pub mod bert;

use ferrocarril_core::tensor::Tensor;

/// Forward trait for neural network layers
pub trait Forward {
    type Output;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output;
}

/// Learnable parameters
#[derive(Debug, Clone)]
pub struct Parameter {
    data: Tensor<f32>,
}

impl Parameter {
    pub fn new(data: Tensor<f32>) -> Self {
        Self { data }
    }
    
    pub fn data(&self) -> &Tensor<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.data
    }
}