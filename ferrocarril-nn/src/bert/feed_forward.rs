//! Feed-forward network implementation for BERT
//! 
//! This module implements the position-wise feed-forward network used in
//! transformer architectures. It consists of two linear transformations with
//! a GELU activation in between.

use std::error::Error;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// Feed-forward network for BERT
pub struct FeedForward {
    /// Input/output dimension
    hidden_size: usize,
    /// Intermediate dimension
    intermediate_size: usize,
    /// First linear layer weights
    ffn_weight: Parameter,
    /// First linear layer bias
    ffn_bias: Parameter,
    /// Second linear layer weights
    ffn_output_weight: Parameter,
    /// Second linear layer bias
    ffn_output_bias: Parameter,
    /// Dropout probability (not used for inference)
    dropout_prob: f32,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Self {
        // Initialize weights and biases (will be overwritten by loaded weights)
        let ffn_weight = Parameter::new(Tensor::zeros(vec![intermediate_size, hidden_size]));
        let ffn_bias = Parameter::new(Tensor::zeros(vec![intermediate_size]));
        
        let ffn_output_weight = Parameter::new(Tensor::zeros(vec![hidden_size, intermediate_size]));
        let ffn_output_bias = Parameter::new(Tensor::zeros(vec![hidden_size]));
        
        Self {
            hidden_size,
            intermediate_size,
            ffn_weight,
            ffn_bias,
            ffn_output_weight,
            ffn_output_bias,
            dropout_prob,
        }
    }
    
    /// GELU activation function
    /// 
    /// Gaussian Error Linear Unit as used in BERT
    /// Approximation of GELU from BERT implementation
    fn gelu(x: f32) -> f32 {
        // Approximate GELU: 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3)))
        let scale = (2.0 / std::f32::consts::PI).sqrt();
        let tanh_input = scale * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + tanh_input.tanh())
    }
    
    /// Forward pass for feed-forward network
    /// 
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape [batch_size, seq_len, hidden_size]
    /// 
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, hidden_size]
    pub fn forward(&self, hidden_states: &Tensor<f32>) -> Tensor<f32> {
        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // First linear transformation + GELU activation
        // [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        let mut intermediate_output = Tensor::new(vec![batch_size, seq_len, self.intermediate_size]);
        
        // Compute first linear transformation
        for b in 0..batch_size {
            for s in 0..seq_len {
                for i in 0..self.intermediate_size {
                    // Start with bias
                    intermediate_output[&[b, s, i]] = self.ffn_bias.data()[&[i]];
                    
                    // Add weighted sum
                    for h in 0..self.hidden_size {
                        intermediate_output[&[b, s, i]] += 
                            self.ffn_weight.data()[&[i, h]] * hidden_states[&[b, s, h]];
                    }
                    
                    // Apply GELU activation
                    intermediate_output[&[b, s, i]] = Self::gelu(intermediate_output[&[b, s, i]]);
                }
            }
        }
        
        // Second linear transformation
        // [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
        let mut layer_output = Tensor::new(vec![batch_size, seq_len, self.hidden_size]);
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.hidden_size {
                    // Start with bias
                    layer_output[&[b, s, h]] = self.ffn_output_bias.data()[&[h]];
                    
                    // Add weighted sum
                    for i in 0..self.intermediate_size {
                        layer_output[&[b, s, h]] += 
                            self.ffn_output_weight.data()[&[h, i]] * intermediate_output[&[b, s, i]];
                    }
                }
            }
        }
        
        // No dropout during inference
        layer_output
    }
}

impl LoadWeightsBinary for FeedForward {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Load first linear layer (ffn) weights
        let ffn_weight_path = format!(
            "{}.{}.ffn.weight",
            component_path,
            module_path
        );
        *self.ffn_weight.data_mut() = loader.load_tensor(&ffn_weight_path)?;
        
        // Load first linear layer (ffn) bias
        let ffn_bias_path = format!(
            "{}.{}.ffn.bias",
            component_path,
            module_path
        );
        *self.ffn_bias.data_mut() = loader.load_tensor(&ffn_bias_path)?;
        
        // Load second linear layer (ffn_output) weights
        let ffn_output_weight_path = format!(
            "{}.{}.ffn_output.weight",
            component_path,
            module_path
        );
        *self.ffn_output_weight.data_mut() = loader.load_tensor(&ffn_output_weight_path)?;
        
        // Load second linear layer (ffn_output) bias
        let ffn_output_bias_path = format!(
            "{}.{}.ffn_output.bias",
            component_path,
            module_path
        );
        *self.ffn_output_bias.data_mut() = loader.load_tensor(&ffn_output_bias_path)?;
        
        Ok(())
    }
}