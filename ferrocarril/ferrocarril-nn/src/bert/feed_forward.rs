//! Feed-forward network implementation for BERT
//! 
//! This module implements the position-wise feed-forward network used in
//! transformer architectures. It consists of two linear transformations with
//! a GELU activation in between.

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// Feed-forward network for BERT
pub struct FeedForward {
    /// Input/output dimension
    hidden_size: usize,
    /// Intermediate dimension
    intermediate_size: usize,
    /// First linear layer weights (hidden → intermediate)
    ffn_weight: Parameter,
    /// First linear layer bias
    ffn_bias: Parameter,
    /// Second linear layer weights (intermediate → hidden)
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
        let ffn_weight = Parameter::new(Tensor::from_data(
            vec![0.0; intermediate_size * hidden_size],
            vec![intermediate_size, hidden_size]
        ));
        let ffn_bias = Parameter::new(Tensor::from_data(
            vec![0.0; intermediate_size],
            vec![intermediate_size]
        ));
        
        let ffn_output_weight = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size * intermediate_size],
            vec![hidden_size, intermediate_size]
        ));
        let ffn_output_bias = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size],
            vec![hidden_size]
        ));
        
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
        let input_hidden_size = shape[2];
        
        // Validate input dimensions
        assert_eq!(input_hidden_size, self.hidden_size, 
                  "Input hidden size {} does not match configured hidden size {}", 
                  input_hidden_size, self.hidden_size);
        
        // First linear transformation: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size]
        let mut intermediate_data = vec![0.0; batch_size * seq_len * self.intermediate_size];
        
        // Compute first linear transformation (hidden → intermediate)
        for b in 0..batch_size {
            for s in 0..seq_len {
                for i in 0..self.intermediate_size {
                    let idx = (b * seq_len + s) * self.intermediate_size + i;
                    
                    // Start with bias
                    intermediate_data[idx] = self.ffn_bias.data()[&[i]];
                    
                    // Add weighted sum
                    for h in 0..self.hidden_size {
                        intermediate_data[idx] += 
                            self.ffn_weight.data()[&[i, h]] * hidden_states[&[b, s, h]];
                    }
                    
                    // Apply GELU activation
                    intermediate_data[idx] = Self::gelu(intermediate_data[idx]);
                }
            }
        }
        
        let intermediate_output = Tensor::from_data(
            intermediate_data,
            vec![batch_size, seq_len, self.intermediate_size]
        );
        
        // Second linear transformation: [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
        let mut output_data = vec![0.0; batch_size * seq_len * self.hidden_size];
        
        // Compute second linear transformation (intermediate → hidden)
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.hidden_size {
                    let idx = (b * seq_len + s) * self.hidden_size + h;
                    
                    // Start with bias
                    output_data[idx] = self.ffn_output_bias.data()[&[h]];
                    
                    // Add weighted sum
                    for i in 0..self.intermediate_size {
                        output_data[idx] += 
                            self.ffn_output_weight.data()[&[h, i]] * intermediate_output[&[b, s, i]];
                    }
                }
            }
        }
        
        // Return output tensor
        Tensor::from_data(
            output_data,
            vec![batch_size, seq_len, self.hidden_size]
        )
    }
}

impl LoadWeightsBinary for FeedForward {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        println!("Loading FeedForward weights for {}.{}", component_path, module_path);
        
        // Load first linear layer (ffn) weights
        let ffn_weight_path = format!(
            "{}.{}.ffn.weight",
            component_path,
            module_path
        );
        println!("Loading ffn weight from: {}", ffn_weight_path);
        let ffn_weight_tensor = loader.load_tensor(&ffn_weight_path)?;
        
        // Validate shape
        if ffn_weight_tensor.shape()[0] != self.intermediate_size || 
           ffn_weight_tensor.shape()[1] != self.hidden_size {
            return Err(FerroError::new(format!(
                "FFN weight has incorrect shape: expected [{}, {}], got {:?}",
                self.intermediate_size, self.hidden_size, ffn_weight_tensor.shape()
            )));
        }
        *self.ffn_weight.data_mut() = ffn_weight_tensor;
        
        // Load first linear layer (ffn) bias
        let ffn_bias_path = format!(
            "{}.{}.ffn.bias",
            component_path,
            module_path
        );
        println!("Loading ffn bias from: {}", ffn_bias_path);
        let ffn_bias_tensor = loader.load_tensor(&ffn_bias_path)?;
        
        // Validate shape
        if ffn_bias_tensor.shape()[0] != self.intermediate_size {
            return Err(FerroError::new(format!(
                "FFN bias has incorrect shape: expected [{}], got {:?}",
                self.intermediate_size, ffn_bias_tensor.shape()
            )));
        }
        *self.ffn_bias.data_mut() = ffn_bias_tensor;
        
        // Load second linear layer (ffn_output) weights
        let ffn_output_weight_path = format!(
            "{}.{}.ffn_output.weight",
            component_path,
            module_path
        );
        println!("Loading ffn output weight from: {}", ffn_output_weight_path);
        let ffn_output_weight_tensor = loader.load_tensor(&ffn_output_weight_path)?;
        
        // Validate shape
        if ffn_output_weight_tensor.shape()[0] != self.hidden_size || 
           ffn_output_weight_tensor.shape()[1] != self.intermediate_size {
            return Err(FerroError::new(format!(
                "FFN output weight has incorrect shape: expected [{}, {}], got {:?}",
                self.hidden_size, self.intermediate_size, ffn_output_weight_tensor.shape()
            )));
        }
        *self.ffn_output_weight.data_mut() = ffn_output_weight_tensor;
        
        // Load second linear layer (ffn_output) bias
        let ffn_output_bias_path = format!(
            "{}.{}.ffn_output.bias",
            component_path,
            module_path
        );
        println!("Loading ffn output bias from: {}", ffn_output_bias_path);
        let ffn_output_bias_tensor = loader.load_tensor(&ffn_output_bias_path)?;
        
        // Validate shape
        if ffn_output_bias_tensor.shape()[0] != self.hidden_size {
            return Err(FerroError::new(format!(
                "FFN output bias has incorrect shape: expected [{}], got {:?}",
                self.hidden_size, ffn_output_bias_tensor.shape()
            )));
        }
        *self.ffn_output_bias.data_mut() = ffn_output_bias_tensor;
        
        Ok(())
    }
}