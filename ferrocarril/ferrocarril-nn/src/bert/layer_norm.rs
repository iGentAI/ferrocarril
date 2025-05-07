//! Layer normalization implementation for ALBERT
//! 
//! This implements layer normalization with trainable parameters for
//! scaling (gamma) and shifting (beta).

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// Layer normalization with learnable parameters
pub struct LayerNorm {
    /// Dimension of the input vectors
    hidden_size: usize,
    /// Scaling factors (learnable parameter)
    gamma: Parameter,
    /// Shifting factors (learnable parameter)
    beta: Parameter,
    /// Epsilon for numerical stability
    eps: f32,
}

impl LayerNorm {
    /// Create a new layer normalization module
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        // Initialize gamma to ones and beta to zeros
        let gamma = Parameter::new(Tensor::from_data(
            vec![1.0; hidden_size],
            vec![hidden_size]
        ));
        
        let beta = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size],
            vec![hidden_size]
        ));
        
        Self {
            hidden_size,
            gamma,
            beta,
            eps,
        }
    }
    
    /// Apply layer normalization to input tensor
    /// 
    /// Input shape: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
    /// Output shape: Same as input
    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        
        // Layer norm applies along the last dimension (hidden_size)
        // Get the actual hidden dimension from the input tensor
        let actual_hidden_size = *shape.last().unwrap();
        let gamma_size = self.gamma.data().shape()[0];
        let beta_size = self.beta.data().shape()[0];
        
        // Get the minimum size to avoid index out of bounds
        let norm_size = actual_hidden_size.min(gamma_size).min(beta_size);
        
        println!("LayerNorm: input shape={:?}, expected hidden_size={}, actual={}, gamma={}, beta={}, using={}",
            shape, self.hidden_size, actual_hidden_size, gamma_size, beta_size, norm_size);
        
        // Handle both 2D and 3D inputs
        match shape.len() {
            2 => {
                // Input is [batch_size, hidden_size]
                let batch_size = shape[0];
                let mut output_data = vec![0.0; batch_size * actual_hidden_size];
                
                for b in 0..batch_size {
                    // Calculate mean
                    let mut mean = 0.0;
                    for h in 0..actual_hidden_size {
                        mean += x[&[b, h]];
                    }
                    mean /= actual_hidden_size as f32;
                    
                    // Calculate variance
                    let mut variance = 0.0;
                    for h in 0..actual_hidden_size {
                        variance += (x[&[b, h]] - mean).powi(2);
                    }
                    variance /= actual_hidden_size as f32;
                    
                    // Apply normalization, scaling, and shifting
                    for h in 0..actual_hidden_size {
                        let idx = b * actual_hidden_size + h;
                        // For values within gamma/beta range, apply normalization
                        if h < norm_size {
                            output_data[idx] = (x[&[b, h]] - mean) / (variance + self.eps).sqrt() 
                                * self.gamma.data()[&[h]] + self.beta.data()[&[h]];
                        } else {
                            // For values outside gamma/beta range, just keep the normalized value
                            output_data[idx] = (x[&[b, h]] - mean) / (variance + self.eps).sqrt();
                        }
                    }
                }
                
                Tensor::from_data(output_data, vec![batch_size, actual_hidden_size])
            },
            3 => {
                // Input is [batch_size, seq_len, hidden_size]
                let batch_size = shape[0];
                let seq_len = shape[1];
                let mut output_data = vec![0.0; batch_size * seq_len * actual_hidden_size];
                
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        // Calculate mean
                        let mut mean = 0.0;
                        for h in 0..actual_hidden_size {
                            mean += x[&[b, s, h]];
                        }
                        mean /= actual_hidden_size as f32;
                        
                        // Calculate variance
                        let mut variance = 0.0;
                        for h in 0..actual_hidden_size {
                            variance += (x[&[b, s, h]] - mean).powi(2);
                        }
                        variance /= actual_hidden_size as f32;
                        
                        // Apply normalization, scaling, and shifting
                        for h in 0..actual_hidden_size {
                            let idx = (b * seq_len + s) * actual_hidden_size + h;
                            // For values within gamma/beta range, apply normalization
                            if h < norm_size {
                                output_data[idx] = (x[&[b, s, h]] - mean) / (variance + self.eps).sqrt() 
                                    * self.gamma.data()[&[h]] + self.beta.data()[&[h]];
                            } else {
                                // For values outside gamma/beta range, just keep the normalized value
                                output_data[idx] = (x[&[b, s, h]] - mean) / (variance + self.eps).sqrt();
                            }
                        }
                    }
                }
                
                Tensor::from_data(output_data, vec![batch_size, seq_len, actual_hidden_size])
            },
            _ => {
                panic!("LayerNorm expects 2D or 3D input, got shape: {:?}", shape);
            }
        }
    }
}

impl LoadWeightsBinary for LayerNorm {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        println!("Loading LayerNorm weights for component={}, module={}", component_path, module_path);
        
        // Load gamma weights (weight)
        let gamma_path = format!(
            "{}.{}.LayerNorm.weight",
            component_path,
            module_path
        );
        if let Ok(gamma) = loader.load_tensor(&gamma_path) {
            println!("Loaded gamma from {} with shape {:?}", gamma_path, gamma.shape());
            *self.gamma.data_mut() = gamma;
        } else {
            // Try alternative path format
            let alt_gamma_path = format!(
                "{}.{}.weight",
                component_path,
                module_path
            );
            println!("Trying alternative gamma path: {}", alt_gamma_path);
            *self.gamma.data_mut() = loader.load_tensor(&alt_gamma_path)?;
            println!("Loaded gamma from {} with shape {:?}", alt_gamma_path, self.gamma.data().shape());
        }
        
        // Load beta weights (bias)
        let beta_path = format!(
            "{}.{}.LayerNorm.bias",
            component_path,
            module_path
        );
        if let Ok(beta) = loader.load_tensor(&beta_path) {
            println!("Loaded beta from {} with shape {:?}", beta_path, beta.shape());
            *self.beta.data_mut() = beta;
        } else {
            // Try alternative path format
            let alt_beta_path = format!(
                "{}.{}.bias",
                component_path,
                module_path
            );
            println!("Trying alternative beta path: {}", alt_beta_path);
            *self.beta.data_mut() = loader.load_tensor(&alt_beta_path)?;
            println!("Loaded beta from {} with shape {:?}", alt_beta_path, self.beta.data().shape());
        }
        
        Ok(())
    }
}