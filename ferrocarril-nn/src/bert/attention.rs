//! Multi-head attention implementation for BERT
//! 
//! This module implements the multi-head attention mechanism used in transformers,
//! with query, key, and value projections and attention output.

use std::error::Error;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use super::layer_norm::LayerNorm;

/// Linear projection layer for attention
struct LinearProjection {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Weight matrix
    weight: Parameter,
    /// Bias vector
    bias: Parameter,
}

impl LinearProjection {
    /// Create a new linear projection with specified dimensions
    fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights and bias (will be overwritten by loaded weights)
        let weight = Parameter::new(Tensor::zeros(vec![output_dim, input_dim]));
        let bias = Parameter::new(Tensor::zeros(vec![output_dim]));
        
        Self {
            input_dim,
            output_dim,
            weight,
            bias,
        }
    }
    
    /// Forward pass for linear projection
    /// 
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_dim]
    /// 
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, output_dim]
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // Verify input shape
        if shape[2] != self.input_dim {
            panic!("Linear projection expected input dimension {}, got {}", 
                   self.input_dim, shape[2]);
        }
        
        // Create output tensor
        let mut output = Tensor::new(vec![batch_size, seq_len, self.output_dim]);
        
        // Perform matrix multiplication and add bias
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..self.output_dim {
                    // Initialize with bias
                    output[&[b, s, o]] = self.bias.data()[&[o]];
                    
                    // Add weighted sum of inputs
                    for i in 0..self.input_dim {
                        output[&[b, s, o]] += self.weight.data()[&[o, i]] * x[&[b, s, i]];
                    }
                }
            }
        }
        
        output
    }
    
    /// Load weights for linear projection
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
        name: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Load weights
        let weight_path = format!(
            "{}.{}.{}.weight",
            component_path,
            module_path,
            name
        );
        *self.weight.data_mut() = loader.load_tensor(&weight_path)?;
        
        // Load bias
        let bias_path = format!(
            "{}.{}.{}.bias",
            component_path,
            module_path,
            name
        );
        *self.bias.data_mut() = loader.load_tensor(&bias_path)?;
        
        Ok(())
    }
}

/// Multi-head attention module for BERT
pub struct MultiHeadAttention {
    /// Hidden size (input and output dimension)
    hidden_size: usize,
    /// Number of attention heads
    num_attention_heads: usize,
    /// Size of each attention head
    attention_head_size: usize,
    /// Query projection
    query: LinearProjection,
    /// Key projection
    key: LinearProjection,
    /// Value projection
    value: LinearProjection,
    /// Output projection
    output: LinearProjection,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Dropout probability (not used for inference)
    dropout_prob: f32,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention module
    pub fn new(
        hidden_size: usize, 
        num_attention_heads: usize,
        dropout_prob: f32,
    ) -> Self {
        // Verify hidden size is divisible by the number of heads
        if hidden_size % num_attention_heads != 0 {
            panic!("Hidden size ({}) must be divisible by the number of attention heads ({})",
                   hidden_size, num_attention_heads);
        }
        
        let attention_head_size = hidden_size / num_attention_heads;
        
        // Create projections
        let query = LinearProjection::new(hidden_size, hidden_size);
        let key = LinearProjection::new(hidden_size, hidden_size);
        let value = LinearProjection::new(hidden_size, hidden_size);
        let output = LinearProjection::new(hidden_size, hidden_size);
        
        // Create layer norm
        let layer_norm = LayerNorm::new(hidden_size, 1e-12);
        
        Self {
            hidden_size,
            num_attention_heads,
            attention_head_size,
            query,
            key,
            value,
            output,
            layer_norm,
            dropout_prob,
        }
    }
    

    
    /// Forward pass for multi-head attention
    /// 
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape [batch_size, seq_len, hidden_size]
    /// * `attention_mask` - Optional mask of shape [batch_size, seq_len, seq_len], 
    ///                      where 1 indicates masked positions to ignore
    /// 
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // Project inputs to query, key, and value
        let query_layer = self.query.forward(hidden_states);
        let key_layer = self.key.forward(hidden_states);
        let value_layer = self.value.forward(hidden_states);
        
        // Reshape query, key, and value to separate attention heads
        // Original shape: [batch_size, seq_len, hidden_size]
        // Target shape: [batch_size, num_heads, seq_len, head_size]
        
        // Reshape query
        let mut query_states = Tensor::new(vec![
            batch_size, 
            self.num_attention_heads, 
            seq_len, 
            self.attention_head_size
        ]);
        
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for s in 0..seq_len {
                    for d in 0..self.attention_head_size {
                        let hidden_idx = h * self.attention_head_size + d;
                        query_states[&[b, h, s, d]] = query_layer[&[b, s, hidden_idx]];
                    }
                }
            }
        }
        
        // Reshape key
        let mut key_states = Tensor::new(vec![
            batch_size, 
            self.num_attention_heads, 
            seq_len, 
            self.attention_head_size
        ]);
        
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for s in 0..seq_len {
                    for d in 0..self.attention_head_size {
                        let hidden_idx = h * self.attention_head_size + d;
                        key_states[&[b, h, s, d]] = key_layer[&[b, s, hidden_idx]];
                    }
                }
            }
        }
        
        // Reshape value
        let mut value_states = Tensor::new(vec![
            batch_size, 
            self.num_attention_heads, 
            seq_len, 
            self.attention_head_size
        ]);
        
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for s in 0..seq_len {
                    for d in 0..self.attention_head_size {
                        let hidden_idx = h * self.attention_head_size + d;
                        value_states[&[b, h, s, d]] = value_layer[&[b, s, hidden_idx]];
                    }
                }
            }
        }
        
        // Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        let mut attention_scores = Tensor::new(vec![
            batch_size,
            self.num_attention_heads,
            seq_len,
            seq_len
        ]);
        
        // Compute dot product between query and key
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for q_pos in 0..seq_len {
                    for k_pos in 0..seq_len {
                        let mut score = 0.0;
                        for d in 0..self.attention_head_size {
                            score += query_states[&[b, h, q_pos, d]] * key_states[&[b, h, k_pos, d]];
                        }
                        
                        // Scale by square root of the attention head size
                        attention_scores[&[b, h, q_pos, k_pos]] = score / (self.attention_head_size as f32).sqrt();
                    }
                }
            }
        }
        
        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // Using binary mask where 1 = masked position, 0 = valid position
            // For masked positions, set score to negative infinity (we use a large negative value)
            for b in 0..batch_size {
                for h in 0..self.num_attention_heads {
                    for q_pos in 0..seq_len {
                        for k_pos in 0..seq_len {
                            if mask[&[b, q_pos, k_pos]] > 0 {
                                // Masked position
                                attention_scores[&[b, h, q_pos, k_pos]] = -10000.0;
                            }
                        }
                    }
                }
            }
        }
        
        // Apply softmax to get attention probabilities
        let mut attention_probs = Tensor::new(vec![
            batch_size,
            self.num_attention_heads,
            seq_len,
            seq_len
        ]);
        
        // Compute softmax row-wise
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for q_pos in 0..seq_len {
                    // Find row max for numerical stability
                    let mut row_max = -std::f32::INFINITY;
                    for k_pos in 0..seq_len {
                        row_max = row_max.max(attention_scores[&[b, h, q_pos, k_pos]]);
                    }
                    
                    // Compute exp(x - max) for each element
                    let mut row_sum = 0.0;
                    for k_pos in 0..seq_len {
                        let shifted = attention_scores[&[b, h, q_pos, k_pos]] - row_max;
                        let exp_val = shifted.exp();
                        attention_probs[&[b, h, q_pos, k_pos]] = exp_val;
                        row_sum += exp_val;
                    }
                    
                    // Normalize by sum
                    for k_pos in 0..seq_len {
                        attention_probs[&[b, h, q_pos, k_pos]] /= row_sum;
                    }
                }
            }
        }
        
        // Apply attention probabilities to values
        let mut context_layer = Tensor::new(vec![
            batch_size,
            self.num_attention_heads,
            seq_len,
            self.attention_head_size
        ]);
        
        // Weighted sum of values
        for b in 0..batch_size {
            for h in 0..self.num_attention_heads {
                for q_pos in 0..seq_len {
                    for d in 0..self.attention_head_size {
                        let mut weighted_sum = 0.0;
                        for k_pos in 0..seq_len {
                            weighted_sum += attention_probs[&[b, h, q_pos, k_pos]] * value_states[&[b, h, k_pos, d]];
                        }
                        context_layer[&[b, h, q_pos, d]] = weighted_sum;
                    }
                }
            }
        }
        
        // Reshape context layer back to [batch_size, seq_len, hidden_size]
        let mut context_layer_flat = Tensor::new(vec![batch_size, seq_len, self.hidden_size]);
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_attention_heads {
                    for d in 0..self.attention_head_size {
                        let hidden_idx = h * self.attention_head_size + d;
                        context_layer_flat[&[b, s, hidden_idx]] = context_layer[&[b, h, s, d]];
                    }
                }
            }
        }
        
        // Apply output projection
        let attention_output = self.output.forward(&context_layer_flat);
        
        // Add residual connection and apply layer normalization
        let mut residual_output = Tensor::new(vec![batch_size, seq_len, self.hidden_size]);
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.hidden_size {
                    residual_output[&[b, s, d]] = attention_output[&[b, s, d]] + hidden_states[&[b, s, d]];
                }
            }
        }
        
        // Apply layer normalization
        let normalized_output = self.layer_norm.forward(&residual_output);
        
        normalized_output
    }
}

impl LoadWeightsBinary for MultiHeadAttention {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Load query, key, value, output projections
        self.query.load_weights_binary(loader, component_path, module_path, "query")?;
        self.key.load_weights_binary(loader, component_path, module_path, "key")?;
        self.value.load_weights_binary(loader, component_path, module_path, "value")?;
        self.output.load_weights_binary(loader, component_path, module_path, "dense")?;
        
        // Load layer norm
        self.layer_norm.load_weights_binary(loader, component_path, &format!("{}", module_path))
    }
}