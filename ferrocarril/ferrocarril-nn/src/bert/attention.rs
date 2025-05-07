//! Multi-head attention implementation for BERT
//! 
//! This module implements the multi-head attention mechanism used in transformers,
//! with query, key, and value projections and attention output.

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
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
        let weight = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim * input_dim],
            vec![output_dim, input_dim]
        ));
        let bias = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim],
            vec![output_dim]
        ));
        
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
        let input_dim = shape[2];
        
        // Get actual dimensions from weights
        let weight_in_dim = if self.weight.data().shape().len() > 1 { self.weight.data().shape()[1] } else { 0 };
        let weight_out_dim = self.weight.data().shape()[0];
        let bias_dim = self.bias.data().shape()[0];
        
        // Use the minimum of expected and actual dimensions
        let effective_input_dim = input_dim.min(weight_in_dim);
        let effective_output_dim = weight_out_dim.min(bias_dim);
        
        println!("LinearProjection dimensions: input={}, weight={}x{}, bias={}, output={}",
            input_dim, weight_out_dim, weight_in_dim, bias_dim, effective_output_dim);
        
        // Create output data array
        let mut output_data = vec![0.0; batch_size * seq_len * effective_output_dim];
        
        // Perform matrix multiplication and add bias
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..effective_output_dim {
                    let idx = (b * seq_len + s) * effective_output_dim + o;
                    
                    // Initialize with bias
                    output_data[idx] = self.bias.data()[&[o]];
                    
                    // Add weighted sum of inputs for available dimensions
                    for i in 0..effective_input_dim {
                        output_data[idx] += self.weight.data()[&[o, i]] * x[&[b, s, i]];
                    }
                }
            }
        }
        
        // Create output tensor
        Tensor::from_data(
            output_data,
            vec![batch_size, seq_len, effective_output_dim]
        )
    }
    
    /// Load weights for linear projection
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
        name: &str,
    ) -> Result<(), FerroError> {
        println!("Loading LinearProjection {}...", name);
        
        // Load weights
        let weight_path = format!(
            "{}.{}.{}.weight",
            component_path,
            module_path,
            name
        );
        println!("Loading weight from: {}", weight_path);
        *self.weight.data_mut() = loader.load_tensor(&weight_path)?;
        println!("{} weight loaded with shape: {:?}", name, self.weight.data().shape());
        
        // Load bias
        let bias_path = format!(
            "{}.{}.{}.bias",
            component_path,
            module_path,
            name
        );
        println!("Loading bias from: {}", bias_path);
        *self.bias.data_mut() = loader.load_tensor(&bias_path)?;
        println!("{} bias loaded with shape: {:?}", name, self.bias.data().shape());
        
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
        let input_hidden_size = shape[2];
        
        println!("MultiHeadAttention forward: input shape={:?}", shape);
        
        // Project inputs to query, key, and value
        let query_layer = self.query.forward(hidden_states);
        let key_layer = self.key.forward(hidden_states);
        let value_layer = self.value.forward(hidden_states);
        
        println!("Query layer shape: {:?}", query_layer.shape());
        println!("Key layer shape: {:?}", key_layer.shape());
        println!("Value layer shape: {:?}", value_layer.shape());
        
        // Get actual projection dimensions
        let projection_dim = query_layer.shape()[2]; // Should be the same for query, key, value
        
        // Adjust the head size if needed
        let adjusted_num_heads = if self.num_attention_heads == 0 { 1 } else { self.num_attention_heads };
        let adjusted_head_size = projection_dim / adjusted_num_heads;
        
        println!("Adjusted attention heads: {}, head size: {}", adjusted_num_heads, adjusted_head_size);
        
        // Reshape query, key, and value to separate attention heads
        // Original shape: [batch_size, seq_len, hidden_size]
        // Target shape: [batch_size, num_heads, seq_len, head_size]
        
        // Reshape query
        let mut query_states_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * adjusted_head_size];
        
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
                for s in 0..seq_len {
                    for d in 0..adjusted_head_size {
                        let hidden_idx = h * adjusted_head_size + d;
                        if hidden_idx < projection_dim {
                            let idx = ((b * adjusted_num_heads + h) * seq_len + s) * adjusted_head_size + d;
                            query_states_data[idx] = query_layer[&[b, s, hidden_idx]];
                        }
                    }
                }
            }
        }
        
        let query_states = Tensor::from_data(
            query_states_data, 
            vec![batch_size, adjusted_num_heads, seq_len, adjusted_head_size]
        );
        
        // Reshape key
        let mut key_states_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * adjusted_head_size];
        
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
                for s in 0..seq_len {
                    for d in 0..adjusted_head_size {
                        let hidden_idx = h * adjusted_head_size + d;
                        if hidden_idx < projection_dim {
                            let idx = ((b * adjusted_num_heads + h) * seq_len + s) * adjusted_head_size + d;
                            key_states_data[idx] = key_layer[&[b, s, hidden_idx]];
                        }
                    }
                }
            }
        }
        
        let key_states = Tensor::from_data(
            key_states_data,
            vec![batch_size, adjusted_num_heads, seq_len, adjusted_head_size]
        );
        
        // Reshape value
        let mut value_states_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * adjusted_head_size];
        
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
                for s in 0..seq_len {
                    for d in 0..adjusted_head_size {
                        let hidden_idx = h * adjusted_head_size + d;
                        if hidden_idx < projection_dim {
                            let idx = ((b * adjusted_num_heads + h) * seq_len + s) * adjusted_head_size + d;
                            value_states_data[idx] = value_layer[&[b, s, hidden_idx]];
                        }
                    }
                }
            }
        }
        
        let value_states = Tensor::from_data(
            value_states_data,
            vec![batch_size, adjusted_num_heads, seq_len, adjusted_head_size]
        );
        
        // Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        let mut attention_scores_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * seq_len];
        
        // Compute dot product between query and key
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
                for q_pos in 0..seq_len {
                    for k_pos in 0..seq_len {
                        let mut score = 0.0;
                        for d in 0..adjusted_head_size {
                            score += query_states[&[b, h, q_pos, d]] * key_states[&[b, h, k_pos, d]];
                        }
                        
                        // Scale by square root of the attention head size
                        let idx = ((b * adjusted_num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
                        attention_scores_data[idx] = score / (adjusted_head_size as f32).sqrt();
                    }
                }
            }
        }
        
        let mut attention_scores = Tensor::from_data(
            attention_scores_data,
            vec![batch_size, adjusted_num_heads, seq_len, seq_len]
        );
        
        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            // Using binary mask where 1 = masked position, 0 = valid position
            // For masked positions, set score to negative infinity (we use a large negative value)
            for b in 0..batch_size {
                for h in 0..adjusted_num_heads {
                    for q_pos in 0..seq_len {
                        for k_pos in 0..seq_len {
                            // Make sure we don't access out of bounds in the mask
                            if b < mask.shape()[0] && q_pos < mask.shape()[1] && k_pos < mask.shape()[2] {
                                if mask[&[b, q_pos, k_pos]] > 0 {
                                    // Masked position
                                    attention_scores[&[b, h, q_pos, k_pos]] = -10000.0;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Apply softmax to get attention probabilities
        let mut attention_probs_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * seq_len];
        
        // Compute softmax row-wise
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
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
                        let idx = ((b * adjusted_num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
                        attention_probs_data[idx] = exp_val;
                        row_sum += exp_val;
                    }
                    
                    // Normalize by sum
                    for k_pos in 0..seq_len {
                        let idx = ((b * adjusted_num_heads + h) * seq_len + q_pos) * seq_len + k_pos;
                        attention_probs_data[idx] /= row_sum;
                    }
                }
            }
        }
        
        let attention_probs = Tensor::from_data(
            attention_probs_data,
            vec![batch_size, adjusted_num_heads, seq_len, seq_len]
        );
        
        // Apply attention probabilities to values
        let mut context_layer_data = vec![0.0; batch_size * adjusted_num_heads * seq_len * adjusted_head_size];
        
        // Weighted sum of values
        for b in 0..batch_size {
            for h in 0..adjusted_num_heads {
                for q_pos in 0..seq_len {
                    for d in 0..adjusted_head_size {
                        let mut weighted_sum = 0.0;
                        for k_pos in 0..seq_len {
                            weighted_sum += attention_probs[&[b, h, q_pos, k_pos]] * value_states[&[b, h, k_pos, d]];
                        }
                        let idx = ((b * adjusted_num_heads + h) * seq_len + q_pos) * adjusted_head_size + d;
                        context_layer_data[idx] = weighted_sum;
                    }
                }
            }
        }
        
        let context_layer = Tensor::from_data(
            context_layer_data,
            vec![batch_size, adjusted_num_heads, seq_len, adjusted_head_size]
        );
        
        // Reshape context layer back to [batch_size, seq_len, hidden_size]
        // Use the actual output dimension from the projection weights
        let output_hidden_size = projection_dim;
        let mut context_layer_flat_data = vec![0.0; batch_size * seq_len * output_hidden_size];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..adjusted_num_heads {
                    for d in 0..adjusted_head_size {
                        let hidden_idx = h * adjusted_head_size + d;
                        if hidden_idx < output_hidden_size {
                            let idx = (b * seq_len + s) * output_hidden_size + hidden_idx;
                            context_layer_flat_data[idx] = context_layer[&[b, h, s, d]];
                        }
                    }
                }
            }
        }
        
        let context_layer_flat = Tensor::from_data(
            context_layer_flat_data,
            vec![batch_size, seq_len, output_hidden_size]
        );
        
        // Apply output projection
        let attention_output = self.output.forward(&context_layer_flat);
        
        // Add residual connection and apply layer normalization
        // Make sure to match the dimensions of the attention output
        let output_shape = attention_output.shape();
        let out_dim = output_shape[2];
        let mut residual_output_data = vec![0.0; batch_size * seq_len * out_dim];
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..out_dim {
                    let idx = (b * seq_len + s) * out_dim + d;
                    // Only add from input if dimension is within bounds
                    if d < input_hidden_size {
                        residual_output_data[idx] = attention_output[&[b, s, d]] + hidden_states[&[b, s, d]];
                    } else {
                        residual_output_data[idx] = attention_output[&[b, s, d]];
                    }
                }
            }
        }
        
        let residual_output = Tensor::from_data(
            residual_output_data,
            vec![batch_size, seq_len, out_dim]
        );
        
        // Apply layer normalization
        println!("Applying layer normalization to shape: {:?}", residual_output.shape());
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
    ) -> Result<(), FerroError> {
        println!("Loading MultiHeadAttention weights for component={}, module={}", component_path, module_path);
        
        // Load query, key, value, output projections
        self.query.load_weights_binary(loader, component_path, module_path, "query")
            .map_err(|e| FerroError::new(format!("Failed to load query weights: {}", e)))?;
        
        self.key.load_weights_binary(loader, component_path, module_path, "key")
            .map_err(|e| FerroError::new(format!("Failed to load key weights: {}", e)))?;
        
        self.value.load_weights_binary(loader, component_path, module_path, "value")
            .map_err(|e| FerroError::new(format!("Failed to load value weights: {}", e)))?;
        
        self.output.load_weights_binary(loader, component_path, module_path, "dense")
            .map_err(|e| FerroError::new(format!("Failed to load output weights: {}", e)))?;
        
        // Load layer norm
        self.layer_norm.load_weights_binary(loader, component_path, &format!("{}", module_path))
            .map_err(|e| FerroError::new(format!("Failed to load layer norm weights: {}", e)))?;
        
        Ok(())
    }
}