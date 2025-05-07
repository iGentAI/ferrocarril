//! BERT embeddings implementation for Ferrocarril TTS
//! 
//! This module provides embeddings for ALBERT, including token embeddings,
//! position embeddings, and token type embeddings.

use std::error::Error;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use super::layer_norm::LayerNorm;

/// BERT embeddings component
pub struct BertEmbeddings {
    /// Vocabulary size for token embeddings
    vocab_size: usize,
    /// Hidden size for embeddings
    hidden_size: usize,
    /// Maximum position embeddings
    max_position_embeddings: usize,
    /// Token embeddings lookup table
    word_embeddings: Parameter,
    /// Position embeddings lookup table
    position_embeddings: Parameter,
    /// Token type embeddings lookup table
    token_type_embeddings: Parameter,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Dropout probability (not used for inference)
    dropout_prob: f32,
}

impl BertEmbeddings {
    /// Create a new embeddings layer
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        max_position_embeddings: usize,
        dropout_prob: f32,
    ) -> Self {
        // Initialize embeddings randomly (will be overwritten by loaded weights)
        let word_embeddings = Parameter::new(Tensor::zeros(vec![vocab_size, hidden_size]));
        let position_embeddings = Parameter::new(Tensor::zeros(vec![max_position_embeddings, hidden_size]));
        let token_type_embeddings = Parameter::new(Tensor::zeros(vec![2, hidden_size]));
        let layer_norm = LayerNorm::new(hidden_size, 1e-12);

        Self {
            vocab_size,
            hidden_size,
            max_position_embeddings,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout_prob,
        }
    }
    
    /// Forward pass for embeddings
    /// 
    /// # Arguments
    /// * `input_ids` - Token IDs with shape [batch_size, seq_len]
    /// * `token_type_ids` - Optional token type IDs, defaults to zeros
    /// 
    /// # Returns
    /// Embedded representation of input with shape [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<i64>,
        token_type_ids: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        let shape = input_ids.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        if seq_len > self.max_position_embeddings {
            panic!("Input sequence length ({}) exceeds maximum position embeddings ({})", 
                seq_len, self.max_position_embeddings);
        }
        
        // Create output tensor
        let mut embeddings = Tensor::zeros(vec![batch_size, seq_len, self.hidden_size]);
        
        // Create default token type IDs if not provided
        let default_token_types = Tensor::<i64>::zeros(vec![batch_size, seq_len]);
        
        // Get reference to token type IDs (either provided or default)
        let token_type_ids_ref = match token_type_ids {
            Some(ids) => ids,
            None => &default_token_types,
        };
        
        // Loop through batch and sequence positions
        for b in 0..batch_size {
            for s in 0..seq_len {
                // Get token ID
                let token_id = input_ids[&[b, s]] as usize;
                if token_id >= self.vocab_size {
                    panic!("Token ID {} out of vocabulary range (0-{})", token_id, self.vocab_size-1);
                }
                
                // Get token type ID
                let token_type_id = token_type_ids_ref[&[b, s]] as usize;
                if token_type_id >= 2 {
                    panic!("Token type ID {} out of range (0-1)", token_type_id);
                }
                
                // Add token embeddings
                for h in 0..self.hidden_size {
                    embeddings[&[b, s, h]] += self.word_embeddings.data()[&[token_id, h]];
                }
                
                // Add position embeddings
                for h in 0..self.hidden_size {
                    embeddings[&[b, s, h]] += self.position_embeddings.data()[&[s, h]];
                }
                
                // Add token type embeddings
                for h in 0..self.hidden_size {
                    embeddings[&[b, s, h]] += self.token_type_embeddings.data()[&[token_type_id, h]];
                }
            }
        }
        
        // Apply layer normalization
        let embeddings = self.layer_norm.forward(&embeddings);
        
        // No dropout during inference
        embeddings
    }
}

impl LoadWeightsBinary for BertEmbeddings {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str, 
        module_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        // Load word embeddings
        let word_embeddings_path = format!(
            "{}.{}.embeddings.word_embeddings.weight",
            component_path,
            module_path
        );
        *self.word_embeddings.data_mut() = loader.load_tensor(&word_embeddings_path)?;
        
        // Load position embeddings
        let position_embeddings_path = format!(
            "{}.{}.embeddings.position_embeddings.weight", 
            component_path,
            module_path
        );
        *self.position_embeddings.data_mut() = loader.load_tensor(&position_embeddings_path)?;
        
        // Load token type embeddings
        let token_type_embeddings_path = format!(
            "{}.{}.embeddings.token_type_embeddings.weight",
            component_path,
            module_path
        );
        *self.token_type_embeddings.data_mut() = loader.load_tensor(&token_type_embeddings_path)?;
        
        // Load layer norm
        self.layer_norm.load_weights_binary(
            loader,
            component_path,
            &format!("{}.embeddings", module_path)
        )?;
        
        Ok(())
    }
}