//! BERT embeddings implementation for Ferrocarril TTS
//! 
//! This module provides embeddings for ALBERT, including token embeddings,
//! position embeddings, and token type embeddings.

use std::error::Error;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
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
        let word_embeddings = Parameter::new(Tensor::from_data(
            vec![0.0; vocab_size * hidden_size],
            vec![vocab_size, hidden_size]
        ));
        let position_embeddings = Parameter::new(Tensor::from_data(
            vec![0.0; max_position_embeddings * hidden_size],
            vec![max_position_embeddings, hidden_size]
        ));
        let token_type_embeddings = Parameter::new(Tensor::from_data(
            vec![0.0; 2 * hidden_size],
            vec![2, hidden_size]
        ));
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
        let mut embeddings = Tensor::from_data(
            vec![0.0; batch_size * seq_len * self.hidden_size],
            vec![batch_size, seq_len, self.hidden_size]
        );
        
        // Create default token type IDs if not provided
        let default_token_types = Tensor::<i64>::from_data(
            vec![0; batch_size * seq_len], 
            vec![batch_size, seq_len]
        );
        
        // Get reference to token type IDs (either provided or default)
        let token_type_ids_ref = match token_type_ids {
            Some(ids) => ids,
            None => &default_token_types,
        };
        
        // Get actual dimensions from the embedded tensors
        let embedding_hidden_size = self.word_embeddings.data().shape()[1];
        let position_hidden_size = self.position_embeddings.data().shape()[1];
        let token_type_hidden_size = self.token_type_embeddings.data().shape()[1];

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
                
                // Add token embeddings - ensure we don't exceed bounds
                // Note: In ALBERT, embeddings might have different dimensions than the model's hidden_size
                for h in 0..embedding_hidden_size.min(self.hidden_size) {
                    embeddings[&[b, s, h]] += self.word_embeddings.data()[&[token_id, h]];
                }
                
                // Add position embeddings - ensure we don't exceed bounds
                for h in 0..position_hidden_size.min(self.hidden_size) {
                    if s < self.position_embeddings.data().shape()[0] {
                        embeddings[&[b, s, h]] += self.position_embeddings.data()[&[s, h]];
                    }
                }
                
                // Add token type embeddings - ensure we don't exceed bounds
                for h in 0..token_type_hidden_size.min(self.hidden_size) {
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
    ) -> Result<(), FerroError> {
        // Load word embeddings
        let word_embeddings_path = format!(
            "{}.{}.embeddings.word_embeddings.weight",
            component_path,
            module_path
        );
        let tensor = loader.load_tensor(&word_embeddings_path)?;
        *self.word_embeddings.data_mut() = tensor;

        // Load position embeddings
        let position_embeddings_path = format!(
            "{}.{}.embeddings.position_embeddings.weight",
            component_path,
            module_path
        );
        let tensor = loader.load_tensor(&position_embeddings_path)?;
        *self.position_embeddings.data_mut() = tensor;

        // Load token type embeddings
        let token_type_embeddings_path = format!(
            "{}.{}.embeddings.token_type_embeddings.weight",
            component_path,
            module_path
        );
        let tensor = loader.load_tensor(&token_type_embeddings_path)?;
        *self.token_type_embeddings.data_mut() = tensor;

        // Load layer norm
        self.layer_norm.load_weights_binary(
            loader,
            component_path,
            &format!("{}.embeddings", module_path)
        )?;

        Ok(())
    }
}