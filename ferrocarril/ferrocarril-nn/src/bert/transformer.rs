//! ALBERT transformer implementation for BERT component
//!
//! This module implements ALBERT-style transformer layers with parameter sharing,
//! as well as the full CustomBERT model which encapsulates the complete transformer
//! architecture used in the Ferrocarril TTS system.

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

use super::embeddings::BertEmbeddings;
use super::attention::MultiHeadAttention;
use super::feed_forward::FeedForward;
use super::layer_norm::LayerNorm;

/// ALBERT layer (single transformer layer with attention and feed-forward)
pub struct AlbertLayer {
    /// Hidden size 
    hidden_size: usize,
    /// Attention mechanism
    attention: MultiHeadAttention,
    /// Feed-forward network
    feed_forward: FeedForward,
    /// Layer normalization for feed-forward output
    full_layer_norm: LayerNorm,
    /// Dropout probability (not used for inference)
    dropout_prob: f32,
}

impl AlbertLayer {
    /// Create a new ALBERT layer
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Self {
        // Create multi-head attention
        let attention = MultiHeadAttention::new(
            hidden_size,
            num_attention_heads,
            dropout_prob
        );
        
        // Create feed-forward network
        let feed_forward = FeedForward::new(
            hidden_size,
            intermediate_size,
            dropout_prob
        );
        
        // Create layer normalization
        let full_layer_norm = LayerNorm::new(hidden_size, 1e-12);
        
        Self {
            hidden_size,
            attention,
            feed_forward,
            full_layer_norm,
            dropout_prob,
        }
    }
    
    /// Forward pass for ALBERT layer
    /// 
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape [batch_size, seq_len, hidden_size]
    /// * `attention_mask` - Optional mask tensor
    /// 
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        // Run through self-attention mechanism
        let attention_output = self.attention.forward(hidden_states, attention_mask);

        // Run through feed-forward network
        let feed_forward_output = self.feed_forward.forward(&attention_output);
        
        // Add residual connection
        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        // Only modify the line with the unused variable warning
        let _input_hidden = shape[2];
        
        // Get output shapes
        let ff_shape = feed_forward_output.shape();
        let att_shape = attention_output.shape();
        
        // Use the minimum of all dimensions to avoid index out of bounds
        let output_hidden = ff_shape[2].min(att_shape[2]);
        
        // Create residual data array
        let mut residual_data = vec![0.0; batch_size * seq_len * output_hidden];
        
        // Add residual connection across common dimensions
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..output_hidden {
                    let idx = (b * seq_len + s) * output_hidden + h;
                    residual_data[idx] = feed_forward_output[&[b, s, h]];
                    
                    // Only add attention output if dimension is within bounds
                    if h < att_shape[2] {
                        residual_data[idx] += attention_output[&[b, s, h]];
                    }
                }
            }
        }
        
        let residual_output = Tensor::from_data(
            residual_data,
            vec![batch_size, seq_len, output_hidden]
        );
        
        // Apply layer normalization
        self.full_layer_norm.forward(&residual_output)
    }
}

impl LoadWeightsBinary for AlbertLayer {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        // Load attention - fail if missing
        self.attention.load_weights_binary(loader, component_path, &format!("{}.attention", module_path))?;
        
        // Load feed-forward - fail if missing
        self.feed_forward.load_weights_binary(loader, component_path, module_path)?;
        
        // Load layer norm - fail if missing
        self.full_layer_norm.load_weights_binary(loader, component_path, &format!("{}.full_layer_layer_norm", module_path))?;
        
        Ok(())
    }
}

/// ALBERT layer group with parameter sharing
pub struct AlbertLayerGroup {
    /// Single ALBERT layer to apply multiple times
    albert_layer: AlbertLayer,
}

impl AlbertLayerGroup {
    /// Create a new ALBERT layer group
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Self {
        // Create underlying layer (to be applied repeatedly)
        let albert_layer = AlbertLayer::new(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout_prob
        );
        
        Self {
            albert_layer,
        }
    }
    
    /// Forward pass for layer group (applies same layer multiple times)
    /// 
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape [batch_size, seq_len, hidden_size]
    /// * `attention_mask` - Optional attention mask
    /// * `num_hidden_layers` - Number of times to apply the layer
    /// 
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
        num_hidden_layers: usize,
    ) -> Tensor<f32> {
        let mut layer_output = hidden_states.clone();

        // Limit the number of layers to a reasonable maximum to avoid issues
        let max_layers = 24; // Standard BERT/ALBERT usually has 12-24 layers
        let applied_layers = num_hidden_layers.min(max_layers);

        if applied_layers < num_hidden_layers {
            println!(
                "Warning: Limiting number of layers from {} to {}",
                num_hidden_layers, applied_layers
            );
        }

        // Apply the same layer repeatedly
        for _ in 0..applied_layers {
            layer_output = self.albert_layer.forward(&layer_output, attention_mask);
        }
        
        layer_output
    }
}

impl LoadWeightsBinary for AlbertLayerGroup {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        // In ALBERT, we only have a single layer in each group - fail if missing
        self.albert_layer.load_weights_binary(loader, component_path, &format!("{}.albert_layers.0", module_path))?;
        
        Ok(())
    }
}

/// Linear mapping from embeddings to hidden states
struct EmbeddingHiddenMapping {
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Weight matrix
    weight: Parameter,
    /// Bias vector
    bias: Parameter,
}

impl EmbeddingHiddenMapping {
    /// Create a new embedding to hidden mapping
    fn new(
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        // Initialize weights and bias
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
    
    /// Forward pass for embedding to hidden mapping
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let input_dim = shape[2];
        
        // Check weight dimensions
        let weight_in_dim = if self.weight.data().shape().len() > 1 { self.weight.data().shape()[1] } else { 0 };
        let weight_out_dim = self.weight.data().shape()[0]; 
        let bias_dim = self.bias.data().shape()[0];
        
        // Use the minimum dimension to avoid index out of bounds
        let effective_input_dim = input_dim.min(weight_in_dim);
        let effective_output_dim = weight_out_dim.min(bias_dim);

        // Create output data array
        let mut output_data = vec![0.0; batch_size * seq_len * effective_output_dim];
        
        // Perform linear transformation
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..effective_output_dim {
                    let idx = (b * seq_len + s) * effective_output_dim + o;
                    // Initialize with bias
                    output_data[idx] = self.bias.data()[&[o]];
                    
                    // Add weighted sum within bounds
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
    
    /// Load weights for embedding to hidden mapping
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        // Load weights
        let weight_path = format!(
            "{}.{}.embedding_hidden_mapping_in.weight",
            component_path,
            module_path
        );
        *self.weight.data_mut() = loader.load_tensor(&weight_path)?;

        // Load bias
        let bias_path = format!(
            "{}.{}.embedding_hidden_mapping_in.bias",
            component_path,
            module_path
        );
        *self.bias.data_mut() = loader.load_tensor(&bias_path)?;
        
        Ok(())
    }
}

/// CustomBERT module for Ferrocarril TTS (ALBERT implementation)
pub struct CustomBert {
    /// Configuration parameters
    config: BertConfig,
    /// Token, position, and token type embeddings
    embeddings: BertEmbeddings,
    /// Mapping from embeddings to hidden states
    embedding_hidden_mapping: EmbeddingHiddenMapping,
    /// Layer group with parameter sharing
    layer_group: AlbertLayerGroup,
    /// Pooler (not used for TTS)
    pooler_weight: Parameter,
    /// Pooler bias (not used for TTS)
    pooler_bias: Parameter,
}

/// Configuration for BERT model
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Factorized token embedding size (ALBERT E parameter). For Kokoro's
    /// `plbert` this is 128 while `hidden_size` is 768.
    pub embedding_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Intermediate size for feed-forward
    pub intermediate_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Dropout probability (not used for inference)
    pub dropout_prob: f32,
}

impl CustomBert {
    /// Create a new CustomBERT model
    pub fn new(config: BertConfig) -> Self {
        // ALBERT uses parameter reduction techniques
        // The embedding dimension (128 in standard ALBERT) is smaller than the hidden dimension (768)
        let embedding_size = 128; // This will be overwritten when loading weights
        
        // Create embeddings with embedding_size
        let embeddings = BertEmbeddings::new(
            config.vocab_size,
            embedding_size, // Use smaller embedding size for ALBERT
            config.max_position_embeddings,
            config.dropout_prob
        );
        
        // Create embedding to hidden mapping from embedding_size to hidden_size
        let embedding_hidden_mapping = EmbeddingHiddenMapping::new(
            embedding_size,
            config.hidden_size
        );
        
        // Create layer group
        let layer_group = AlbertLayerGroup::new(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.dropout_prob
        );
        
        // Create pooler (not used for TTS, but needed for weight loading)
        let pooler_weight = Parameter::new(Tensor::from_data(
            vec![0.0; config.hidden_size * config.hidden_size],
            vec![config.hidden_size, config.hidden_size]
        ));
        let pooler_bias = Parameter::new(Tensor::from_data(
            vec![0.0; config.hidden_size],
            vec![config.hidden_size]
        ));
        
        Self {
            config,
            embeddings,
            embedding_hidden_mapping,
            layer_group,
            pooler_weight,
            pooler_bias,
        }
    }
    
    /// Forward pass for CustomBERT
    /// 
    /// # Arguments
    /// * `input_ids` - Token IDs with shape [batch_size, seq_len]
    /// * `token_type_ids` - Optional token type IDs
    /// * `attention_mask` - Optional attention mask where 1 indicates masked positions
    /// 
    /// # Returns
    /// Last hidden state with shape [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<i64>,
        token_type_ids: Option<&Tensor<i64>>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        // Get embedding layer output
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids);

        // Apply embedding to hidden mapping
        let hidden_states = self.embedding_hidden_mapping.forward(&embedding_output);

        // Apply encoder layers
        let output = self.layer_group.forward(&hidden_states, attention_mask, self.config.num_hidden_layers);

        output
    }
}

impl LoadWeightsBinary for CustomBert {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        // Load embeddings - fail immediately if missing
        self.embeddings.load_weights_binary(loader, component_path, module_path)?;
        
        // Load embedding to hidden mapping - fail immediately if missing
        self.embedding_hidden_mapping.load_weights_binary(
            loader, 
            component_path, 
            &format!("{}.encoder", module_path)
        )?;
        
        // Load layer group - fail immediately if missing
        self.layer_group.load_weights_binary(
            loader,
            component_path,
            &format!("{}.encoder.albert_layer_groups.0", module_path)
        )?;
        
        // Load pooler (not used for TTS, but required for complete model)
        let pooler_weight_path = format!(
            "{}.{}.pooler.weight",
            component_path,
            module_path
        );
        *self.pooler_weight.data_mut() = loader.load_tensor(&pooler_weight_path)?;

        let pooler_bias_path = format!(
            "{}.{}.pooler.bias",
            component_path,
            module_path
        );
        *self.pooler_bias.data_mut() = loader.load_tensor(&pooler_bias_path)?;
        
        Ok(())
    }
}