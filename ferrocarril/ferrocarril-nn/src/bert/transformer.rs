//! ALBERT transformer implementation for BERT component
//!
//! ALBERT-style transformer layers with parameter sharing, plus the
//! full `CustomBert` module. All hot-path forward methods use
//! direct-slice access and go through the fast `linear_f32` kernel
//! for projections.

#![allow(dead_code)]

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::ops::matmul::linear_f32;
use ferrocarril_core::weights_binary::BinaryWeightLoader;

use super::embeddings::BertEmbeddings;
use super::attention::MultiHeadAttention;
use super::feed_forward::FeedForward;
use super::layer_norm::LayerNorm;

/// ALBERT layer (single transformer layer with attention and feed-forward)
pub struct AlbertLayer {
    hidden_size: usize,
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    full_layer_norm: LayerNorm,
    dropout_prob: f32,
}

impl AlbertLayer {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Self {
        let attention = MultiHeadAttention::new(hidden_size, num_attention_heads, dropout_prob);
        let feed_forward = FeedForward::new(hidden_size, intermediate_size, dropout_prob);
        let full_layer_norm = LayerNorm::new(hidden_size, 1e-12);

        Self {
            hidden_size,
            attention,
            feed_forward,
            full_layer_norm,
            dropout_prob,
        }
    }

    /// Forward pass: attention → feed-forward → residual(attention) → layernorm.
    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        let attention_output = self.attention.forward(hidden_states, attention_mask);
        let feed_forward_output = self.feed_forward.forward(&attention_output);

        // Residual: feed_forward + attention (both are [B, T, H]).
        assert_eq!(
            feed_forward_output.shape(),
            attention_output.shape(),
            "AlbertLayer: ff output shape {:?} does not match attn shape {:?}",
            feed_forward_output.shape(),
            attention_output.shape()
        );

        let shape = feed_forward_output.shape().to_vec();
        let ff = feed_forward_output.data();
        let att = attention_output.data();
        debug_assert_eq!(ff.len(), att.len());
        let mut residual = vec![0.0f32; ff.len()];
        for i in 0..ff.len() {
            residual[i] = ff[i] + att[i];
        }

        let residual_output = Tensor::from_data(residual, shape);
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
        self.attention.load_weights_binary(
            loader,
            component_path,
            &format!("{}.attention", module_path),
        )?;
        self.feed_forward
            .load_weights_binary(loader, component_path, module_path)?;
        self.full_layer_norm.load_weights_binary(
            loader,
            component_path,
            &format!("{}.full_layer_layer_norm", module_path),
        )?;

        Ok(())
    }
}

/// ALBERT layer group with parameter sharing
pub struct AlbertLayerGroup {
    albert_layer: AlbertLayer,
}

impl AlbertLayerGroup {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        dropout_prob: f32,
    ) -> Self {
        let albert_layer = AlbertLayer::new(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout_prob,
        );

        Self { albert_layer }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<f32>,
        attention_mask: Option<&Tensor<i64>>,
        num_hidden_layers: usize,
    ) -> Tensor<f32> {
        let mut layer_output = hidden_states.clone();

        let max_layers = 24;
        let applied_layers = num_hidden_layers.min(max_layers);

        if applied_layers < num_hidden_layers {
            eprintln!(
                "ferrocarril: warning: limiting BERT layers from {} to max {}",
                num_hidden_layers, applied_layers
            );
        }

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
        self.albert_layer.load_weights_binary(
            loader,
            component_path,
            &format!("{}.albert_layers.0", module_path),
        )?;

        Ok(())
    }
}

/// Linear mapping from embeddings to hidden states (ALBERT factorised
/// embedding projection). Uses `linear_f32` so the one call per
/// forward pass runs at the same speed as a regular `Linear`.
struct EmbeddingHiddenMapping {
    input_dim: usize,
    output_dim: usize,
    weight: Parameter,
    bias: Parameter,
}

impl EmbeddingHiddenMapping {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim * input_dim],
            vec![output_dim, input_dim],
        ));
        let bias = Parameter::new(Tensor::from_data(
            vec![0.0; output_dim],
            vec![output_dim],
        ));

        Self {
            input_dim,
            output_dim,
            weight,
            bias,
        }
    }

    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        assert_eq!(
            shape.len(),
            3,
            "EmbeddingHiddenMapping: expected 3D input [B, T, in], got {:?}",
            shape
        );
        let batch_size = shape[0];
        let seq_len = shape[1];
        let input_dim = shape[2];
        assert_eq!(
            input_dim, self.input_dim,
            "EmbeddingHiddenMapping: input dim {} does not match configured {}",
            input_dim, self.input_dim
        );

        let m = batch_size * seq_len;
        let k = self.input_dim;
        let n = self.output_dim;

        let mut out = vec![0.0f32; m * n];
        linear_f32(
            x.data(),
            self.weight.data().data(),
            Some(self.bias.data().data()),
            &mut out,
            m,
            k,
            n,
        );

        Tensor::from_data(out, vec![batch_size, seq_len, n])
    }

    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        let weight_path = format!(
            "{}.{}.embedding_hidden_mapping_in.weight",
            component_path, module_path
        );
        *self.weight.data_mut() = loader.load_tensor(&weight_path)?;

        let bias_path = format!(
            "{}.{}.embedding_hidden_mapping_in.bias",
            component_path, module_path
        );
        *self.bias.data_mut() = loader.load_tensor(&bias_path)?;

        Ok(())
    }
}

/// CustomBERT module for Ferrocarril TTS (ALBERT implementation)
pub struct CustomBert {
    config: BertConfig,
    embeddings: BertEmbeddings,
    embedding_hidden_mapping: EmbeddingHiddenMapping,
    layer_group: AlbertLayerGroup,
    pooler_weight: Parameter,
    pooler_bias: Parameter,
}

/// Configuration for BERT model
pub struct BertConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub dropout_prob: f32,
}

impl CustomBert {
    pub fn new(config: BertConfig) -> Self {
        // ALBERT uses factorised embedding.
        let embedding_size = 128;

        let embeddings = BertEmbeddings::new(
            config.vocab_size,
            embedding_size,
            config.max_position_embeddings,
            config.dropout_prob,
        );

        let embedding_hidden_mapping = EmbeddingHiddenMapping::new(embedding_size, config.hidden_size);

        let layer_group = AlbertLayerGroup::new(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.dropout_prob,
        );

        let pooler_weight = Parameter::new(Tensor::from_data(
            vec![0.0; config.hidden_size * config.hidden_size],
            vec![config.hidden_size, config.hidden_size],
        ));
        let pooler_bias = Parameter::new(Tensor::from_data(
            vec![0.0; config.hidden_size],
            vec![config.hidden_size],
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

    pub fn forward(
        &self,
        input_ids: &Tensor<i64>,
        token_type_ids: Option<&Tensor<i64>>,
        attention_mask: Option<&Tensor<i64>>,
    ) -> Tensor<f32> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids);
        let hidden_states = self.embedding_hidden_mapping.forward(&embedding_output);
        self.layer_group
            .forward(&hidden_states, attention_mask, self.config.num_hidden_layers)
    }
}

impl LoadWeightsBinary for CustomBert {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        self.embeddings
            .load_weights_binary(loader, component_path, module_path)?;

        self.embedding_hidden_mapping.load_weights_binary(
            loader,
            component_path,
            &format!("{}.encoder", module_path),
        )?;

        self.layer_group.load_weights_binary(
            loader,
            component_path,
            &format!("{}.encoder.albert_layer_groups.0", module_path),
        )?;

        let pooler_weight_path = format!("{}.{}.pooler.weight", component_path, module_path);
        *self.pooler_weight.data_mut() = loader.load_tensor(&pooler_weight_path)?;

        let pooler_bias_path = format!("{}.{}.pooler.bias", component_path, module_path);
        *self.pooler_bias.data_mut() = loader.load_tensor(&pooler_bias_path)?;

        Ok(())
    }
}