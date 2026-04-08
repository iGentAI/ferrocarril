//! Feed-forward network implementation for BERT
//!
//! Two linear transformations with a GELU activation in between.
//! Both linear ops go through the optimised `linear_f32` kernel in
//! `ferrocarril-core::ops::matmul`, and GELU runs as a contiguous
//! in-place pass over the flattened intermediate buffer.

#![allow(dead_code)]

use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{Parameter, LoadWeightsBinary, FerroError};
use ferrocarril_core::ops::matmul::linear_f32;
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
        let ffn_weight = Parameter::new(Tensor::from_data(
            vec![0.0; intermediate_size * hidden_size],
            vec![intermediate_size, hidden_size],
        ));
        let ffn_bias = Parameter::new(Tensor::from_data(
            vec![0.0; intermediate_size],
            vec![intermediate_size],
        ));
        let ffn_output_weight = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size * intermediate_size],
            vec![hidden_size, intermediate_size],
        ));
        let ffn_output_bias = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size],
            vec![hidden_size],
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

    /// Approximate GELU as used in BERT:
    ///   `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
    #[inline]
    fn gelu(x: f32) -> f32 {
        const SCALE: f32 = 0.7978845608028654; // sqrt(2/π)
        const C: f32 = 0.044715;
        let x3 = x * x * x;
        let tanh_input = SCALE * (x + C * x3);
        0.5 * x * (1.0 + tanh_input.tanh())
    }

    /// Forward pass for the feed-forward network.
    ///
    /// Input / output: `[B, T, hidden_size]` tensors. Internally
    /// flattens to `[m = B*T, hidden_size]` so the two `linear_f32`
    /// calls operate on a single dense `m × k × n` matmul each.
    pub fn forward(&self, hidden_states: &Tensor<f32>) -> Tensor<f32> {
        let shape = hidden_states.shape();
        assert_eq!(
            shape.len(),
            3,
            "FeedForward: expected 3D input [B, T, hidden_size], got {:?}",
            shape
        );
        let batch_size = shape[0];
        let seq_len = shape[1];
        let input_hidden_size = shape[2];
        assert_eq!(
            input_hidden_size, self.hidden_size,
            "FeedForward: input hidden {} does not match configured {}",
            input_hidden_size, self.hidden_size
        );

        let m = batch_size * seq_len;

        // First projection: hidden → intermediate
        let mut intermediate = vec![0.0f32; m * self.intermediate_size];
        linear_f32(
            hidden_states.data(),
            self.ffn_weight.data().data(),
            Some(self.ffn_bias.data().data()),
            &mut intermediate,
            m,
            self.hidden_size,
            self.intermediate_size,
        );

        // GELU activation (in place). Contiguous pass → vectorised.
        for v in intermediate.iter_mut() {
            *v = Self::gelu(*v);
        }

        // Second projection: intermediate → hidden
        let mut output = vec![0.0f32; m * self.hidden_size];
        linear_f32(
            &intermediate,
            self.ffn_output_weight.data().data(),
            Some(self.ffn_output_bias.data().data()),
            &mut output,
            m,
            self.intermediate_size,
            self.hidden_size,
        );

        Tensor::from_data(output, vec![batch_size, seq_len, self.hidden_size])
    }
}

impl LoadWeightsBinary for FeedForward {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        let ffn_weight_path = format!("{}.{}.ffn.weight", component_path, module_path);
        let ffn_weight_tensor = loader.load_tensor(&ffn_weight_path)?;
        if ffn_weight_tensor.shape()[0] != self.intermediate_size
            || ffn_weight_tensor.shape()[1] != self.hidden_size
        {
            return Err(FerroError::new(format!(
                "FFN weight has incorrect shape: expected [{}, {}], got {:?}",
                self.intermediate_size,
                self.hidden_size,
                ffn_weight_tensor.shape()
            )));
        }
        *self.ffn_weight.data_mut() = ffn_weight_tensor;

        let ffn_bias_path = format!("{}.{}.ffn.bias", component_path, module_path);
        let ffn_bias_tensor = loader.load_tensor(&ffn_bias_path)?;
        if ffn_bias_tensor.shape()[0] != self.intermediate_size {
            return Err(FerroError::new(format!(
                "FFN bias has incorrect shape: expected [{}], got {:?}",
                self.intermediate_size,
                ffn_bias_tensor.shape()
            )));
        }
        *self.ffn_bias.data_mut() = ffn_bias_tensor;

        let ffn_output_weight_path =
            format!("{}.{}.ffn_output.weight", component_path, module_path);
        let ffn_output_weight_tensor = loader.load_tensor(&ffn_output_weight_path)?;
        if ffn_output_weight_tensor.shape()[0] != self.hidden_size
            || ffn_output_weight_tensor.shape()[1] != self.intermediate_size
        {
            return Err(FerroError::new(format!(
                "FFN output weight has incorrect shape: expected [{}, {}], got {:?}",
                self.hidden_size,
                self.intermediate_size,
                ffn_output_weight_tensor.shape()
            )));
        }
        *self.ffn_output_weight.data_mut() = ffn_output_weight_tensor;

        let ffn_output_bias_path =
            format!("{}.{}.ffn_output.bias", component_path, module_path);
        let ffn_output_bias_tensor = loader.load_tensor(&ffn_output_bias_path)?;
        if ffn_output_bias_tensor.shape()[0] != self.hidden_size {
            return Err(FerroError::new(format!(
                "FFN output bias has incorrect shape: expected [{}], got {:?}",
                self.hidden_size,
                ffn_output_bias_tensor.shape()
            )));
        }
        *self.ffn_output_bias.data_mut() = ffn_output_bias_tensor;

        Ok(())
    }
}