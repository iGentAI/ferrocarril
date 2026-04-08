//! Layer normalization implementation for ALBERT
//!
//! This implements layer normalization with trainable parameters for
//! scaling (gamma) and shifting (beta). The forward pass uses direct
//! slice access over the last-dim `hidden` slice of the input, with
//! three contiguous passes per row (sum → mean, squared-deviation →
//! variance, scale+shift), which LLVM auto-vectorises.

#![allow(dead_code)]

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
        let gamma = Parameter::new(Tensor::from_data(
            vec![1.0; hidden_size],
            vec![hidden_size],
        ));
        let beta = Parameter::new(Tensor::from_data(
            vec![0.0; hidden_size],
            vec![hidden_size],
        ));

        Self {
            hidden_size,
            gamma,
            beta,
            eps,
        }
    }

    /// Apply layer normalization to input tensor.
    ///
    /// Supports both 2D `[B, H]` and 3D `[B, T, H]` (or any rank where
    /// the last dim is the normalisation axis). Normalises each
    /// `hidden`-length row independently.
    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let shape = x.shape();
        assert!(
            !shape.is_empty(),
            "LayerNorm: expected at least 1D input, got {:?}",
            shape
        );

        let hidden = *shape.last().unwrap();
        assert_eq!(
            hidden,
            self.gamma.data().shape()[0],
            "LayerNorm: input last-dim {} does not match configured hidden_size {}",
            hidden,
            self.gamma.data().shape()[0]
        );

        let gamma = self.gamma.data().data();
        let beta = self.beta.data().data();
        let x_data = x.data();
        let eps = self.eps;
        let inv_h = 1.0f32 / hidden as f32;

        let num_rows = x_data.len() / hidden;
        debug_assert_eq!(num_rows * hidden, x_data.len());

        let mut out = vec![0.0f32; x_data.len()];

        for row in 0..num_rows {
            let start = row * hidden;
            let in_row = &x_data[start..start + hidden];
            let out_row = &mut out[start..start + hidden];

            // Pass 1: sum → mean
            let mut sum = 0.0f32;
            for &v in in_row {
                sum += v;
            }
            let mean = sum * inv_h;

            // Pass 2: sum of squared deviations → variance → inv_std
            let mut sq_sum = 0.0f32;
            for &v in in_row {
                let d = v - mean;
                sq_sum += d * d;
            }
            let var = sq_sum * inv_h;
            let inv_std = 1.0f32 / (var + eps).sqrt();

            // Pass 3: fused normalise + affine
            //   y = (x - mean) * inv_std * gamma + beta
            //     = x * (inv_std * gamma) + (beta - mean * inv_std * gamma)
            //     = x * scale + offset   (but we keep the explicit form
            //                             since gamma/beta vary per channel)
            for (i, &v) in in_row.iter().enumerate() {
                out_row[i] = (v - mean) * inv_std * gamma[i] + beta[i];
            }
        }

        Tensor::from_data(out, shape.to_vec())
    }
}

impl LoadWeightsBinary for LayerNorm {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component_path: &str,
        module_path: &str,
    ) -> Result<(), FerroError> {
        // Load gamma weights (weight). Try the LayerNorm-suffixed key
        // first and fall back to the bare `.weight` key.
        let gamma_path = format!("{}.{}.LayerNorm.weight", component_path, module_path);
        if let Ok(gamma) = loader.load_tensor(&gamma_path) {
            *self.gamma.data_mut() = gamma;
        } else {
            let alt_gamma_path = format!("{}.{}.weight", component_path, module_path);
            *self.gamma.data_mut() = loader.load_tensor(&alt_gamma_path)?;
        }

        let beta_path = format!("{}.{}.LayerNorm.bias", component_path, module_path);
        if let Ok(beta) = loader.load_tensor(&beta_path) {
            *self.beta.data_mut() = beta;
        } else {
            let alt_beta_path = format!("{}.{}.bias", component_path, module_path);
            *self.beta.data_mut() = loader.load_tensor(&alt_beta_path)?;
        }

        Ok(())
    }
}