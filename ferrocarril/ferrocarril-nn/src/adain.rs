//! Adaptive Instance Normalization (AdaIN) implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use crate::linear::Linear;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

/// Instance normalization for 1D data
#[derive(Debug)]
pub struct InstanceNorm1d {
    num_features: usize,
    eps: f32,
    affine: bool,
    weight: Option<Parameter>,
    bias: Option<Parameter>,
}

impl InstanceNorm1d {
    pub fn new(num_features: usize, eps: f32, affine: bool) -> Self {
        let (weight, bias) = if affine {
            (
                Some(Parameter::new(Tensor::from_data(vec![1.0; num_features], vec![num_features]))),
                Some(Parameter::new(Tensor::from_data(vec![0.0; num_features], vec![num_features]))),
            )
        } else {
            (None, None)
        };

        Self {
            num_features,
            eps,
            affine,
            weight,
            bias,
        }
    }

    /// Compute mean and variance for a channel
    fn compute_stats(&self, data: &[f32]) -> (f32, f32) {
        let len = data.len() as f32;

        // Compute mean
        let mean = data.iter().sum::<f32>() / len;

        // Compute variance
        let var = data.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>() / len;

        (mean, var)
    }

    /// Normalize data
    fn normalize(&self, data: &[f32], mean: f32, var: f32) -> Vec<f32> {
        let inv_std = 1.0 / (var + self.eps).sqrt();
        data.iter()
            .map(|&x| (x - mean) * inv_std)
            .collect()
    }
}

impl Forward for InstanceNorm1d {
    type Output = Tensor<f32>;

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let shape = input.shape();
        assert_eq!(shape.len(), 3, "Expected 3D input [batch, channels, length]");

        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        assert_eq!(channels, self.num_features, "Channel mismatch");

        let mut output_data = Vec::with_capacity(batch * channels * length);

        for b in 0..batch {
            for c in 0..channels {
                // Extract channel data
                let mut channel_data = Vec::with_capacity(length);
                for l in 0..length {
                    channel_data.push(input[&[b, c, l]]);
                }

                // Compute stats and normalize
                let (mean, var) = self.compute_stats(&channel_data);
                let normalized = self.normalize(&channel_data, mean, var);

                // Apply affine transformation if enabled
                let transformed = if self.affine {
                    let weight = self.weight.as_ref().unwrap().data();
                    let bias = self.bias.as_ref().unwrap().data();

                    normalized.iter()
                        .map(|&x| x * weight[&[c]] + bias[&[c]])
                        .collect()
                } else {
                    normalized
                };

                output_data.extend(transformed);
            }
        }

        Tensor::from_data(output_data, shape.to_vec())
    }
}

/// Adaptive Instance Normalization.
///
/// Matches Kokoro's Python definition:
/// ```python
/// class AdaIN1d(nn.Module):
///     def __init__(self, style_dim, num_features):
///         self.norm = nn.InstanceNorm1d(num_features, affine=False)
///         self.fc = nn.Linear(style_dim, num_features * 2)
/// ```
///
/// The `fc` field is public so callers and tests can poke it directly when
/// loading weights from non-binary sources (e.g. the in-process weight
/// fixtures in `tests/adain_test.rs`). Binary-weight loading goes through
/// `LoadWeightsBinary` below.
#[derive(Debug)]
pub struct AdaIN1d {
    style_dim: usize,
    num_features: usize,
    pub fc: Linear,
    pub norm: InstanceNorm1d,
}

impl AdaIN1d {
    pub fn new(style_dim: usize, num_features: usize) -> Self {
        Self {
            style_dim,
            num_features,
            fc: Linear::new(style_dim, num_features * 2, true),
            norm: InstanceNorm1d::new(num_features, 1e-5, false),
        }
    }

    /// Create a new AdaIN1d with a custom linear layer and normalization.
    pub fn with_linear(
        style_dim: usize,
        num_features: usize,
        fc: Linear,
        norm: InstanceNorm1d,
    ) -> Self {
        Self {
            style_dim,
            num_features,
            fc,
            norm,
        }
    }

    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // Style transformation
        let h = self.fc.forward(style);

        // Shape sanity: style FC must emit 2 * num_features so we can split
        // into gamma and beta.
        let h_shape = h.shape();
        assert_eq!(
            h_shape.len(),
            2,
            "AdaIN1d: style fc output must be 2D [batch, 2*num_features], got shape {:?}",
            h_shape
        );
        assert_eq!(
            h_shape[1],
            2 * self.num_features,
            "AdaIN1d: style fc out_features {} does not match 2 * num_features {}",
            h_shape[1],
            2 * self.num_features
        );
        let batch = h_shape[0];

        // Split into gamma and beta
        let mut gamma_data = Vec::with_capacity(batch * self.num_features);
        let mut beta_data = Vec::with_capacity(batch * self.num_features);

        for b in 0..batch {
            for i in 0..self.num_features {
                gamma_data.push(h[&[b, i]]);
                beta_data.push(h[&[b, i + self.num_features]]);
            }
        }

        let gamma = Tensor::from_data(gamma_data, vec![batch, self.num_features, 1]);
        let beta = Tensor::from_data(beta_data, vec![batch, self.num_features, 1]);

        // Apply normalization
        let normalized = self.norm.forward(x);

        // Apply style conditioning
        let shape = normalized.shape();
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        assert_eq!(
            channels,
            self.num_features,
            "AdaIN1d: input channel count {} does not match num_features {}",
            channels,
            self.num_features
        );
        let mut output_data = Vec::with_capacity(batch * channels * length);

        // Apply style directly: scale by gamma and shift by beta
        for b in 0..batch {
            for c in 0..channels {
                for l in 0..length {
                    let normalized_val = normalized[&[b, c, l]];
                    let gamma_val = gamma[&[b, c, 0]];
                    let beta_val = beta[&[b, c, 0]];

                    output_data.push((1.0 + gamma_val) * normalized_val + beta_val);
                }
            }
        }

        Tensor::from_data(output_data, shape.to_vec())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for InstanceNorm1d {
    /// `InstanceNorm1d` has no learnable parameters when `affine=false`
    /// (Kokoro's configuration), so the default load is a no-op. If
    /// `affine=true`, load `{prefix}.weight` and `{prefix}.bias`, each of
    /// length `num_features`.
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        if !self.affine {
            return Ok(());
        }

        let w_path = format!("{}.weight", prefix);
        let b_path = format!("{}.bias", prefix);

        let w = loader.load_component_parameter(component, &w_path).map_err(|e| {
            FerroError::new(format!(
                "InstanceNorm1d::load_weights_binary: missing '{}.{}': {}",
                component, w_path, e
            ))
        })?;
        let b = loader.load_component_parameter(component, &b_path).map_err(|e| {
            FerroError::new(format!(
                "InstanceNorm1d::load_weights_binary: missing '{}.{}': {}",
                component, b_path, e
            ))
        })?;

        if w.data().len() != self.num_features || b.data().len() != self.num_features {
            return Err(FerroError::new(format!(
                "InstanceNorm1d::load_weights_binary: shape mismatch at '{}.{}' — weight has {} elements, bias has {}, expected {}",
                component,
                prefix,
                w.data().len(),
                b.data().len(),
                self.num_features
            )));
        }

        self.weight = Some(Parameter::new(w));
        self.bias = Some(Parameter::new(b));
        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdaIN1d {
    /// `AdaIN1d` in Kokoro is
    /// ```python
    /// class AdaIN1d(nn.Module):
    ///     def __init__(self, style_dim, num_features):
    ///         self.norm = nn.InstanceNorm1d(num_features, affine=False)
    ///         self.fc = nn.Linear(style_dim, num_features * 2)
    /// ```
    /// so weights live at `{prefix}.fc.weight` and `{prefix}.fc.bias`, and
    /// the normalization layer has nothing to load.
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let fc_prefix = format!("{}.fc", prefix);
        self.fc
            .load_weights_binary(loader, component, &fc_prefix)
            .map_err(|e| {
                FerroError::new(format!(
                    "AdaIN1d::load_weights_binary: failed to load fc at '{}.{}': {}",
                    component, fc_prefix, e
                ))
            })?;
        // The InstanceNorm1d under `norm` has affine=false in Kokoro, so its
        // `load_weights_binary` is a no-op, but we call it for completeness
        // and forward compatibility with any affine=true uses.
        let norm_prefix = format!("{}.norm", prefix);
        self.norm
            .load_weights_binary(loader, component, &norm_prefix)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_norm1d() {
        let norm = InstanceNorm1d::new(3, 1e-5, true);
        let input = Tensor::from_data(vec![0.0; 24], vec![2, 3, 4]); // [batch, channels, length]
        let output = norm.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_adain1d() {
        let adain = AdaIN1d::new(10, 5);
        let x = Tensor::from_data(vec![0.0; 40], vec![2, 5, 4]); // [batch, channels, length]
        let style = Tensor::from_data(vec![0.0; 20], vec![2, 10]); // [batch, style_dim]
        let output = adain.forward(&x, &style);
        assert_eq!(output.shape(), x.shape());
    }
}