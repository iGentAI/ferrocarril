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
    pub(crate) eps: f32,
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

    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

impl Forward for InstanceNorm1d {
    type Output = Tensor<f32>;

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        // Direct-slice implementation. Three contiguous passes per
        // `(batch, channel)`: mean, variance, then normalize+affine.
        // Each pass streams `length` contiguous f32s from a single
        // slice, which LLVM auto-vectorises.
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            3,
            "InstanceNorm1d: expected 3D input [batch, channels, length], got {:?}",
            shape
        );
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        assert_eq!(
            channels, self.num_features,
            "InstanceNorm1d: input channels {} != configured num_features {}",
            channels, self.num_features
        );

        let x_data = input.data();
        let mut out = vec![0.0f32; batch * channels * length];

        let len_f = length as f32;
        let inv_len = 1.0f32 / len_f;

        // Optional affine weight/bias (Kokoro uses affine=false; the
        // code below reads them only when `self.affine` is set so the
        // fast path has no branch inside the inner loop).
        let aff_w: Option<&[f32]> = self.weight.as_ref().map(|p| p.data().data());
        let aff_b: Option<&[f32]> = self.bias.as_ref().map(|p| p.data().data());

        for b in 0..batch {
            let b_off = b * channels * length;
            for c in 0..channels {
                let start = b_off + c * length;
                let chan = &x_data[start..start + length];

                // Pass 1: sum → mean
                let mut sum = 0.0f32;
                for &v in chan {
                    sum += v;
                }
                let mean = sum * inv_len;

                // Pass 2: sum of squared deviations → variance → inv_std
                let mut sq_sum = 0.0f32;
                for &v in chan {
                    let d = v - mean;
                    sq_sum += d * d;
                }
                let var = sq_sum * inv_len;
                let inv_std = 1.0f32 / (var + self.eps).sqrt();

                // Pass 3: normalize, then optionally affine.
                let out_chan = &mut out[start..start + length];
                if self.affine {
                    let w = aff_w.unwrap()[c];
                    let bv = aff_b.unwrap()[c];
                    // y = w * ((x - mean) * inv_std) + bv
                    //   = (w * inv_std) * x + (bv - w * inv_std * mean)
                    let scale = w * inv_std;
                    let offset = bv - mean * scale;
                    for (dst, &v) in out_chan.iter_mut().zip(chan.iter()) {
                        *dst = v * scale + offset;
                    }
                } else {
                    // y = (x - mean) * inv_std
                    //   = inv_std * x - mean * inv_std
                    let offset = -mean * inv_std;
                    for (dst, &v) in out_chan.iter_mut().zip(chan.iter()) {
                        *dst = v * inv_std + offset;
                    }
                }
            }
        }

        Tensor::from_data(out, shape.to_vec())
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
    #[allow(dead_code)]
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

    /// Forward pass.
    ///
    /// Fused implementation: computes `y = (1 + gamma) * normalize(x) + beta`
    /// without materialising the intermediate normalised tensor. Per
    /// `(batch, channel)`:
    ///   1. Compute mean (contiguous sum over `length` elements).
    ///   2. Compute variance (contiguous sum of squared deviations).
    ///   3. Combine the normalize + affine into a single
    ///      `y = scale * x + offset` pass, where
    ///      `scale = (1 + gamma) * inv_std` and
    ///      `offset = beta - mean * scale`.
    ///
    /// All three passes are contiguous `length`-element loops that the
    /// compiler auto-vectorises into wide SIMD FMA sequences.
    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // Style → gamma/beta.
        let h = self.fc.forward(style);
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
        let h_data = h.data();
        let h_row_stride = 2 * self.num_features;

        // Input x: (B, C, L)
        let x_shape = x.shape();
        assert_eq!(
            x_shape.len(),
            3,
            "AdaIN1d: input x must be 3D [batch, channels, length], got {:?}",
            x_shape
        );
        let (batch, channels, length) = (x_shape[0], x_shape[1], x_shape[2]);
        assert_eq!(
            channels, self.num_features,
            "AdaIN1d: input channel count {} does not match num_features {}",
            channels, self.num_features
        );
        assert_eq!(
            h_shape[0], batch,
            "AdaIN1d: style batch {} does not match input batch {}",
            h_shape[0], batch
        );

        let x_data = x.data();
        let eps = self.norm.eps;
        let len_f = length as f32;
        let inv_len = 1.0f32 / len_f;

        let mut out = vec![0.0f32; batch * channels * length];

        for b in 0..batch {
            let b_off = b * channels * length;
            let h_row = &h_data[b * h_row_stride..(b + 1) * h_row_stride];

            for c in 0..channels {
                let gamma = h_row[c];
                let beta = h_row[c + channels];

                let start = b_off + c * length;
                let chan = &x_data[start..start + length];

                // Pass 1: sum → mean
                let mut sum = 0.0f32;
                for &v in chan {
                    sum += v;
                }
                let mean = sum * inv_len;

                // Pass 2: sq_sum → variance → inv_std
                let mut sq_sum = 0.0f32;
                for &v in chan {
                    let d = v - mean;
                    sq_sum += d * d;
                }
                let var = sq_sum * inv_len;
                let inv_std = 1.0f32 / (var + eps).sqrt();

                // Pass 3: fused normalize + affine.
                //   y = (1 + gamma) * ((x - mean) * inv_std) + beta
                //     = scale * x + offset
                // where scale  = (1 + gamma) * inv_std
                //       offset = beta - mean * scale
                let scale = (1.0 + gamma) * inv_std;
                let offset = beta - mean * scale;

                let out_chan = &mut out[start..start + length];
                for (dst, &v) in out_chan.iter_mut().zip(chan.iter()) {
                    *dst = v * scale + offset;
                }
            }
        }

        Tensor::from_data(out, x_shape.to_vec())
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