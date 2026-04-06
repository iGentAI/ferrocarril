//! 1D convolution implementation.
//!
//! Supports the standard `Conv1d(weight, bias)` form and also the PyTorch
//! `weight_norm` parameterization used extensively by Kokoro's iSTFTNet
//! decoder, where the weight is stored as two tensors (`weight_g`, `weight_v`)
//! and the effective weight is reconstructed on load as
//! `w[o] = g[o] * v[o] / ‖v[o]‖₂`.

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

#[derive(Debug)]
pub struct Conv1d {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    in_channels: usize,
    out_channels: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
        assert!(out_channels % groups == 0, "out_channels must be divisible by groups");

        let weight_shape = vec![out_channels, in_channels / groups, kernel_size];
        let weight = Parameter::new(Tensor::new(weight_shape));

        let bias = if bias {
            Some(Parameter::new(Tensor::new(vec![out_channels])))
        } else {
            None
        };

        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            in_channels,
            out_channels,
        }
    }

    /// Accessor for the current weight parameter.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// Accessor for the optional bias parameter.
    pub fn bias_opt(&self) -> Option<&Parameter> {
        self.bias.as_ref()
    }

    /// Reconstruct the convolution weight from the PyTorch `weight_norm`
    /// parameterization and install it in place.
    ///
    /// In PyTorch, `weight_norm(module, dim=0)` stores the weight as two
    /// tensors:
    ///
    /// * `weight_g` — scalar magnitude per output channel. Shape can be
    ///   `[out_c]`, `[out_c, 1]`, or `[out_c, 1, 1]`; in every case the
    ///   underlying flat storage holds `out_c` values.
    /// * `weight_v` — direction, shape `[out_c, in_c/groups, kernel]`, same
    ///   shape as the unnormalized weight.
    ///
    /// The effective weight is:
    ///
    /// `w[o, i, k] = g[o] * v[o, i, k] / ‖v[o, :, :]‖₂`
    ///
    /// where the L2 norm is taken over every dim except the first.
    pub fn set_weight_norm(
        &mut self,
        weight_g: &Tensor<f32>,
        weight_v: &Tensor<f32>,
    ) -> Result<(), FerroError> {
        let v_shape = weight_v.shape().to_vec();
        if v_shape.is_empty() {
            return Err(FerroError::new(
                "set_weight_norm: weight_v has zero dimensions",
            ));
        }
        let out_c = v_shape[0];
        if out_c == 0 {
            return Err(FerroError::new(
                "set_weight_norm: weight_v out-channel dim is zero",
            ));
        }
        let rest: usize = v_shape.iter().skip(1).product();
        if rest == 0 {
            return Err(FerroError::new(format!(
                "set_weight_norm: weight_v inner dims product to zero ({:?})",
                v_shape
            )));
        }

        let g_data = weight_g.data();
        if g_data.len() != out_c {
            return Err(FerroError::new(format!(
                "set_weight_norm: weight_g has {} elements, expected {} (out_channels from weight_v shape {:?})",
                g_data.len(),
                out_c,
                v_shape
            )));
        }

        let v_data = weight_v.data();
        if v_data.len() != out_c * rest {
            return Err(FerroError::new(format!(
                "set_weight_norm: weight_v data length {} does not match shape {:?}",
                v_data.len(),
                v_shape
            )));
        }

        let mut result = Vec::with_capacity(v_data.len());
        for oc in 0..out_c {
            let start = oc * rest;
            let end = start + rest;
            let slice = &v_data[start..end];
            let norm_sq: f32 = slice.iter().map(|&x| x * x).sum();
            // Guard against zero-norm channels; fall back to g*v (i.e. scale 0
            // stays 0). A true 1/0 would be a NaN trap, and Kokoro weights do
            // not contain zero-norm channels in practice.
            let norm = norm_sq.sqrt().max(1e-12);
            let scale = g_data[oc] / norm;
            for &v in slice {
                result.push(v * scale);
            }
        }

        self.weight = Parameter::new(Tensor::from_data(result, v_shape));
        Ok(())
    }

    /// Install a bias tensor on this conv layer. Promotes a biasless conv to
    /// a biased conv if necessary.
    pub fn set_bias(&mut self, bias: &Tensor<f32>) -> Result<(), FerroError> {
        if bias.shape().len() != 1 {
            return Err(FerroError::new(format!(
                "set_bias: bias must be 1D, got shape {:?}",
                bias.shape()
            )));
        }
        if bias.shape()[0] != self.out_channels {
            return Err(FerroError::new(format!(
                "set_bias: bias length {} does not match out_channels {}",
                bias.shape()[0],
                self.out_channels
            )));
        }
        self.bias = Some(Parameter::new(bias.clone()));
        Ok(())
    }

    fn conv1d(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight = self.weight.data();
        let input_shape = input.shape();
        let weight_shape = weight.shape();

        assert_eq!(input_shape.len(), 3, "Input must be 3D: [batch_size, channels, length]");
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);

        assert_eq!(in_channels, self.in_channels, "Input channels mismatch");

        let kernel_size = weight_shape[2];
        let out_channels = self.out_channels;

        // Calculate output dimensions
        let dilated_kernel_size = (kernel_size - 1) * self.dilation + 1;
        let out_length = (in_length + 2 * self.padding - dilated_kernel_size) / self.stride + 1;

        let mut output = Tensor::new(vec![batch_size, out_channels, out_length]);

        // Perform convolution (naive implementation for MVP)
        for b in 0..batch_size {
            for oc in 0..out_channels {
                let group_id = oc / (out_channels / self.groups);
                let group_in_channels = in_channels / self.groups;
                let group_in_start = group_id * group_in_channels;

                for ol in 0..out_length {
                    let mut sum = 0.0;

                    // Convolve over input channels in this group
                    for ic_rel in 0..group_in_channels {
                        let ic = group_in_start + ic_rel;

                        // Convolve over kernel
                        for k in 0..kernel_size {
                            let il = ol * self.stride + k * self.dilation;
                            if il < self.padding || il >= in_length + self.padding {
                                continue;
                            }
                            let il_actual = il - self.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight[&[oc, ic_rel, k]];
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[&[oc]];
                    }

                    output[&[b, oc, ol]] = sum;
                }
            }
        }

        output
    }
}

impl Forward for Conv1d {
    type Output = Tensor<f32>;

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        self.conv1d(input)
    }
}

/// Create a Conv1d layer with standard defaults
pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
) -> Conv1d {
    let padding = (kernel_size - 1) / 2 * dilation;
    Conv1d::new(
        in_channels,
        out_channels,
        kernel_size,
        1, // stride
        padding,
        dilation,
        1, // groups
        true, // bias
    )
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for Conv1d {
    /// Load `Conv1d` weights from the converted binary format.
    ///
    /// Kokoro's iSTFTNet decoder uses `torch.nn.utils.weight_norm` on almost
    /// every conv layer, so the typical on-disk layout is
    /// `{prefix}.weight_g` + `{prefix}.weight_v` (+ optional `{prefix}.bias`).
    /// A small number of Ferrocarril call sites historically built the
    /// sub-prefix with an underscore instead of a dot (e.g.
    /// `{prefix}_weight_g`), so we try the dotted form first and fall back
    /// to the underscored form, and finally to a plain `{prefix}.weight`
    /// for conv layers that are not weight-normed.
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let dot_g = format!("{}.weight_g", prefix);
        let dot_v = format!("{}.weight_v", prefix);
        let under_g = format!("{}_weight_g", prefix);
        let under_v = format!("{}_weight_v", prefix);
        let plain_w = format!("{}.weight", prefix);
        let dot_b = format!("{}.bias", prefix);
        let under_b = format!("{}_bias", prefix);

        // Try dotted weight_norm first.
        let mut weight_loaded = false;
        if let (Ok(g), Ok(v)) = (
            loader.load_component_parameter(component, &dot_g),
            loader.load_component_parameter(component, &dot_v),
        ) {
            self.set_weight_norm(&g, &v)?;
            weight_loaded = true;
        }

        // Fall back to underscored weight_norm.
        if !weight_loaded {
            if let (Ok(g), Ok(v)) = (
                loader.load_component_parameter(component, &under_g),
                loader.load_component_parameter(component, &under_v),
            ) {
                self.set_weight_norm(&g, &v)?;
                weight_loaded = true;
            }
        }

        // Fall back to a plain weight tensor.
        if !weight_loaded {
            let w = loader
                .load_component_parameter(component, &plain_w)
                .map_err(|e| {
                    FerroError::new(format!(
                        "Conv1d::load_weights_binary: no weight_norm (.weight_g/_weight_g) nor plain .weight for '{}.{}': {}",
                        component, prefix, e
                    ))
                })?;
            self.weight = Parameter::new(w);
            weight_loaded = true;
        }
        debug_assert!(weight_loaded);

        // Bias is optional. Try dotted first, then underscored. A biasless
        // conv simply leaves `self.bias` as-is.
        if let Ok(b) = loader.load_component_parameter(component, &dot_b) {
            // Accept biases that come in at a shape such as [out_c, 1] by
            // flattening. The Kokoro converter sometimes preserves the PyTorch
            // module's stored bias shape rather than flattening it.
            let flat = b.data().to_vec();
            if flat.len() == self.out_channels {
                self.bias = Some(Parameter::new(Tensor::from_data(
                    flat,
                    vec![self.out_channels],
                )));
            } else {
                return Err(FerroError::new(format!(
                    "Conv1d::load_weights_binary: bias '{}.{}' has {} elements, expected {}",
                    component,
                    dot_b,
                    flat.len(),
                    self.out_channels
                )));
            }
        } else if let Ok(b) = loader.load_component_parameter(component, &under_b) {
            let flat = b.data().to_vec();
            if flat.len() == self.out_channels {
                self.bias = Some(Parameter::new(Tensor::from_data(
                    flat,
                    vec![self.out_channels],
                )));
            } else {
                return Err(FerroError::new(format!(
                    "Conv1d::load_weights_binary: bias '{}.{}' has {} elements, expected {}",
                    component,
                    under_b,
                    flat.len(),
                    self.out_channels
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1d::new(3, 6, 3, 1, 1, 1, 1, true);
        let input = Tensor::new(vec![2, 3, 10]); // [batch_size, channels, length]
        let output = conv.forward(&input);

        assert_eq!(output.shape(), &[2, 6, 10]);
    }

    #[test]
    fn test_set_weight_norm_reconstruction() {
        // A 2-out-channel conv with a known v and g; check that the
        // reconstructed weight is g * v / ||v||.
        let mut conv = Conv1d::new(1, 2, 3, 1, 0, 1, 1, false);
        // weight_v: [[1, 0, 0], [0, 2, 0]] over shape [2, 1, 3]
        let wv = Tensor::from_data(
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            vec![2, 1, 3],
        );
        // weight_g: [3, 4] → reconstructed = [[3, 0, 0], [0, 4, 0]]
        let wg = Tensor::from_data(vec![3.0, 4.0], vec![2]);

        conv.set_weight_norm(&wg, &wv).expect("set_weight_norm");

        let w = conv.weight().data();
        let want = vec![3.0, 0.0, 0.0, 0.0, 4.0, 0.0];
        for (i, &exp) in want.iter().enumerate() {
            let got = w.data()[i];
            assert!(
                (got - exp).abs() < 1e-6,
                "element {} mismatch: got {}, want {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_set_bias_installs_tensor() {
        let mut conv = Conv1d::new(1, 4, 3, 1, 0, 1, 1, false);
        let b = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], vec![4]);
        conv.set_bias(&b).expect("set_bias");
        let installed = conv.bias_opt().expect("bias installed").data();
        assert_eq!(installed.shape(), &[4]);
        assert!((installed.data()[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_set_bias_rejects_wrong_shape() {
        let mut conv = Conv1d::new(1, 4, 3, 1, 0, 1, 1, false);
        let bad = Tensor::from_data(vec![0.1, 0.2], vec![2]);
        assert!(conv.set_bias(&bad).is_err());
    }
}