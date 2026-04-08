//! 1D Transposed Convolution (Deconvolution) Implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use ferrocarril_core::ops::matmul::matmul_f32;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

/// A 1D transposed convolution layer for upsampling
#[derive(Debug)]
pub struct ConvTranspose1d {
    weight: Parameter,   // [in_channels, out_channels/groups, kernel_size]
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        // Check divisibility, but don't assert - instead, adjust values to make them compatible.
        // In practice this branch never triggers for the real Kokoro weights, so the warnings
        // go to stderr to keep the production stdout clean.
        let (in_channels_adjusted, out_channels_adjusted) =
            if in_channels % groups != 0 || out_channels % groups != 0 {
                eprintln!(
                    "ferrocarril: warning: channel counts not divisible by groups in ConvTranspose1d (in={}, out={}, groups={}); adjusting for compatibility",
                    in_channels, out_channels, groups
                );
                let in_adjusted = (in_channels / groups) * groups;
                let out_adjusted = (out_channels / groups) * groups;
                (in_adjusted.max(groups), out_adjusted.max(groups))
            } else {
                (in_channels, out_channels)
            };

        if in_channels_adjusted != in_channels || out_channels_adjusted != out_channels {
            eprintln!(
                "ferrocarril: warning: ConvTranspose1d adjusted channels: in={} -> {}, out={} -> {}",
                in_channels, in_channels_adjusted,
                out_channels, out_channels_adjusted
            );
        }

        // Weight shape: [in_channels, out_channels/groups, kernel_size]
        let weight_shape = vec![in_channels_adjusted, out_channels_adjusted / groups, kernel_size];
        let weight = Parameter::new(Tensor::new(weight_shape));

        // Bias shape: [out_channels]
        let bias = if bias {
            Some(Parameter::new(Tensor::new(vec![out_channels_adjusted])))
        } else {
            None
        };

        Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            in_channels: in_channels_adjusted,
            out_channels: out_channels_adjusted,
            kernel_size,
        }
    }

    pub fn set_weight_norm(
        &mut self,
        weight_g: &Tensor<f32>,
        weight_v: &Tensor<f32>,
    ) -> Result<(), FerroError> {
        // ConvTranspose1d weight shape is [in_channels, out_channels/groups, kernel_size].
        // PyTorch `weight_norm(..., dim=0)` L2-normalizes over dims (1, 2).
        let v_shape = weight_v.shape().to_vec();
        if v_shape.is_empty() {
            return Err(FerroError::new(
                "ConvTranspose1d::set_weight_norm: weight_v has zero dims",
            ));
        }
        let in_c = v_shape[0];
        if in_c == 0 {
            return Err(FerroError::new(
                "ConvTranspose1d::set_weight_norm: weight_v dim 0 is zero",
            ));
        }
        let rest: usize = v_shape.iter().skip(1).product();
        if rest == 0 {
            return Err(FerroError::new(format!(
                "ConvTranspose1d::set_weight_norm: inner dims product to zero ({:?})",
                v_shape
            )));
        }

        let g_data = weight_g.data();
        if g_data.len() != in_c {
            return Err(FerroError::new(format!(
                "ConvTranspose1d::set_weight_norm: weight_g has {} elements, expected {} (in_channels)",
                g_data.len(),
                in_c
            )));
        }

        let v_data = weight_v.data();
        if v_data.len() != in_c * rest {
            return Err(FerroError::new(format!(
                "ConvTranspose1d::set_weight_norm: weight_v length {} != product of shape {:?}",
                v_data.len(),
                v_shape
            )));
        }

        let mut result = Vec::with_capacity(v_data.len());
        for ic in 0..in_c {
            let start = ic * rest;
            let end = start + rest;
            let slice = &v_data[start..end];
            let norm_sq: f32 = slice.iter().map(|&x| x * x).sum();
            let norm = norm_sq.sqrt().max(1e-12);
            let scale = g_data[ic] / norm;
            for &v in slice {
                result.push(v * scale);
            }
        }
        self.weight = Parameter::new(Tensor::from_data(result, v_shape));
        Ok(())
    }

    pub fn set_bias(&mut self, bias: &Tensor<f32>) -> Result<(), FerroError> {
        if bias.shape().len() != 1 {
            return Err(FerroError::new(format!(
                "ConvTranspose1d::set_bias: bias must be 1D, got shape {:?}",
                bias.shape()
            )));
        }
        if bias.shape()[0] != self.out_channels {
            return Err(FerroError::new(format!(
                "ConvTranspose1d::set_bias: bias length {} != out_channels {}",
                bias.shape()[0],
                self.out_channels
            )));
        }
        self.bias = Some(Parameter::new(bias.clone()));
        Ok(())
    }

    /// Matmul-based fast path for the `batch_size == 1 && groups == 1`
    /// case, which is what every `ConvTranspose1d` in Kokoro's Generator
    /// takes during inference.
    ///
    /// The math: a 1D transposed convolution is
    /// `out[oc, in_pos*stride + k - padding] += input[ic, in_pos] * weight[ic, oc, k]`.
    /// Matrix form:
    ///   1. Transpose `input` from `(C_in, L_in)` to `(L_in, C_in)`.
    ///   2. `y = x_t @ W_flat` where `W_flat = (C_in, C_out*K)` is just
    ///      the existing row-major weight storage viewed as 2D.
    ///      Result shape: `(L_in, C_out*K)`.
    ///   3. Scatter `y[in_pos, oc*K + k]` into
    ///      `out[oc, in_pos*stride + k - padding]` (bounds checked).
    ///   4. Add bias.
    fn forward_b1_g1_matmul(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight = self.weight.data();
        let weight_data = weight.data();
        let input_data = input.data();

        let in_shape = input.shape();
        debug_assert_eq!(in_shape[0], 1);
        debug_assert_eq!(self.groups, 1);
        debug_assert_eq!(in_shape[1], self.in_channels);

        let in_length = in_shape[2];
        let c_in = self.in_channels;
        let c_out = self.out_channels;
        let k_size = self.kernel_size;
        let stride = self.stride;
        let padding = self.padding;
        let output_padding = self.output_padding;

        // Output length: `(L_in - 1)*stride - 2*padding + K + output_padding`
        let out_length = ((in_length - 1) * stride).saturating_sub(2 * padding)
            + k_size
            + output_padding;

        // Step 1: transpose input (C_in, L_in) → x_t (L_in, C_in).
        // This is a bandwidth-bound pass over the input, cheap.
        let mut x_t = vec![0.0f32; in_length * c_in];
        for ic in 0..c_in {
            let in_off = ic * in_length;
            for il in 0..in_length {
                x_t[il * c_in + ic] = input_data[in_off + il];
            }
        }

        // Step 2: matmul x_t (L_in, C_in) @ W (C_in, C_out*K) = y (L_in, C_out*K).
        // W is already in (C_in, C_out*K) row-major because its storage is
        // (C_in, C_out, K) row-major and C_out*K is contiguous.
        let m_dim = c_out * k_size;
        let mut y = vec![0.0f32; in_length * m_dim];
        matmul_f32(&x_t, weight_data, &mut y, in_length, m_dim, c_in);

        // Step 3: scatter y into the output buffer.
        //   out[oc, in_pos*stride + k - padding] += y[in_pos, oc*K + k]
        // Per in_pos, pre-compute the valid kernel range to avoid a
        // per-k branch.
        let mut output_data = vec![0.0f32; c_out * out_length];
        let pad_isz = padding as isize;

        for in_pos in 0..in_length {
            let base = (in_pos * stride) as isize - pad_isz; // out_pos when k=0

            // k_min: smallest k with base + k >= 0
            let k_min: usize = if base >= 0 { 0 } else { (-base) as usize };
            // k_max: smallest k with base + k >= out_length
            let k_max_isz = out_length as isize - base;
            let k_max: usize = if k_max_isz <= 0 {
                0
            } else {
                (k_max_isz as usize).min(k_size)
            };

            if k_max <= k_min {
                continue;
            }

            let out_start = (base + k_min as isize) as usize;
            let span = k_max - k_min;
            let y_row_off = in_pos * m_dim;

            for oc in 0..c_out {
                let y_oc_off = y_row_off + oc * k_size + k_min;
                let out_off = oc * out_length + out_start;

                // Contiguous accumulating add. Auto-vectorisable.
                let out_slice = &mut output_data[out_off..out_off + span];
                let y_slice = &y[y_oc_off..y_oc_off + span];
                for kk in 0..span {
                    out_slice[kk] += y_slice[kk];
                }
            }
        }

        // Step 4: bias (broadcast across out_length).
        if let Some(ref b) = self.bias {
            let bias = b.data().data();
            debug_assert_eq!(bias.len(), c_out);
            for oc in 0..c_out {
                let off = oc * out_length;
                let bv = bias[oc];
                let row = &mut output_data[off..off + out_length];
                for v in row.iter_mut() {
                    *v += bv;
                }
            }
        }

        Tensor::from_data(output_data, vec![1, c_out, out_length])
    }

    /// Fallback direct-loop implementation for grouped or multi-batch
    /// transposed convolutions. Uses direct slice access (no `tensor[&[...]]`
    /// indexing overhead) with hoisted strides, but it's still `O(B·C_out·C_in·L_in·K)`.
    /// Not used on any Kokoro inference hot path.
    fn forward_direct(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        assert_eq!(
            input_shape.len(),
            3,
            "ConvTranspose1d: expected 3D input [batch, channels, length], got {:?}",
            input_shape
        );

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];
        assert_eq!(
            in_channels, self.in_channels,
            "ConvTranspose1d: input channels {} != configured {}",
            in_channels, self.in_channels
        );

        let c_out = self.out_channels;
        let groups = self.groups;
        let k_size = self.kernel_size;
        let stride = self.stride;
        let padding = self.padding;
        let output_padding = self.output_padding;

        let oc_per_group = c_out / groups;
        let ic_per_group = in_channels / groups;

        let out_length = ((in_length - 1) * stride).saturating_sub(2 * padding)
            + k_size
            + output_padding;

        let mut output_data = vec![0.0f32; batch_size * c_out * out_length];

        let input_data = input.data();
        let weight_data = self.weight.data().data();

        let in_b_stride = in_channels * in_length;
        let out_b_stride = c_out * out_length;
        let w_ic_stride = oc_per_group * k_size;
        let w_oc_stride = k_size;

        let pad_isz = padding as isize;

        for b in 0..batch_size {
            let in_b_off = b * in_b_stride;
            let out_b_off = b * out_b_stride;

            for g in 0..groups {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;

                for ic_rel in 0..ic_per_group {
                    let ic = ic_start + ic_rel;
                    let in_off = in_b_off + ic * in_length;
                    let w_ic_off = ic * w_ic_stride;

                    for in_pos in 0..in_length {
                        let x_val = input_data[in_off + in_pos];
                        let base = (in_pos * stride) as isize - pad_isz;

                        // Valid k range for this output position
                        let k_min: usize = if base >= 0 { 0 } else { (-base) as usize };
                        let k_max_isz = out_length as isize - base;
                        let k_max: usize = if k_max_isz <= 0 {
                            0
                        } else {
                            (k_max_isz as usize).min(k_size)
                        };

                        if k_max <= k_min {
                            continue;
                        }
                        let out_start = (base + k_min as isize) as usize;
                        let span = k_max - k_min;

                        for oc_rel in 0..oc_per_group {
                            let oc = oc_start + oc_rel;
                            let w_off = w_ic_off + oc_rel * w_oc_stride + k_min;
                            let out_off = out_b_off + oc * out_length + out_start;

                            let out_slice = &mut output_data[out_off..out_off + span];
                            let w_slice = &weight_data[w_off..w_off + span];
                            for kk in 0..span {
                                out_slice[kk] += x_val * w_slice[kk];
                            }
                        }
                    }
                }
            }
        }

        // Bias
        if let Some(ref bp) = self.bias {
            let bias = bp.data().data();
            for b in 0..batch_size {
                for oc in 0..c_out {
                    let off = b * out_b_stride + oc * out_length;
                    let bv = bias[oc];
                    let row = &mut output_data[off..off + out_length];
                    for v in row.iter_mut() {
                        *v += bv;
                    }
                }
            }
        }

        Tensor::from_data(output_data, vec![batch_size, c_out, out_length])
    }

    /// Perform transposed 1D convolution.
    ///
    /// Dispatches to `forward_b1_g1_matmul` for the common
    /// `batch_size == 1 && groups == 1` case (Kokoro Generator upsample
    /// blocks), and to `forward_direct` for anything else.
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        assert_eq!(
            shape.len(),
            3,
            "ConvTranspose1d: expected 3D input, got shape {:?}",
            shape
        );

        if shape[0] == 1 && self.groups == 1 && shape[1] == self.in_channels {
            self.forward_b1_g1_matmul(input)
        } else {
            self.forward_direct(input)
        }
    }
}

impl Forward for ConvTranspose1d {
    type Output = Tensor<f32>;

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        self.forward(input)
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for ConvTranspose1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let dot_g = format!("{}.weight_g", prefix);
        let dot_v = format!("{}.weight_v", prefix);
        let plain_w = format!("{}.weight", prefix);
        let dot_b = format!("{}.bias", prefix);

        let mut weight_loaded = false;
        if let (Ok(g), Ok(v)) = (
            loader.load_component_parameter(component, &dot_g),
            loader.load_component_parameter(component, &dot_v),
        ) {
            self.set_weight_norm(&g, &v)?;
            weight_loaded = true;
        }

        if !weight_loaded {
            match loader.load_component_parameter(component, &plain_w) {
                Ok(w) => {
                    self.weight = Parameter::new(w);
                    weight_loaded = true;
                }
                Err(e) => {
                    return Err(FerroError::new(format!(
                        "ConvTranspose1d::load_weights_binary: no weight_norm and no plain .weight for '{}.{}': {}",
                        component, prefix, e
                    )));
                }
            }
        }
        debug_assert!(weight_loaded);

        if let Ok(b) = loader.load_component_parameter(component, &dot_b) {
            // Accept biases shipped with extra singleton dims.
            let flat = b.data().to_vec();
            if flat.len() == self.out_channels {
                self.bias = Some(Parameter::new(Tensor::from_data(
                    flat,
                    vec![self.out_channels],
                )));
            } else {
                return Err(FerroError::new(format!(
                    "ConvTranspose1d::load_weights_binary: bias '{}.{}' has {} elements, expected {}",
                    component, dot_b, flat.len(), self.out_channels
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
    fn test_convtranspose1d_shape() {
        // Create a 1D transposed convolution with stride 2
        let conv_t = ConvTranspose1d::new(
            3, // in_channels
            6, // out_channels
            3, // kernel_size
            2, // stride
            1, // padding
            1, // output_padding
            1, // groups
            true, // bias
        );

        // Input: [batch_size=1, channels=3, length=5] (fast path)
        let input = Tensor::from_data(vec![0.1; 1 * 3 * 5], vec![1, 3, 5]);

        // Calculate expected output length:
        // (in_length - 1) * stride - 2 * padding + kernel_size + output_padding
        // (5 - 1) * 2 - 2*1 + 3 + 1 = 8 - 2 + 4 = 10
        let output = conv_t.forward(&input);

        assert_eq!(output.shape(), &[1, 6, 10]);
    }

    #[test]
    fn test_convtranspose1d_batch2() {
        // Batch > 1 goes through the direct fallback
        let conv_t = ConvTranspose1d::new(
            3, // in_channels
            6, // out_channels
            3, // kernel_size
            2, // stride
            1, // padding
            1, // output_padding
            1, // groups
            true, // bias
        );

        let input = Tensor::from_data(vec![0.1; 2 * 3 * 5], vec![2, 3, 5]);
        let output = conv_t.forward(&input);
        assert_eq!(output.shape(), &[2, 6, 10]);
    }

    #[test]
    fn test_convtranspose1d_grouped() {
        // groups > 1 goes through the direct fallback
        let conv_t = ConvTranspose1d::new(
            4, 8, 3, 2, 1, 0, 2, true,
        );
        let input = Tensor::from_data(vec![0.1; 1 * 4 * 4], vec![1, 4, 4]);
        let output = conv_t.forward(&input);
        // (4-1)*2 - 2*1 + 3 + 0 = 7
        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 8);
        assert_eq!(output.shape()[2], 7);
    }
}