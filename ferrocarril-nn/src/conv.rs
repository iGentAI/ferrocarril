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
use ferrocarril_core::ops::matmul::matmul_f32;
use std::cell::RefCell;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

#[cfg(target_arch = "wasm32")]
mod _time_shim {
    #[derive(Copy, Clone)]
    pub struct Instant;
    impl Instant {
        pub fn now() -> Self { Instant }
        pub fn elapsed(&self) -> std::time::Duration { std::time::Duration::ZERO }
    }
    impl std::ops::Sub for Instant {
        type Output = std::time::Duration;
        fn sub(self, _other: Self) -> std::time::Duration { std::time::Duration::ZERO }
    }
}
#[cfg(target_arch = "wasm32")]
use _time_shim::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

thread_local! {
    /// Per-thread im2col scratch buffer reused across `Conv1d::conv1d_b1_g1_im2col`
    /// calls.
    static IM2COL_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());

    /// Accumulating per-phase nanosecond counters for
    /// `Conv1d::conv1d_b1_g1_im2col`.
    ///
    /// Tuple layout: (output_alloc_ns, im2col_ns, matmul_ns, bias_ns, n_calls).
    static CONV1D_STATS: RefCell<(u128, u128, u128, u128, u64)> =
        RefCell::new((0, 0, 0, 0, 0));
}

/// Reset the thread-local Conv1d phase counters.
pub fn reset_conv1d_stats() {
    CONV1D_STATS.with(|s| {
        *s.borrow_mut() = (0, 0, 0, 0, 0);
    });
}

/// Print and then clear the accumulated Conv1d phase counters.
pub fn dump_conv1d_stats() {
    CONV1D_STATS.with(|s| {
        let (alloc_ns, im2col_ns, matmul_ns, bias_ns, n_calls) = *s.borrow();
        if n_calls == 0 {
            return;
        }
        let total_ns = alloc_ns + im2col_ns + matmul_ns + bias_ns;
        let to_ms = |ns: u128| (ns as f64) / 1.0e6;
        eprintln!(
            "[profile] conv1d {:<32} {:>9} calls",
            "(b1 g1 im2col path)", n_calls
        );
        eprintln!(
            "[profile] conv1d {:<32} {:>9.3} ms",
            "  output alloc", to_ms(alloc_ns)
        );
        eprintln!(
            "[profile] conv1d {:<32} {:>9.3} ms",
            "  im2col_b1", to_ms(im2col_ns)
        );
        eprintln!(
            "[profile] conv1d {:<32} {:>9.3} ms",
            "  matmul_f32", to_ms(matmul_ns)
        );
        eprintln!(
            "[profile] conv1d {:<32} {:>9.3} ms",
            "  bias broadcast", to_ms(bias_ns)
        );
        eprintln!(
            "[profile] conv1d {:<32} {:>9.3} ms",
            "  TOTAL phases", to_ms(total_ns)
        );
        *s.borrow_mut() = (0, 0, 0, 0, 0);
    });
}

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

    fn conv1d_b1_g1_im2col(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight = self.weight.data();
        let weight_data = weight.data();
        let input_data = input.data();

        let in_shape = input.shape();
        debug_assert_eq!(in_shape[0], 1);
        debug_assert_eq!(self.groups, 1);

        let in_channels = in_shape[1];
        let in_length = in_shape[2];

        let kernel_size = weight.shape()[2];
        let out_channels = self.out_channels;
        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;

        let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
        let out_length = (in_length + 2 * padding - dilated_kernel_size) / stride + 1;

        let k_total = in_channels * kernel_size;
        let im2col_size = k_total * out_length;

        // Profile gate: read once per call, time each phase if set.
        let profile = std::env::var("FERRO_PROFILE").is_ok();

        let t_alloc_start = if profile {
            Some(Instant::now())
        } else {
            None
        };

        let mut output_data = vec![0.0f32; out_channels * out_length];

        let alloc_ns = t_alloc_start
            .map(|t| t.elapsed().as_nanos())
            .unwrap_or(0);

        let mut im2col_ns: u128 = 0;
        let mut matmul_ns: u128 = 0;

        IM2COL_BUFFER.with(|buf_cell| {
            let mut buf = buf_cell.borrow_mut();
            if buf.len() < im2col_size {
                buf.resize(im2col_size, 0.0);
            }
            let im2col = &mut buf[..im2col_size];

            let t_im2col = if profile {
                Some(Instant::now())
            } else {
                None
            };

            im2col_b1(
                input_data,
                im2col,
                in_channels,
                in_length,
                kernel_size,
                stride,
                padding,
                dilation,
                out_length,
            );

            if let Some(t) = t_im2col {
                im2col_ns = t.elapsed().as_nanos();
            }

            let t_matmul = if profile {
                Some(Instant::now())
            } else {
                None
            };

            matmul_f32(
                weight_data,
                im2col,
                &mut output_data,
                out_channels,
                out_length,
                k_total,
            );

            if let Some(t) = t_matmul {
                matmul_ns = t.elapsed().as_nanos();
            }
        });

        let t_bias_start = if profile {
            Some(Instant::now())
        } else {
            None
        };

        if let Some(b_param) = &self.bias {
            let bias = b_param.data().data();
            debug_assert_eq!(bias.len(), out_channels);
            for oc in 0..out_channels {
                let off = oc * out_length;
                let bv = bias[oc];
                let row = &mut output_data[off..off + out_length];
                for v in row.iter_mut() {
                    *v += bv;
                }
            }
        }

        let bias_ns = t_bias_start
            .map(|t| t.elapsed().as_nanos())
            .unwrap_or(0);

        if profile {
            CONV1D_STATS.with(|s| {
                let mut s = s.borrow_mut();
                s.0 += alloc_ns;
                s.1 += im2col_ns;
                s.2 += matmul_ns;
                s.3 += bias_ns;
                s.4 += 1;
            });
        }

        Tensor::from_data(output_data, vec![1, out_channels, out_length])
    }

    fn conv1d(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight = self.weight.data();
        let input_shape = input.shape();
        let weight_shape = weight.shape();

        assert_eq!(input_shape.len(), 3, "Input must be 3D: [batch_size, channels, length]");
        let (batch_size, in_channels, in_length) =
            (input_shape[0], input_shape[1], input_shape[2]);

        assert_eq!(in_channels, self.in_channels, "Input channels mismatch");

        if batch_size == 1 && self.groups == 1 {
            return self.conv1d_b1_g1_im2col(input);
        }

        let kernel_size = weight_shape[2];
        let out_channels = self.out_channels;
        let groups = self.groups;
        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;

        let dilated_kernel_size = (kernel_size - 1) * dilation + 1;
        let out_length = (in_length + 2 * padding - dilated_kernel_size) / stride + 1;

        let mut output_data = vec![0.0f32; batch_size * out_channels * out_length];

        let input_data: &[f32] = input.data();
        let weight_data: &[f32] = weight.data();
        let bias_slice: Option<&[f32]> =
            self.bias.as_ref().map(|b| b.data().data());

        let oc_per_group = out_channels / groups;
        let ic_per_group = in_channels / groups;
        let in_b_stride = in_channels * in_length;
        let out_b_stride = out_channels * out_length;
        let w_oc_stride = ic_per_group * kernel_size;

        let stride_isz = stride as isize;
        let dil_isz = dilation as isize;
        let pad_isz = padding as isize;
        let in_len_isz = in_length as isize;

        for b in 0..batch_size {
            let in_b_off = b * in_b_stride;
            let out_b_off = b * out_b_stride;

            for oc in 0..out_channels {
                let group_id = oc / oc_per_group;
                let in_c_start = group_id * ic_per_group;
                let in_c_end = in_c_start + ic_per_group;
                let w_oc_off = oc * w_oc_stride;
                let bias_val = bias_slice.map_or(0.0f32, |bb| bb[oc]);
                let out_co_off = out_b_off + oc * out_length;

                for ol in 0..out_length {
                    let il_base = ol as isize * stride_isz - pad_isz;

                    let k_min: usize = if il_base >= 0 {
                        0
                    } else {
                        ((-il_base) as usize + dilation - 1) / dilation
                    };
                    let k_max: usize = {
                        let need = in_len_isz - il_base;
                        if need <= 0 {
                            0
                        } else {
                            (((need + dil_isz - 1) / dil_isz) as usize).min(kernel_size)
                        }
                    };

                    if k_max <= k_min {
                        output_data[out_co_off + ol] = bias_val;
                        continue;
                    }

                    let mut sum = bias_val;

                    if dilation == 1 {
                        let il_start = (il_base + k_min as isize) as usize;
                        let span = k_max - k_min;

                        for ic in in_c_start..in_c_end {
                            let ic_rel = ic - in_c_start;
                            let in_off = in_b_off + ic * in_length;
                            let w_off = w_oc_off + ic_rel * kernel_size + k_min;

                            let in_slice = &input_data[in_off + il_start..in_off + il_start + span];
                            let w_slice = &weight_data[w_off..w_off + span];
                            for kk in 0..span {
                                sum += in_slice[kk] * w_slice[kk];
                            }
                        }
                    } else {
                        for ic in in_c_start..in_c_end {
                            let ic_rel = ic - in_c_start;
                            let in_off = in_b_off + ic * in_length;
                            let w_off = w_oc_off + ic_rel * kernel_size;

                            for k in k_min..k_max {
                                let il = (il_base + k as isize * dil_isz) as usize;
                                sum += input_data[in_off + il] * weight_data[w_off + k];
                            }
                        }
                    }

                    output_data[out_co_off + ol] = sum;
                }
            }
        }

        Tensor::from_data(
            output_data,
            vec![batch_size, out_channels, out_length],
        )
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

fn im2col_b1(
    input: &[f32],
    output: &mut [f32],
    in_channels: usize,
    in_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_length: usize,
) {
    debug_assert_eq!(input.len(), in_channels * in_length);
    debug_assert_eq!(output.len(), in_channels * kernel_size * out_length);

    let stride_isz = stride as isize;
    let dil_isz = dilation as isize;
    let pad_isz = padding as isize;
    let in_len_isz = in_length as isize;

    for ic in 0..in_channels {
        let in_off = ic * in_length;
        for k in 0..kernel_size {
            let row = ic * kernel_size + k;
            let row_off = row * out_length;
            let k_dil = k as isize * dil_isz;

            let pad_minus_kd = pad_isz - k_dil;
            let ol_min: usize = if pad_minus_kd <= 0 {
                0
            } else {
                (((pad_minus_kd + stride_isz - 1) / stride_isz) as usize).min(out_length)
            };

            let need = in_len_isz + pad_isz - k_dil;
            let ol_max: usize = if need <= 0 {
                0
            } else {
                (((need + stride_isz - 1) / stride_isz) as usize).min(out_length)
            };

            if ol_max <= ol_min {
                output[row_off..row_off + out_length].fill(0.0);
                continue;
            }

            if ol_min > 0 {
                output[row_off..row_off + ol_min].fill(0.0);
            }

            if stride == 1 {
                let il_start = (ol_min as isize * stride_isz + k_dil - pad_isz) as usize;
                let span = ol_max - ol_min;
                let dst = &mut output[row_off + ol_min..row_off + ol_min + span];
                let src = &input[in_off + il_start..in_off + il_start + span];
                dst.copy_from_slice(src);
            } else {
                let dst = &mut output[row_off + ol_min..row_off + ol_max];
                for (idx, dst_val) in dst.iter_mut().enumerate() {
                    let ol = ol_min + idx;
                    let il = (ol as isize * stride_isz + k_dil - pad_isz) as usize;
                    *dst_val = input[in_off + il];
                }
            }

            if ol_max < out_length {
                output[row_off + ol_max..row_off + out_length].fill(0.0);
            }
        }
    }
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