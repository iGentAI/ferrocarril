//! AdaINResBlock1 - Residual Block with AdaIN and Snake activation.
//!
//! Faithful port of `kokoro/istftnet.py::AdaINResBlock1`. This is the
//! Generator's residual block, used by both `Generator.resblocks` and
//! `Generator.noise_res`. Unlike the Decoder's `AdainResBlk1d`, this
//! block has NO upsampling, NO `conv1x1` shortcut, and NO `* rsqrt(2)`
//! output scaling — it is a pure 3-branch accumulating residual:
//!
//! ```python
//! for c1, c2, n1, n2, a1, a2 in zip(...):
//!     xt = n1(x, s)
//!     xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
//!     xt = c1(xt)
//!     xt = n2(xt, s)
//!     xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
//!     xt = c2(xt)
//!     x = xt + x
//! return x
//! ```

use crate::{Parameter, Forward, conv::Conv1d, adain::AdaIN1d};
use ferrocarril_core::tensor::Tensor;

#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;
#[cfg(feature = "weights")]
use ferrocarril_core::FerroError;

/// Snake1D activation function: `x + (1/alpha) * sin(alpha*x)^2`.
#[inline]
pub fn snake1d(x: f32, alpha: f32) -> f32 {
    x + (1.0 / alpha) * ((alpha * x).sin().powi(2))
}

/// Generator-side residual block with AdaIN + Snake1D, no shortcut.
pub struct AdaINResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Parameter>,
    alpha2: Vec<Parameter>,
}

impl AdaINResBlock1 {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilation: Vec<usize>,
        style_dim: usize,
    ) -> Self {
        assert_eq!(
            dilation.len(),
            3,
            "AdaINResBlock1 requires exactly 3 dilation values (got {})",
            dilation.len()
        );

        let mut convs1 = Vec::with_capacity(dilation.len());
        let mut convs2 = Vec::with_capacity(dilation.len());
        let mut adain1 = Vec::with_capacity(dilation.len());
        let mut adain2 = Vec::with_capacity(dilation.len());
        let mut alpha1 = Vec::with_capacity(dilation.len());
        let mut alpha2 = Vec::with_capacity(dilation.len());

        for &d in &dilation {
            // convs1[i]: dilated conv with padding = get_padding(k, d)
            convs1.push(Conv1d::new(
                channels,
                channels,
                kernel_size,
                1,                      // stride
                get_padding(kernel_size, d), // padding
                d,                      // dilation
                1,                      // groups
                true,                   // bias (weight_norm outer, but Rust Conv1d has bias slot)
            ));
            // convs2[i]: dilation=1, padding = get_padding(k, 1)
            convs2.push(Conv1d::new(
                channels,
                channels,
                kernel_size,
                1,
                get_padding(kernel_size, 1),
                1,
                1,
                true,
            ));
            adain1.push(AdaIN1d::new(style_dim, channels));
            adain2.push(AdaIN1d::new(style_dim, channels));
            // alpha params: Python initialises `torch.ones(1, channels, 1)`
            // but weight loading will overwrite this.
            alpha1.push(Parameter::new(Tensor::from_data(vec![1.0; channels], vec![channels])));
            alpha2.push(Parameter::new(Tensor::from_data(vec![1.0; channels], vec![channels])));
        }

        Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        }
    }

    /// Apply Snake1D activation per channel, reading alpha from the
    /// parameter's flat buffer. The alpha param may be stored as a 1-D
    /// `[C]` tensor or a 3-D `[1, C, 1]` tensor in the real Kokoro
    /// weights; in either case the underlying contiguous data is
    /// C-length and can be read flatly.
    fn apply_snake_in_place(data: &mut [f32], shape: &[usize], alpha_param: &Parameter) {
        assert_eq!(
            shape.len(),
            3,
            "AdaINResBlock1::apply_snake: expected 3D [B,C,T], got {:?}",
            shape
        );
        let (b, c, t) = (shape[0], shape[1], shape[2]);
        let alpha = alpha_param.data().data();
        assert_eq!(
            alpha.len(),
            c,
            "AdaINResBlock1::apply_snake: alpha has {} elements, expected {} (channels)",
            alpha.len(),
            c
        );
        for bb in 0..b {
            for cc in 0..c {
                let a = alpha[cc];
                let inv_a = 1.0 / a;
                for tt in 0..t {
                    let idx = bb * c * t + cc * t + tt;
                    let v = data[idx];
                    // x + (1/a) * sin(a*x)^2
                    let s = (a * v).sin();
                    data[idx] = v + inv_a * s * s;
                }
            }
        }
    }

    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        // Python:
        //   for c1, c2, n1, n2, a1, a2 in zip(...):
        //       xt = n1(x, s)
        //       xt = xt + (1 / a1) * (sin(a1 * xt) ** 2)
        //       xt = c1(xt)
        //       xt = n2(xt, s)
        //       xt = xt + (1 / a2) * (sin(a2 * xt) ** 2)
        //       xt = c2(xt)
        //       x = xt + x              # <-- accumulating residual
        //   return x
        //
        // Previously the Rust version set `result = xt` at the end of
        // each iteration, throwing away the original x and turning the
        // block into a single-branch chain instead of a 3-branch
        // accumulator. That caused the Generator output RMS to be
        // ~5x smaller than Python's.
        let mut x_acc = x.clone();

        for i in 0..self.convs1.len() {
            // --- branch i ---
            let xt = self.adain1[i].forward(&x_acc, s);

            let shape = xt.shape().to_vec();
            let mut data = xt.data().to_vec();
            Self::apply_snake_in_place(&mut data, &shape, &self.alpha1[i]);
            let xt = Tensor::from_data(data, shape);

            let xt = self.convs1[i].forward(&xt);

            let xt = self.adain2[i].forward(&xt, s);

            let shape = xt.shape().to_vec();
            let mut data = xt.data().to_vec();
            Self::apply_snake_in_place(&mut data, &shape, &self.alpha2[i]);
            let xt = Tensor::from_data(data, shape);

            let xt = self.convs2[i].forward(&xt);

            // --- accumulate: x = xt + x ---
            assert_eq!(
                xt.shape(),
                x_acc.shape(),
                "AdaINResBlock1: branch {} output shape {:?} != accumulator shape {:?}",
                i,
                xt.shape(),
                x_acc.shape()
            );
            let mut out = x_acc.data().to_vec();
            let xt_data = xt.data();
            for k in 0..out.len() {
                out[k] += xt_data[k];
            }
            x_acc = Tensor::from_data(out, x_acc.shape().to_vec());
        }

        x_acc
    }
}

/// Helper function to calculate padding based on kernel size and dilation.
/// Matches Python: `int((kernel_size * dilation - dilation) / 2)`.
fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdaINResBlock1 {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Load convolution layers
        for (i, conv) in self.convs1.iter_mut().enumerate() {
            let conv_prefix = format!("{}.convs1.{}", prefix, i);
            conv.load_weights_binary(loader, component, &conv_prefix).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load convs1.{} at '{}.{}': {}",
                    i, component, conv_prefix, e
                ))
            })?;
        }
        for (i, conv) in self.convs2.iter_mut().enumerate() {
            let conv_prefix = format!("{}.convs2.{}", prefix, i);
            conv.load_weights_binary(loader, component, &conv_prefix).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load convs2.{} at '{}.{}': {}",
                    i, component, conv_prefix, e
                ))
            })?;
        }

        // Load AdaIN layers
        for (i, adain) in self.adain1.iter_mut().enumerate() {
            let adain_prefix = format!("{}.adain1.{}", prefix, i);
            adain.load_weights_binary(loader, component, &adain_prefix).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load adain1.{} at '{}.{}': {}",
                    i, component, adain_prefix, e
                ))
            })?;
        }
        for (i, adain) in self.adain2.iter_mut().enumerate() {
            let adain_prefix = format!("{}.adain2.{}", prefix, i);
            adain.load_weights_binary(loader, component, &adain_prefix).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load adain2.{} at '{}.{}': {}",
                    i, component, adain_prefix, e
                ))
            })?;
        }

        // Load alpha parameters. Stored as `alpha1.i` and `alpha2.i` in
        // the Kokoro checkpoint. The raw tensor shape is `[1, C, 1]` but
        // its flat buffer is C-long, matching our stored representation.
        for (i, alpha) in self.alpha1.iter_mut().enumerate() {
            let alpha_path = format!("{}.alpha1.{}", prefix, i);
            let tensor = loader.load_component_parameter(component, &alpha_path).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load alpha1.{} at '{}.{}': {}",
                    i, component, alpha_path, e
                ))
            })?;
            *alpha = Parameter::new(tensor);
        }
        for (i, alpha) in self.alpha2.iter_mut().enumerate() {
            let alpha_path = format!("{}.alpha2.{}", prefix, i);
            let tensor = loader.load_component_parameter(component, &alpha_path).map_err(|e| {
                FerroError::new(format!(
                    "AdaINResBlock1: failed to load alpha2.{} at '{}.{}': {}",
                    i, component, alpha_path, e
                ))
            })?;
            *alpha = Parameter::new(tensor);
        }

        Ok(())
    }
}