//! Vocoder module for waveform generation

// Several helper methods, loop variables, and the `upsample_rates` field on
// `Generator` are kept as documentation / shape-validation aids even when the
// optimised forward path doesn't read them. Suppress dead-code and
// unused-variable warnings module-wide so the build log stays clean.
#![allow(dead_code, unused_variables)]

mod sinegen;
mod source_module;
mod adain_resblk1;
mod adain_resblk1d;

pub use sinegen::SineGen;
pub use source_module::SourceModuleHnNSF;
pub use adain_resblk1::{AdaINResBlock1, snake1d};
pub use adain_resblk1d::AdainResBlk1d;

use crate::{
    Forward,
    conv::Conv1d,
    conv_transpose::ConvTranspose1d,
};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use ferrocarril_dsp::stft::CustomSTFT;
use std::sync::Arc;

#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;


/// UpSample1d for conditional upsampling operations
#[derive(Clone, Copy)]
pub enum UpsampleType {
    None,
    Nearest,
}

pub struct UpSample1d {
    layer_type: UpsampleType,
}

impl UpSample1d {
    pub fn new(layer_type: UpsampleType) -> Self {
        Self { layer_type }
    }
    
    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        match self.layer_type {
            UpsampleType::None => x.clone(),
            UpsampleType::Nearest => {
                let (batch, channels, length) = (x.shape()[0], x.shape()[1], x.shape()[2]);
                let new_length = length * 2;
                let mut result = vec![0.0; batch * channels * new_length];
                
                for b in 0..batch {
                    for c in 0..channels {
                        for l in 0..length {
                            let src_idx = b * channels * length + c * length + l;
                            let dst_idx1 = b * channels * new_length + c * new_length + l * 2;
                            let dst_idx2 = b * channels * new_length + c * new_length + l * 2 + 1;
                            
                            result[dst_idx1] = x.data()[src_idx];
                            result[dst_idx2] = x.data()[src_idx];
                        }
                    }
                }
                
                Tensor::from_data(result, vec![batch, channels, new_length])
            }
        }
    }
}

/// Main Generator structure for converting features to audio
pub struct Generator {
    // Source path
    source: SourceModuleHnNSF,
    
    // Upsampling network
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<Arc<AdaINResBlock1>>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<Arc<AdaINResBlock1>>,
    
    // Final projection
    conv_post: Conv1d,
    stft: CustomSTFT,
    post_n_fft: usize,
    num_kernels: usize,
    upsample_rates: Vec<usize>,
    upsample_scales_prod: usize,
    gen_istft_n_fft: usize,
    gen_istft_hop_size: usize,
}

impl Generator {
    // Helper to concatenate tensors along channel dimension
    fn concat_channels(&self, tensors: &[&Tensor<f32>]) -> Tensor<f32> {
        assert!(!tensors.is_empty(), "Cannot concatenate empty tensor list");
        
        // Determine output size
        let (batch, _, time) = (tensors[0].shape()[0], tensors[0].shape()[1], tensors[0].shape()[2]);
        let total_channels: usize = tensors.iter().map(|t| t.shape()[1]).sum();
        
        let mut result = vec![0.0; batch * total_channels * time];
        let mut channel_offset = 0;
        
        for tensor in tensors {
            let channels = tensor.shape()[1];
            
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time {
                        let src_idx = b * channels * time + c * time + t;
                        let dst_idx = b * total_channels * time + (channel_offset + c) * time + t;
                        result[dst_idx] = tensor.data()[src_idx];
                    }
                }
            }
            
            channel_offset += channels;
        }
        
        Tensor::from_data(result, vec![batch, total_channels, time])
    }
    
    pub fn new(
        style_dim: usize,
        resblock_kernel_sizes: Vec<usize>,
        upsample_rates: Vec<usize>,
        upsample_initial_channel: usize,
        resblock_dilation_sizes: Vec<Vec<usize>>,
        upsample_kernel_sizes: Vec<usize>,
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
    ) -> Self {
        assert_eq!(upsample_rates.len(), upsample_kernel_sizes.len(), 
                "Must have same number of upsample rates and kernel sizes");
        
        let num_kernels = resblock_kernel_sizes.len();
        let upsample_scales_prod: usize = upsample_rates.iter().product();
        
        // Create source module
        let source = SourceModuleHnNSF::new(
            24000, // sampling_rate
            upsample_scales_prod * gen_istft_hop_size, // upsample_scale
            8, // harmonic_num
            0.1, // sine_amp
            0.003, // add_noise_std
            10.0, // voiced_threshold
        );
        
        
        // Create upsampling layers
        let mut ups = Vec::new();
        for i in 0..upsample_rates.len() {
            let stride = upsample_rates[i];
            let kernel_size = upsample_kernel_sizes[i];
            let padding = (kernel_size - stride) / 2;
            
            ups.push(ConvTranspose1d::new(
                upsample_initial_channel / (1 << i),
                upsample_initial_channel / (1 << (i + 1)),
                kernel_size,
                stride,
                padding,
                0, // output_padding
                1, // groups
                true, // bias
            ));
        }
        
        // Create resblocks
        let mut resblocks = Vec::new();
        for i in 0..upsample_rates.len() {
            let ch = upsample_initial_channel / (1 << (i + 1));
            for k in 0..num_kernels {
                resblocks.push(Arc::new(AdaINResBlock1::new(
                    ch,
                    resblock_kernel_sizes[k],
                    resblock_dilation_sizes[k].clone(),
                    style_dim,
                )));
            }
        }
        
        // Create noise convolutions and residual layers
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();
        
        for i in 0..upsample_rates.len() {
            let c_cur = upsample_initial_channel / (1 << (i + 1));
            
            if i + 1 < upsample_rates.len() {
                // Not the last layer
                let stride_f0 = upsample_rates[i + 1..].iter().product();
                let kernel_size = stride_f0 * 2;
                let padding = (stride_f0 + 1) / 2;
                
                noise_convs.push(Conv1d::new(
                    gen_istft_n_fft + 2, // input channels
                    c_cur,                // output channels
                    kernel_size,
                    stride_f0,           // stride
                    padding,
                    1,                   // dilation
                    1,                   // groups
                    true,                // bias
                ));
                
                noise_res.push(Arc::new(AdaINResBlock1::new(
                    c_cur,
                    7,                   // kernel size
                    vec![1, 3, 5],       // dilation
                    style_dim,
                )));
            } else {
                // Last layer
                noise_convs.push(Conv1d::new(
                    gen_istft_n_fft + 2, // input channels
                    c_cur,                // output channels
                    1,                   // kernel size
                    1,                   // stride
                    0,                   // padding
                    1,                   // dilation
                    1,                   // groups
                    true,                // bias
                ));
                
                noise_res.push(Arc::new(AdaINResBlock1::new(
                    c_cur,
                    11,                  // kernel size
                    vec![1, 3, 5],       // dilation
                    style_dim,
                )));
            }
        }
        
        // Create post convolution
        let conv_post = Conv1d::new(
            upsample_initial_channel / (1 << upsample_rates.len()),
            gen_istft_n_fft + 2,
            7,                   // kernel size
            1,                   // stride
            3,                   // padding
            1,                   // dilation
            1,                   // groups
            true,                // bias
        );
        
        // Create STFT
        let stft = CustomSTFT::new(
            gen_istft_n_fft,
            gen_istft_hop_size,
            gen_istft_n_fft,
        );
        
        Self {
            source,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            stft,
            post_n_fft: gen_istft_n_fft,
            num_kernels,
            upsample_rates,
            upsample_scales_prod,
            gen_istft_n_fft,
            gen_istft_hop_size,
        }
    }
    
    /// Upsample F0 by the appropriate scale factor
    fn upsample_f0(&self, f0: &Tensor<f32>) -> Tensor<f32> {
        // In Kokoro, F0 upsampling is done to match the audio sample rate needed by the source module
        // We need to upsample by the product of all upsample rates times the hop size

        // First, verify the input shape - should be [B, T]
        let (batch, time) = (f0.shape()[0], f0.shape()[1]);

        // Calculate the scale factor
        let hop_scale = self.upsample_scales_prod * self.gen_istft_hop_size;
        let new_time = time * hop_scale;

        // Create the upsampled tensor by repeating each value hop_scale times
        let mut result = vec![0.0; batch * new_time];

        for b in 0..batch {
            for t in 0..time {
                for i in 0..hop_scale {
                    let idx = b * new_time + t * hop_scale + i;
                    if idx < result.len() {
                        result[idx] = f0.data()[b * time + t];
                    }
                }
            }
        }
        
        // Add channel dimension to make it [B, 1, T] for source module
        let mut result_3d = vec![0.0; batch * 1 * new_time];
        for b in 0..batch {
            for t in 0..new_time {
                let src_idx = b * new_time + t;
                let dst_idx = b * 1 * new_time + 0 * new_time + t;
                
                if src_idx < result.len() && dst_idx < result_3d.len() {
                    result_3d[dst_idx] = result[src_idx];
                }
            }
        }
        
        Tensor::from_data(result_3d, vec![batch, 1, new_time])
    }
    
    /// Run the Generator forward up to and including `conv_post`, but
    /// stop before the `spec = exp(x[:n])`, `phase = sin(x[n:])` split
    /// and the iSTFT inverse. Returns the raw conv_post output of shape
    /// `[B, gen_istft_n_fft + 2, T]`. Used by integration tests to
    /// bisect Generator numerical drift against the
    /// `decoder_generator_conv_post.npy` kmodel fixture.
    pub fn forward_to_conv_post(
        &self,
        x: &Tensor<f32>,
        s: &Tensor<f32>,
        f0: &Tensor<f32>,
    ) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3, "forward_to_conv_post: x must be [B, C, T]");
        assert_eq!(s.shape().len(), 2, "forward_to_conv_post: s must be [B, S]");
        assert_eq!(f0.shape().len(), 2, "forward_to_conv_post: f0 must be [B, T]");

        let f0_upsampled_bct = self.upsample_f0(f0);
        let f0_upsampled = {
            let shape = f0_upsampled_bct.shape();
            let (b, _c, t_up) = (shape[0], shape[1], shape[2]);
            let mut transposed = vec![0.0f32; b * t_up];
            for batch in 0..b {
                for t in 0..t_up {
                    transposed[batch * t_up + t] = f0_upsampled_bct[&[batch, 0, t]];
                }
            }
            Tensor::from_data(transposed, vec![b, t_up, 1])
        };

        let (har_source, _noise_source, _uv) = self.source.forward(&f0_upsampled);
        let har_source_for_stft = if har_source.shape().len() == 3 {
            let (b, _c, t) = (har_source.shape()[0], har_source.shape()[1], har_source.shape()[2]);
            let mut data = vec![0.0f32; b * t];
            for batch in 0..b {
                for time in 0..t {
                    data[batch * t + time] = har_source[&[batch, 0, time]];
                }
            }
            Tensor::from_data(data, vec![b, t])
        } else {
            har_source.clone()
        };

        if har_source_for_stft.shape()[1] < self.gen_istft_n_fft {
            return Err(FerroError::new(format!(
                "forward_to_conv_post: har_source too short for STFT ({} < {})",
                har_source_for_stft.shape()[1],
                self.gen_istft_n_fft
            )));
        }

        let (har_spec, har_phase) = self.stft.transform(&har_source_for_stft);
        let (batch, freq_bins, frames) = (
            har_spec.shape()[0],
            har_spec.shape()[1],
            har_spec.shape()[2],
        );
        let mut har_data = vec![0.0f32; batch * (freq_bins * 2) * frames];
        for b in 0..batch {
            for f in 0..freq_bins {
                for t in 0..frames {
                    har_data[b * (freq_bins * 2) * frames + f * frames + t] = har_spec[&[b, f, t]];
                }
            }
        }
        for b in 0..batch {
            for f in 0..freq_bins {
                for t in 0..frames {
                    har_data[b * (freq_bins * 2) * frames + (f + freq_bins) * frames + t] =
                        har_phase[&[b, f, t]];
                }
            }
        }
        let har = Tensor::from_data(har_data, vec![batch, freq_bins * 2, frames]);

        let mut hidden = x.clone();
        for i in 0..self.ups.len() {
            let mut hidden_data = hidden.data().to_vec();
            for j in 0..hidden_data.len() {
                hidden_data[j] = if hidden_data[j] > 0.0 {
                    hidden_data[j]
                } else {
                    0.1 * hidden_data[j]
                };
            }
            hidden = Tensor::from_data(hidden_data, hidden.shape().to_vec());

            let x_source = self.noise_convs[i].forward(&har);
            let x_source = self.noise_res[i].forward(&x_source, s);

            hidden = self.ups[i].forward(&hidden);

            if i == self.ups.len() - 1 {
                let shape = hidden.shape().to_vec();
                let (b, c, t) = (shape[0], shape[1], shape[2]);
                let new_t = t + 1;
                let mut padded = vec![0.0f32; b * c * new_t];
                for bb in 0..b {
                    for cc in 0..c {
                        padded[bb * c * new_t + cc * new_t] = hidden[&[bb, cc, 1]];
                        for tt in 0..t {
                            padded[bb * c * new_t + cc * new_t + (tt + 1)] = hidden[&[bb, cc, tt]];
                        }
                    }
                }
                hidden = Tensor::from_data(padded, vec![b, c, new_t]);
            }

            if hidden.shape() == x_source.shape() {
                let mut combined_data = hidden.data().to_vec();
                for j in 0..combined_data.len() {
                    combined_data[j] += x_source.data()[j];
                }
                hidden = Tensor::from_data(combined_data, hidden.shape().to_vec());
            } else {
                return Err(FerroError::new(format!(
                    "forward_to_conv_post: hidden {:?} vs x_source {:?} shape mismatch",
                    hidden.shape(),
                    x_source.shape()
                )));
            }

            let mut xs: Option<Tensor<f32>> = None;
            for j in 0..self.num_kernels {
                let block_idx = i * self.num_kernels + j;
                let xj = self.resblocks[block_idx].forward(&hidden, s);
                if let Some(ref mut xs_val) = xs {
                    let mut xs_data = xs_val.data().to_vec();
                    let xj_data = xj.data();
                    for k in 0..xs_data.len() {
                        xs_data[k] += xj_data[k];
                    }
                    *xs_val = Tensor::from_data(xs_data, xs_val.shape().to_vec());
                } else {
                    xs = Some(xj);
                }
            }
            if let Some(xs_val) = xs {
                let mut xs_data = xs_val.data().to_vec();
                for j in 0..xs_data.len() {
                    xs_data[j] /= self.num_kernels as f32;
                }
                hidden = Tensor::from_data(xs_data, xs_val.shape().to_vec());
            }
        }

        let mut hidden_data = hidden.data().to_vec();
        for i in 0..hidden_data.len() {
            hidden_data[i] = if hidden_data[i] > 0.0 {
                hidden_data[i]
            } else {
                0.01 * hidden_data[i]
            };
        }
        hidden = Tensor::from_data(hidden_data, hidden.shape().to_vec());

        Ok(self.conv_post.forward(&hidden))
    }

    #[allow(unused_assignments)]
    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>, f0: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        let profile = std::env::var("FERRO_PROFILE").is_ok();
        let t_start = std::time::Instant::now();
        let mut t_mark = t_start;
        macro_rules! gstage {
            ($name:expr) => {
                if profile {
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[profile]   gen {:<34} {:>9.3} ms",
                        $name,
                        (now - t_mark).as_secs_f64() * 1000.0
                    );
                    t_mark = now;
                }
            };
        }

        // Verify input shapes
        if x.shape().len() != 3 {
            panic!("Generator input x must be 3-dimensional [B, C, T], got: {:?}", x.shape());
        }
        
        if s.shape().len() != 2 {
            panic!("Generator input s (style) must be 2-dimensional [B, S], got: {:?}", s.shape());
        }

        if f0.shape().len() != 2 {
            panic!("Generator input f0 must be 2-dimensional [B, T], got: {:?}", f0.shape());
        }
        
        // Extract batch size for consistency checks
        let batch_size = x.shape()[0];
        
        // Ensure batch dimensions match across all inputs
        if s.shape()[0] != batch_size {
            panic!("Style tensor batch dimension ({}) doesn't match input batch dimension ({})", 
                  s.shape()[0], batch_size);
        }
        
        if f0.shape()[0] != batch_size {
            panic!("F0 tensor batch dimension ({}) doesn't match input batch dimension ({})", 
                  f0.shape()[0], batch_size);
        }
        
        // Upsample F0 for source module
        let f0_upsampled_bct = self.upsample_f0(f0);

        // Transpose from [B, 1, T_up] to [B, T_up, 1] for SineGen
        let f0_upsampled = {
            let shape = f0_upsampled_bct.shape();
            assert_eq!(shape.len(), 3,
                "Generator: upsample_f0 must produce a 3D tensor, got {:?}", shape);
            let (b, c, t_up) = (shape[0], shape[1], shape[2]);
            assert_eq!(c, 1,
                "Generator: upsample_f0 must have channel dim 1, got shape {:?}", shape);
            let mut transposed = vec![0.0f32; b * t_up * 1];
            for batch in 0..b {
                for t in 0..t_up {
                    transposed[batch * t_up + t] = f0_upsampled_bct[&[batch, 0, t]];
                }
            }
            Tensor::from_data(transposed, vec![b, t_up, 1])
        };
        gstage!("upsample_f0 + transpose");

        // Generate source signals
        let (har_source, _noise_source, _uv) = self.source.forward(&f0_upsampled);
        gstage!("source.forward (SineGen)");

        // Ensure harmonic source has the right shape for STFT
        let har_source_for_stft = if har_source.shape().len() == 3 {
            // Convert [B, C, T] to [B, T] by taking only the first channel
            let (b, c, t) = (har_source.shape()[0], har_source.shape()[1], har_source.shape()[2]);
            
            let mut data = vec![0.0; b * t];
            for batch in 0..b {
                for time in 0..t {
                    data[batch * t + time] = har_source[&[batch, 0, time]];
                }
            }
            
            Tensor::from_data(data, vec![b, t])
        } else if har_source.shape().len() == 2 {
            // Already in [B, T] format
            har_source.clone()
        } else {
            panic!("Harmonic source has unexpected shape: {:?}", har_source.shape());
        };
        gstage!("har source shape prep");
        
        // Verify STFT input has sufficient samples - STRICT: NO SYNTHETIC FALLBACKS
        if har_source_for_stft.shape()[1] < self.gen_istft_n_fft {
            return Err(FerroError::new(format!(
                "CRITICAL TENSOR DIMENSION FAILURE: Harmonic source has insufficient samples for STFT ({} < {}). \
                This indicates upstream tensor dimension collapse in the pipeline. \
                NO SYNTHETIC CONTENT GENERATION ALLOWED.",
                har_source_for_stft.shape()[1], self.gen_istft_n_fft
            )));
        }
        
        // Apply STFT to get harmonic spectrogram
        let (har_spec, har_phase) = self.stft.transform(&har_source_for_stft);
        gstage!("STFT transform");
        if har_spec.shape() != har_phase.shape() {
            panic!("STFT output shape mismatch: spec {:?}, phase {:?}",
                   har_spec.shape(), har_phase.shape());
        }
        
        // Concatenate magnitude and phase along channel dimension
        let (batch, freq_bins, frames) = (har_spec.shape()[0], har_spec.shape()[1], har_spec.shape()[2]);
        let mut har_data = vec![0.0; batch * (freq_bins * 2) * frames];
        
        // First copy the magnitude spectrum
        for b in 0..batch {
            for f in 0..freq_bins {
                for t in 0..frames {
                    har_data[b * (freq_bins * 2) * frames + f * frames + t] = har_spec[&[b, f, t]];
                }
            }
        }
        
        // Then copy the phase
        for b in 0..batch {
            for f in 0..freq_bins {
                for t in 0..frames {
                    har_data[b * (freq_bins * 2) * frames + (f + freq_bins) * frames + t] = 
                        har_phase[&[b, f, t]];
                }
            }
        }
        
        let har = Tensor::from_data(har_data, vec![batch, freq_bins * 2, frames]);
        gstage!("har concat (spec + phase)");

        // Upsampling network - process the input through ups layers
        let mut hidden = x.clone();
        
        for i in 0..self.ups.len() {
            // Apply LeakyReLU (slope 0.1)
            let mut hidden_data = hidden.data().to_vec();
            for j in 0..hidden_data.len() {
                hidden_data[j] = if hidden_data[j] > 0.0 {
                    hidden_data[j]
                } else {
                    0.1 * hidden_data[j]
                };
            }
            hidden = Tensor::from_data(hidden_data, hidden.shape().to_vec());
            
            let x_source = self.noise_convs[i].forward(&har);
            let x_source = self.noise_res[i].forward(&x_source, s);
            gstage!(format!("stage {} leaky + noise_conv + noise_res", i).as_str());
            hidden = self.ups[i].forward(&hidden);

            // ReflectionPad1d((1, 0)) after the last transposed convolution
            if i == self.ups.len() - 1 {
                let shape = hidden.shape().to_vec();
                assert_eq!(
                    shape.len(),
                    3,
                    "Generator: reflection_pad input must be 3D, got {:?}",
                    shape
                );
                let (b, c, t) = (shape[0], shape[1], shape[2]);
                assert!(
                    t >= 2,
                    "Generator: reflection_pad needs at least 2 time frames, got {}",
                    t
                );
                let new_t = t + 1;
                let mut padded = vec![0.0f32; b * c * new_t];
                for bb in 0..b {
                    for cc in 0..c {
                        padded[bb * c * new_t + cc * new_t + 0] = hidden[&[bb, cc, 1]];
                        for tt in 0..t {
                            padded[bb * c * new_t + cc * new_t + (tt + 1)] =
                                hidden[&[bb, cc, tt]];
                        }
                    }
                }
                hidden = Tensor::from_data(padded, vec![b, c, new_t]);
            }

            // Verify hidden and x_source can be combined
            if hidden.shape()[0] != x_source.shape()[0] {
                panic!("Batch dimension mismatch after upsampling: hidden={}, x_source={}",
                      hidden.shape()[0], x_source.shape()[0]);
            }
            
            // Add noise source - requires matching dimensions
            if hidden.shape() == x_source.shape() {
                let mut combined_data = hidden.data().to_vec();
                for j in 0..combined_data.len() {
                    combined_data[j] += x_source.data()[j];
                }
                hidden = Tensor::from_data(combined_data, hidden.shape().to_vec());
            } else {
                panic!(
                    "Cannot combine hidden ({:?}) and x_source ({:?}) shapes don't match",
                    hidden.shape(), x_source.shape()
                );
            }
            gstage!(format!("stage {} ups + pad + combine", i).as_str());
            let mut xs: Option<Tensor<f32>> = None;
            
            for j in 0..self.num_kernels {
                let block_idx = i * self.num_kernels + j;
                let xj = self.resblocks[block_idx].forward(&hidden, s);
                
                if let Some(ref mut xs_val) = xs {
                    // Add to existing tensor
                    if xs_val.shape() != xj.shape() {
                        panic!("ResBlock output shape mismatch at {}: {:?} vs {:?}", 
                              block_idx, xs_val.shape(), xj.shape());
                    }
                    
                    let mut xs_data = xs_val.data().to_vec();
                    let xj_data = xj.data();
                    
                    for k in 0..xs_data.len() {
                        xs_data[k] += xj_data[k];
                    }
                    
                    *xs_val = Tensor::from_data(xs_data, xs_val.shape().to_vec());
                } else {
                    xs = Some(xj);
                }
            }
            gstage!(format!("stage {} resblocks ({}x)", i, self.num_kernels).as_str());
            
            // Average by num_kernels
            if let Some(xs_val) = xs {
                let mut xs_data = xs_val.data().to_vec();
                for j in 0..xs_data.len() {
                    xs_data[j] /= self.num_kernels as f32;
                }
                hidden = Tensor::from_data(xs_data, xs_val.shape().to_vec());
            }
            gstage!(format!("stage {} average", i).as_str());
        }
        
        // Apply final LeakyReLU and post-convolution
        let mut hidden_data = hidden.data().to_vec();
        for i in 0..hidden_data.len() {
            hidden_data[i] = if hidden_data[i] > 0.0 {
                hidden_data[i]
            } else {
                0.01 * hidden_data[i]
            };
        }
        hidden = Tensor::from_data(hidden_data, hidden.shape().to_vec());
        
        let x = self.conv_post.forward(&hidden);
        gstage!("final leaky + conv_post");
        let (batch, channels, time) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let n_fft_half = self.post_n_fft / 2 + 1;
        
        if channels < n_fft_half * 2 {
            panic!("Conv post output channels ({}) must be >= 2 * n_fft_half ({})",
                  channels, n_fft_half * 2);
        }
        
        let mut spec_data = vec![0.0; batch * n_fft_half * time];
        let mut phase_data = vec![0.0; batch * n_fft_half * time];
        
        for b in 0..batch {
            for f in 0..n_fft_half {
                for t in 0..time {
                    let x_idx = b * channels * time + f * time + t; 
                    let spec_idx = b * n_fft_half * time + f * time + t;
                    
                    if x_idx < x.data().len() {
                        spec_data[spec_idx] = f32::exp(x.data()[x_idx]);
                    }
                    
                    let phase_channel = n_fft_half + f;
                    if phase_channel < channels {
                        let x_phase_idx = b * channels * time + phase_channel * time + t;
                        
                        if x_phase_idx < x.data().len() {
                            phase_data[spec_idx] = f32::sin(x.data()[x_phase_idx]);
                        }
                    }
                }
            }
        }
        
        let spec_tensor = Tensor::from_data(spec_data, vec![batch, n_fft_half, time]);
        let phase_tensor = Tensor::from_data(phase_data, vec![batch, n_fft_half, time]);
        gstage!("spec/phase split (exp + sin)");
        
        // Generate waveform using iSTFT
        let audio = self.stft.inverse(&spec_tensor, &phase_tensor);
        gstage!("iSTFT inverse");

        if profile {
            let total = (std::time::Instant::now() - t_start).as_secs_f64() * 1000.0;
            eprintln!(
                "[profile]   gen {:<34} {:>9.3} ms",
                "TOTAL Generator::forward", total
            );
        }

        Ok(audio)
    }
}

/// Decoder for generating audio from asr, f0, and noise inputs
pub struct Decoder {
    pub encode: Arc<AdainResBlk1d>,
    pub decode: Vec<Arc<AdainResBlk1d>>,
    pub f0_conv: Conv1d,
    pub n_conv: Conv1d,
    pub asr_res: Conv1d,
    pub generator: Generator,
}

impl Decoder {
    pub fn new(
        dim_in: usize,
        style_dim: usize,
        _dim_out: usize,
        resblock_kernel_sizes: Vec<usize>,
        upsample_rates: Vec<usize>,
        upsample_initial_channel: usize,
        resblock_dilation_sizes: Vec<Vec<usize>>,
        upsample_kernel_sizes: Vec<usize>,
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
    ) -> Self {
        let encode = Arc::new(AdainResBlk1d::new(
            dim_in + 2,
            1024,
            style_dim,
            false,
            0.0,
        ));

        let mut decode: Vec<Arc<AdainResBlk1d>> = Vec::with_capacity(4);
        for _ in 0..3 {
            decode.push(Arc::new(AdainResBlk1d::new(
                1024 + 2 + 64,
                1024,
                style_dim,
                false,
                0.0,
            )));
        }
        decode.push(Arc::new(AdainResBlk1d::new(
            1024 + 2 + 64,
            512,
            style_dim,
            true,
            0.0,
        )));

        let f0_conv = Conv1d::new(1, 1, 3, 2, 1, 1, 1, true);
        let n_conv = Conv1d::new(1, 1, 3, 2, 1, 1, 1, true);

        let asr_res = Conv1d::new(dim_in, 64, 1, 1, 0, 1, 1, true);

        let generator = Generator::new(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        );

        Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator,
        }
    }

    #[allow(unused_assignments)]
    pub fn forward(
        &self,
        asr: &Tensor<f32>,
        f0_curve: &Tensor<f32>,
        n: &Tensor<f32>,
        s: &Tensor<f32>,
    ) -> Result<Tensor<f32>, FerroError> {
        let profile = std::env::var("FERRO_PROFILE").is_ok();
        let t_start = std::time::Instant::now();
        let mut t_mark = t_start;
        macro_rules! dstage {
            ($name:expr) => {
                if profile {
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[profile]  dec {:<34} {:>9.3} ms",
                        $name,
                        (now - t_mark).as_secs_f64() * 1000.0
                    );
                    t_mark = now;
                }
            };
        }

        assert_eq!(
            asr.shape().len(),
            3,
            "Decoder::forward: asr must be 3D [B, C, T], got shape {:?}",
            asr.shape()
        );
        assert_eq!(
            f0_curve.shape().len(),
            2,
            "Decoder::forward: f0_curve must be 2D [B, T], got shape {:?}",
            f0_curve.shape()
        );
        assert_eq!(
            n.shape().len(),
            2,
            "Decoder::forward: noise must be 2D [B, T], got shape {:?}",
            n.shape()
        );
        assert_eq!(
            s.shape().len(),
            2,
            "Decoder::forward: style must be 2D [B, style_dim], got shape {:?}",
            s.shape()
        );

        let batch = asr.shape()[0];
        let asr_time = asr.shape()[2];
        let t_curve = f0_curve.shape()[1];

        assert_eq!(
            f0_curve.shape()[0], batch,
            "Decoder::forward: batch mismatch between asr ({}) and f0_curve ({})",
            batch, f0_curve.shape()[0]
        );
        assert_eq!(
            n.shape()[0], batch,
            "Decoder::forward: batch mismatch between asr ({}) and noise ({})",
            batch, n.shape()[0]
        );
        assert_eq!(
            s.shape()[0], batch,
            "Decoder::forward: batch mismatch between asr ({}) and style ({})",
            batch, s.shape()[0]
        );
        assert_eq!(
            n.shape(),
            f0_curve.shape(),
            "Decoder::forward: noise shape {:?} must equal f0_curve shape {:?}",
            n.shape(),
            f0_curve.shape()
        );

        let mut f0_flat = vec![0.0f32; batch * t_curve];
        let mut n_flat = vec![0.0f32; batch * t_curve];
        for b in 0..batch {
            for t in 0..t_curve {
                f0_flat[b * t_curve + t] = f0_curve[&[b, t]];
                n_flat[b * t_curve + t] = n[&[b, t]];
            }
        }
        let f0_bct = Tensor::from_data(f0_flat, vec![batch, 1, t_curve]);
        let n_bct = Tensor::from_data(n_flat, vec![batch, 1, t_curve]);

        let f0 = self.f0_conv.forward(&f0_bct);
        let nn_ = self.n_conv.forward(&n_bct);
        dstage!("f0_conv + n_conv");

        assert_eq!(
            f0.shape(),
            nn_.shape(),
            "Decoder::forward: downsampled f0 {:?} != downsampled n {:?}",
            f0.shape(),
            nn_.shape()
        );
        assert_eq!(
            f0.shape()[2],
            asr_time,
            "Decoder::forward: downsampled f0 time {} != asr time {}. \
             This means the prosody predictor produced an f0_curve whose length \
             is not exactly 2 * asr_time. Upstream bug.",
            f0.shape()[2],
            asr_time
        );

        let x = self.concat_channels(&[asr, &f0, &nn_]);

        let mut x = self.encode.forward(&x, s);
        dstage!("concat + encode AdainResBlk1d");

        let asr_res = self.asr_res.forward(asr);
        dstage!("asr_res Conv1d");

        let mut res = true;
        for (i, block) in self.decode.iter().enumerate() {
            if res {
                x = self.concat_channels(&[&x, &asr_res, &f0, &nn_]);
            }
            x = block.forward(&x, s);
            if block.is_upsample() {
                res = false;
            }
            dstage!(format!("decode block {} ({})", i, if block.is_upsample() { "upsample" } else { "plain" }).as_str());
        }

        let audio = self.generator.forward(&x, s, f0_curve)?;
        dstage!("generator.forward");

        if profile {
            let total = (std::time::Instant::now() - t_start).as_secs_f64() * 1000.0;
            eprintln!(
                "[profile]  dec {:<34} {:>9.3} ms",
                "TOTAL Decoder::forward", total
            );
        }

        Ok(audio)
    }

    /// Concatenate 3D tensors along the channel dimension.
    fn concat_channels(&self, tensors: &[&Tensor<f32>]) -> Tensor<f32> {
        assert!(!tensors.is_empty(), "concat_channels: empty input list");

        let first_shape = tensors[0].shape();
        assert_eq!(
            first_shape.len(),
            3,
            "concat_channels: first tensor must be 3D [B, C, T], got {:?}",
            first_shape
        );
        let batch = first_shape[0];
        let time = first_shape[2];

        for (i, t) in tensors.iter().enumerate() {
            assert_eq!(
                t.shape().len(),
                3,
                "concat_channels: tensor[{}] must be 3D [B, C, T], got {:?}",
                i,
                t.shape()
            );
            assert_eq!(
                t.shape()[0], batch,
                "concat_channels: tensor[{}] batch {} != expected {} (shape {:?})",
                i, t.shape()[0], batch, t.shape()
            );
            assert_eq!(
                t.shape()[2], time,
                "concat_channels: tensor[{}] time {} != expected {} (shape {:?})",
                i, t.shape()[2], time, t.shape()
            );
        }

        let total_channels: usize = tensors.iter().map(|t| t.shape()[1]).sum();
        let mut result = vec![0.0f32; batch * total_channels * time];
        let mut channel_offset = 0usize;

        for tensor in tensors {
            let channels = tensor.shape()[1];
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time {
                        let src_idx = b * channels * time + c * time + t;
                        let dst_idx =
                            b * total_channels * time + (channel_offset + c) * time + t;
                        result[dst_idx] = tensor.data()[src_idx];
                    }
                }
            }
            channel_offset += channels;
        }

        Tensor::from_data(result, vec![batch, total_channels, time])
    }
}

// Fix the forward implementation for Decoder
impl Forward for Decoder {
    type Output = Tensor<f32>;
    
    fn forward(&self, _input: &Tensor<f32>) -> Self::Output {
        panic!("Decoder requires multiple inputs, use forward(asr, f0_curve, n, s) instead");
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for Generator {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        // Load source module. Python: `self.m_source = SourceModuleHnNSF(...)`
        let source_prefix = format!("{}.m_source", prefix);
        if let Err(e) = self.source.load_weights_binary(loader, component, &source_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load source module weights at '{}.{}': {} (continuing with default-init source)",
                component, source_prefix, e
            );
        }
        
        // Load upsampling blocks
        for (i, block) in self.ups.iter_mut().enumerate() {
            let ups_prefix = format!("{}.ups.{}", prefix, i);
            if let Err(e) = block.load_weights_binary(loader, component, &ups_prefix) {
                eprintln!(
                    "ferrocarril: warning: failed to load ups.{} weights: {}",
                    i, e
                );
            }
        }

        // Load resblocks
        for (i, block) in self.resblocks.iter_mut().enumerate() {
            // Need to get mutable reference inside Arc
            if let Some(block_ptr) = Arc::get_mut(block) {
                let block_prefix = format!("{}.resblocks.{}", prefix, i);
                if let Err(e) = block_ptr.load_weights_binary(loader, component, &block_prefix) {
                    eprintln!(
                        "ferrocarril: warning: failed to load resblocks.{} weights: {}",
                        i, e
                    );
                }
            } else {
                eprintln!(
                    "ferrocarril: warning: could not get mutable reference to resblock {}",
                    i
                );
            }
        }
        
        // Load noise convolutions
        for (i, conv) in self.noise_convs.iter_mut().enumerate() {
            let noise_conv_prefix = format!("{}.noise_convs.{}", prefix, i);
            if let Err(e) = conv.load_weights_binary(loader, component, &noise_conv_prefix) {
                eprintln!(
                    "ferrocarril: warning: failed to load noise_convs.{} weights: {}",
                    i, e
                );
            }
        }
        
        // Load noise residual blocks
        for (i, block) in self.noise_res.iter_mut().enumerate() {
            // Need to get mutable reference inside Arc
            if let Some(block_ptr) = Arc::get_mut(block) {
                let block_prefix = format!("{}.noise_res.{}", prefix, i);
                if let Err(e) = block_ptr.load_weights_binary(loader, component, &block_prefix) {
                    eprintln!(
                        "ferrocarril: warning: failed to load noise_res.{} weights: {}",
                        i, e
                    );
                }
            } else {
                eprintln!(
                    "ferrocarril: warning: could not get mutable reference to noise_res block {}",
                    i
                );
            }
        }
        
        // Load final projection (conv_post)
        let conv_post_prefix = format!("{}.conv_post", prefix);
        if let Err(e) = self.conv_post.load_weights_binary(loader, component, &conv_post_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load conv_post weights at '{}.{}': {}",
                component, conv_post_prefix, e
            );
        }
        
        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for Decoder {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        let generator_prefix = format!("{}.generator", prefix);
        if let Err(e) = self.generator.load_weights_binary(loader, component, &generator_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load generator weights: {}",
                e
            );
        }

        let encode_prefix = format!("{}.encode", prefix);
        let encode_ptr = Arc::get_mut(&mut self.encode).ok_or_else(|| {
            FerroError::new(format!(
                "Decoder::load_weights_binary: could not get mut ref to encode block for '{}.{}'",
                component, encode_prefix
            ))
        })?;
        encode_ptr
            .load_weights_binary(loader, component, &encode_prefix)
            .map_err(|e| {
                FerroError::new(format!(
                    "Decoder::load_weights_binary: failed to load encode block '{}.{}': {}",
                    component, encode_prefix, e
                ))
            })?;

        for (i, block) in self.decode.iter_mut().enumerate() {
            let decode_prefix = format!("{}.decode.{}", prefix, i);
            let block_ptr = Arc::get_mut(block).ok_or_else(|| {
                FerroError::new(format!(
                    "Decoder::load_weights_binary: could not get mut ref to decode block {} for '{}.{}'",
                    i, component, decode_prefix
                ))
            })?;
            block_ptr
                .load_weights_binary(loader, component, &decode_prefix)
                .map_err(|e| {
                    FerroError::new(format!(
                        "Decoder::load_weights_binary: failed to load decode block {} at '{}.{}': {}",
                        i, component, decode_prefix, e
                    ))
                })?;
        }

        let f0_conv_prefix = format!("{}.F0_conv", prefix);
        if let Err(e) = self.f0_conv.load_weights_binary(loader, component, &f0_conv_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load F0_conv weights: {}",
                e
            );
        }

        let n_conv_prefix = format!("{}.N_conv", prefix);
        if let Err(e) = self.n_conv.load_weights_binary(loader, component, &n_conv_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load N_conv weights: {}",
                e
            );
        }

        let asr_res_prefix = format!("{}.asr_res.0", prefix);
        if let Err(e) = self.asr_res.load_weights_binary(loader, component, &asr_res_prefix) {
            eprintln!(
                "ferrocarril: warning: failed to load asr_res weights: {}",
                e
            );
        }

        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for SourceModuleHnNSF {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        let linear_prefix = format!("{}.l_linear", prefix);
        self.linear_mut().load_weights_binary(loader, component, &linear_prefix)
            .map_err(|e| FerroError::new(format!(
                "SourceModuleHnNSF::load_weights_binary: failed to load l_linear at '{}.{}': {}",
                component, linear_prefix, e
            )))?;

        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for SineGen {
    fn load_weights_binary(
        &mut self,
        _loader: &BinaryWeightLoader,
        _component: &str,
        _prefix: &str
    ) -> Result<(), FerroError> {
        // SineGen has no learnable parameters
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrocarril_core::tensor::Tensor;
    
    #[test]
    fn test_upsample1d_none() {
        let upsample = UpSample1d::new(UpsampleType::None);
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let output = upsample.forward(&input);
        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.data(), input.data());
    }
    
    #[test]
    fn test_upsample1d_nearest() {
        let upsample = UpSample1d::new(UpsampleType::Nearest);
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let output = upsample.forward(&input);
        assert_eq!(output.shape(), &[1, 1, 8]);
        let expected = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        assert_eq!(output.data(), expected.as_slice());
    }
    
    #[test]
    fn test_generator_basic() {
        // Create a small Generator for testing
        let generator = Generator::new(
            64,                         // style_dim
            vec![3, 7, 11],             // resblock_kernel_sizes
            vec![4, 4],                 // upsample_rates
            256,                        // upsample_initial_channel
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]], // resblock_dilation_sizes
            vec![8, 8],                 // upsample_kernel_sizes
            16,                         // gen_istft_n_fft
            4,                          // gen_istft_hop_size
        );
        
        // Create test input tensors
        let batch_size = 1;
        let time_dim = 8;
        
        // Input features
        let x = Tensor::from_data(vec![0.1; batch_size * 256 * time_dim], vec![batch_size, 256, time_dim]);
        
        // Style vector
        let s = Tensor::from_data(vec![0.1; batch_size * 64], vec![batch_size, 64]);
        
        // F0 curve
        let f0 = Tensor::from_data(vec![440.0; batch_size * time_dim], vec![batch_size, time_dim]);
        
        // Forward pass
        let output = generator.forward(&x, &s, &f0).expect("Generator forward should succeed");
        
        // Check output shape (should be [B, T])
        assert_eq!(output.shape().len(), 2, "Output should be 2D");
        assert_eq!(output.shape()[0], batch_size, "Batch size should be preserved");
        assert!(output.shape()[1] > 0, "Output should have non-zero time dimension");
    }
    
    #[test]
    fn test_f0_upsampling() {
        let generator = Generator::new(
            64,                         // style_dim
            vec![3],                    // resblock_kernel_sizes
            vec![2],                    // upsample_rates
            128,                        // upsample_initial_channel
            vec![vec![1, 3, 5]],        // resblock_dilation_sizes
            vec![4],                    // upsample_kernel_sizes
            16,                         // gen_istft_n_fft
            4,                          // gen_istft_hop_size
        );
        
        let f0 = Tensor::from_data(vec![440.0, 880.0], vec![1, 2]);
        let upsampled = generator.upsample_f0(&f0);
        
        assert_eq!(upsampled.shape().len(), 3);
        assert_eq!(upsampled.shape()[0], 1); // batch
        assert_eq!(upsampled.shape()[1], 1); // channels
        assert_eq!(upsampled.shape()[2], 2 * 2 * 4); // original * rates * istft_hop
    }
}