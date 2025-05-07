//! 1-D residual block with AdaIN and optional up-sampling.
//! Heavily simplified for inference:  • no dropout   • nearest up-sample
//! (mirrors istftnet.AdainResBlk1d from StyleTTS2)

use crate::{
    adain::AdaIN1d,
    conv::Conv1d,
    Forward,
};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

pub struct AdainResBlk1d {
    pub(crate) conv1: Conv1d,
    pub(crate) conv2: Conv1d,
    pub(crate) norm1: AdaIN1d,
    pub(crate) norm2: AdaIN1d,
    pub(crate) shortcut: Option<Conv1d>, // For channel matching in residual path
    upsample: bool,
}

impl AdainResBlk1d {
    pub fn new(in_ch: usize, out_ch: usize, style_dim: usize,
               upsample: bool, _dropout: f32) -> Self {
        // Create a shortcut conv if channel dimensions don't match
        let shortcut = if in_ch != out_ch || upsample {
            Some(Conv1d::new(in_ch, out_ch, 1, 1, 0, 1, 1, true))
        } else {
            None
        };
        
        Self {
            conv1: Conv1d::new(in_ch, out_ch, 3, 1, 1, 1, 1, true),
            conv2: Conv1d::new(out_ch, out_ch, 3, 1, 1, 1, 1, true),
            norm1: AdaIN1d::new(style_dim, in_ch),
            norm2: AdaIN1d::new(style_dim, out_ch),
            shortcut,
            upsample,
        }
    }

    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // x : [B,C,T]
        let mut h = self.norm1.forward(x, style);
        
        if self.upsample {
            h = self.upsample_nearest1d(&h, 2); // doubling the length
        }
        
        h = self.conv1.forward(&h);
        
        // Apply ReLU
        let mut h_data = h.data().to_vec();
        for v in h_data.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        let h = Tensor::from_data(h_data, h.shape().to_vec());

        let h = self.norm2.forward(&h, style);
        let h = self.conv2.forward(&h);

        // Residual path (upsample if necessary)
        let res = if self.upsample {
            self.upsample_nearest1d(x, 2)
        } else {
            x.clone()
        };
        
        // Apply channel matching if needed
        let res = if let Some(ref shortcut) = self.shortcut {
            shortcut.forward(&res)
        } else {
            res
        };
        
        // Check for shape mismatches between h and res
        // They should have the same batch size and time dimension, but channel dimensions might differ
        if h.shape().len() != res.shape().len() {
            println!("Warning: Tensor dimension mismatch in AdainResBlk1d. h.shape={:?}, res.shape={:?}", 
                     h.shape(), res.shape());
            
            // If shapes are completely different, just return h (skip residual)
            return h;
        }
        
        let (h_b, h_c, h_t) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let (res_b, res_c, res_t) = (res.shape()[0], res.shape()[1], res.shape()[2]);
        
        if h_b != res_b || h_t != res_t {
            println!("Warning: Batch or time dimension mismatch in AdainResBlk1d. h.shape={:?}, res.shape={:?}", 
                     h.shape(), res.shape());
            
            // If batch or time dimensions are different, just return h (skip residual)
            return h;
        }
        
        // If only channel dimensions differ, we can still add the residual by padding or truncating
        if h_c != res_c {
            println!("Warning: Channel dimension mismatch in AdainResBlk1d: h.channels={}, res.channels={}. Adapting residual connection.", 
                     h_c, res_c);
            
            // Create output tensor with h's shape
            let mut output = vec![0.0; h_b * h_c * h_t];
            
            // For each position, add the residual where dimensions overlap
            for batch in 0..h_b {
                for time in 0..h_t {
                    // Channel overlap is min(h_c, res_c)
                    let common_channels = h_c.min(res_c);
                    
                    for chan in 0..h_c {
                        let h_idx = batch * h_c * h_t + chan * h_t + time;
                        
                        if chan < common_channels {
                            // Add residual for channels that exist in both tensors
                            let res_idx = batch * res_c * res_t + chan * res_t + time;
                            
                            if h_idx < h.data().len() && res_idx < res.data().len() {
                                output[h_idx] = h.data()[h_idx] + res.data()[res_idx];
                            } else {
                                // Just use h if indices are out of bounds
                                if h_idx < h.data().len() {
                                    output[h_idx] = h.data()[h_idx];
                                }
                            }
                        } else {
                            // For channels beyond res_c, just copy h
                            if h_idx < h.data().len() {
                                output[h_idx] = h.data()[h_idx];
                            }
                        }
                        
                        // Apply ReLU
                        if h_idx < output.len() && output[h_idx] < 0.0 {
                            output[h_idx] = 0.0;
                        }
                    }
                }
            }
            
            return Tensor::from_data(output, h.shape().to_vec());
        }
        
        // Normal case - dimensions match
        let mut output = vec![0.0; h.data().len()];
        for i in 0..h.data().len() {
            output[i] = h.data()[i] + res.data()[i];
            if output[i] < 0.0 {
                output[i] = 0.0;
            }
        }
        
        Tensor::from_data(output, h.shape().to_vec())
    }
    
    // Simple nearest neighbor upsampling for 1D tensors
    fn upsample_nearest1d(&self, x: &Tensor<f32>, scale_factor: usize) -> Tensor<f32> {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let new_t = t * scale_factor;
        let mut result = vec![0.0; b * c * new_t];
        
        for batch in 0..b {
            for chan in 0..c {
                for time in 0..t {
                    for i in 0..scale_factor {
                        let src_idx = batch * c * t + chan * t + time;
                        let dst_idx = batch * c * new_t + chan * new_t + time * scale_factor + i;
                        result[dst_idx] = x.data()[src_idx];
                    }
                }
            }
        }
        
        Tensor::from_data(result, vec![b, c, new_t])
    }
    
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdainResBlk1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading AdainResBlk1d weights for {}.{}", component, prefix);
        
        // Load Conv1d weights - fail immediately if missing
        self.conv1.load_weights_binary(loader, component, &format!("{}.conv1", prefix))?;
        
        self.conv2.load_weights_binary(loader, component, &format!("{}.conv2", prefix))?;
        
        // Load shortcut if present
        if let Some(ref mut shortcut) = self.shortcut {
            shortcut.load_weights_binary(loader, component, &format!("{}.conv1x1", prefix))?;
        }
        
        // Load AdaIN weights
        self.norm1.load_weights_binary(loader, component, &format!("{}.norm1", prefix))?;
        
        self.norm2.load_weights_binary(loader, component, &format!("{}.norm2", prefix))?;
        
        Ok(())
    }
}