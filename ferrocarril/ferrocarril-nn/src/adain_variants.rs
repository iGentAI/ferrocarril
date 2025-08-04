//! Specialized AdaIN implementations for different Kokoro TTS components
//! 
//! PyTorch Kokoro uses AdaIN in several distinct patterns:
//! - DecoderAdaIN: Variable input dimensions [1028,1024,2048,2180] → 128 style
//! - GeneratorAdaIN: 512 → 128 style for vocoder conditioning  
//! - DurationAdaIN: 1024 → 128 style for duration encoder conditioning

use crate::{Parameter, linear::Linear};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// AdaIN for Decoder components with variable input dimensions
/// 
/// Handles the different decoder normalization patterns:
/// - norm1: [1028, 128], [2180, 128] etc.
/// - norm2: [2048, 128], [1024, 128] etc.
#[derive(Debug)]
pub struct DecoderAdaIN {
    fc: Linear,              // style projection: input_dim → 2*channels
    channels: usize,         // target normalization channels  
    input_dim: usize,        // variable style input dimension
    eps: f32,
}

impl DecoderAdaIN {
    pub fn new(input_dim: usize, channels: usize) -> Self {
        Self {
            fc: Linear::new(128, input_dim * 2, true), // style_dim=128 → 2*channels  
            channels,
            input_dim,
            eps: 1e-5,
        }
    }
    
    /// Forward pass: Apply AdaIN with variable input dimensions
    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // x: [B, C, T] where C can be 1028, 2048, 2180, 1024, etc.
        // style: [B, 128]
        
        assert_eq!(x.shape().len(), 3, "Input must be 3D [batch, channels, time]");
        assert_eq!(style.shape().len(), 2, "Style must be 2D [batch, style_dim]");
        assert_eq!(style.shape()[1], 128, "Style dimension must be 128");
        assert_eq!(x.shape()[1], self.input_dim, "Input channels must match configured input_dim");
        
        let (batch_size, channels, time) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        
        // Project style to gamma and beta parameters
        let style_proj = self.fc.forward(style); // [B, 2*input_dim]
        
        // Split into gamma and beta
        let mut result = vec![0.0; batch_size * channels * time];
        
        for b in 0..batch_size {
            for c in 0..channels {
                // Calculate mean and variance for this channel across time
                let mut mean = 0.0;
                let mut var = 0.0;
                
                for t in 0..time {
                    let val = x[&[b, c, t]];
                    mean += val;
                    var += val * val;
                }
                
                mean /= time as f32;
                var = var / time as f32 - mean * mean;
                let std = (var + self.eps).sqrt();
                
                // Get gamma and beta for this channel from style projection
                let gamma = style_proj[&[b, c]];
                let beta = style_proj[&[b, c + channels]];
                
                // Apply AdaIN: (x - mean) / std * gamma + beta
                for t in 0..time {
                    let normalized = (x[&[b, c, t]] - mean) / std;
                    result[b * channels * time + c * time + t] = normalized * gamma + beta;
                }
            }
        }
        
        Tensor::from_data(result, vec![batch_size, channels, time])
    }
}

impl LoadWeightsBinary for DecoderAdaIN {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Load the fc Linear layer weights
        self.fc.load_weights_binary(loader, component, &format!("{}.fc", prefix))?;
        Ok(())
    }
}

/// AdaIN for Generator components (512 → 128 style conditioning)
#[derive(Debug)] 
pub struct GeneratorAdaIN {
    fc: Linear,              // 128 → 1024 (2*512)
    eps: f32,
}

impl GeneratorAdaIN {
    pub fn new() -> Self {
        Self {
            fc: Linear::new(128, 1024, true), // style_dim=128 → 2*512  
            eps: 1e-5,
        }
    }
    
    /// Forward pass optimized for generator usage (fixed 512 channels)
    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // x: [B, 512, T]  style: [B, 128]
        assert_eq!(x.shape().len(), 3, "Input must be 3D [batch, channels, time]");
        assert_eq!(x.shape()[1], 512, "Generator AdaIN expects 512 channels");
        assert_eq!(style.shape()[1], 128, "Style dimension must be 128");
        
        let (batch_size, _, time) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        
        // Project style to gamma and beta
        let style_proj = self.fc.forward(style); // [B, 1024]
        
        let mut result = vec![0.0; batch_size * 512 * time];
        
        for b in 0..batch_size {
            for c in 0..512 {
                // Calculate statistics for this channel
                let mut mean = 0.0;
                let mut var = 0.0;
                
                for t in 0..time {
                    let val = x[&[b, c, t]];
                    mean += val; 
                    var += val * val;
                }
                
                mean /= time as f32;
                var = var / time as f32 - mean * mean;
                let std = (var + self.eps).sqrt();
                
                // Extract gamma and beta for this channel
                let gamma = style_proj[&[b, c]];
                let beta = style_proj[&[b, c + 512]];
                
                // Apply AdaIN normalization
                for t in 0..time {
                    let normalized = (x[&[b, c, t]] - mean) / std;
                    result[b * 512 * time + c * time + t] = normalized * gamma + beta;
                }
            }
        }
        
        Tensor::from_data(result, vec![batch_size, 512, time])
    }
}

impl LoadWeightsBinary for GeneratorAdaIN {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        self.fc.load_weights_binary(loader, component, &format!("{}.fc", prefix))?;
        Ok(())
    }
}

/// AdaIN for DurationEncoder (AdaLayerNorm pattern)
#[derive(Debug)]
pub struct DurationAdaIN {
    fc: Linear,              // 128 → 1024 (2*512)
    eps: f32,
}

impl DurationAdaIN {
    pub fn new() -> Self {
        Self {
            fc: Linear::new(128, 1024, true), // style_dim=128 → 2*512
            eps: 1e-5,
        }
    }
    
    /// Forward pass for DurationEncoder AdaLayerNorm pattern
    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // x: [B, T, 512]  style: [B, 128] (different from decoder pattern!)
        assert_eq!(x.shape().len(), 3, "Input must be 3D [batch, time, channels]");
        assert_eq!(x.shape()[2], 512, "DurationAdaIN expects 512 channels");
        assert_eq!(style.shape()[1], 128, "Style dimension must be 128");
        
        let (batch_size, time, channels) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        
        // Project style
        let style_proj = self.fc.forward(style); // [B, 1024]
        
        let mut result = vec![0.0; batch_size * time * channels];
        
        // Apply layer normalization + style conditioning per timestep
        for b in 0..batch_size {
            for t in 0..time {
                // Calculate mean and variance across channels for this timestep
                let mut mean = 0.0;
                let mut var = 0.0;
                
                for c in 0..channels {
                    let val = x[&[b, t, c]];
                    mean += val;
                    var += val * val;
                }
                
                mean /= channels as f32;
                var = var / channels as f32 - mean * mean;
                let std = (var + self.eps).sqrt();
                
                // Apply normalization with style-specific gamma/beta for each channel
                for c in 0..channels {
                    let gamma = style_proj[&[b, c]];
                    let beta = style_proj[&[b, c + channels]];
                    
                    let normalized = (x[&[b, t, c]] - mean) / std;
                    result[b * time * channels + t * channels + c] = normalized * gamma + beta;
                }
            }
        }
        
        Tensor::from_data(result, vec![batch_size, time, channels])
    }
}

impl LoadWeightsBinary for DurationAdaIN {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        self.fc.load_weights_binary(loader, component, &format!("{}.fc", prefix))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decoder_adain() {
        let adain = DecoderAdaIN::new(1028, 1028);  // Variable input size example
        let input = Tensor::new(vec![1, 1028, 50]);
        let style = Tensor::new(vec![1, 128]);
        let output = adain.forward(&input, &style);
        assert_eq!(output.shape(), &[1, 1028, 50]);
    }
    
    #[test]
    fn test_generator_adain() {
        let adain = GeneratorAdaIN::new();
        let input = Tensor::new(vec![2, 512, 30]);
        let style = Tensor::new(vec![2, 128]);
        let output = adain.forward(&input, &style);
        assert_eq!(output.shape(), &[2, 512, 30]);
    }
    
    #[test]
    fn test_duration_adain() {
        let adain = DurationAdaIN::new();
        let input = Tensor::new(vec![1, 10, 512]);  // [B, T, C] format
        let style = Tensor::new(vec![1, 128]);
        let output = adain.forward(&input, &style);
        assert_eq!(output.shape(), &[1, 10, 512]);
    }
}