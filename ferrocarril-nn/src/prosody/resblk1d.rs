//! AdaINResBlk1d - EXACT PyTorch Implementation
//! ZERO TOLERANCE for silent fallbacks or adaptations
//! From pytorch_adain_analysis: norm1 → LeakyReLU → pool → conv1 → norm2 → LeakyReLU → conv2

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

/// LeakyReLU activation function - EXACT PyTorch pattern
/// PyTorch: nn.LeakyReLU(0.2) - negative_slope=0.2
#[inline]
fn leaky_relu_activation(x: f32, negative_slope: f32) -> f32 {
    if x > 0.0 { x } else { negative_slope * x }
}

/// UpSample1d for upsampling operations - EXACT PyTorch pattern
pub struct UpSample1d {
    upsample_factor: usize,
}

impl UpSample1d {
    pub fn new(upsample_factor: usize) -> Self {
        Self { upsample_factor }
    }
    
    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        if self.upsample_factor <= 1 {
            return x.clone();
        }
        
        let (batch, channels, time) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let new_time = time * self.upsample_factor;
        let mut result = vec![0.0; batch * channels * new_time];
        
        for b in 0..batch {
            for c in 0..channels {
                for t in 0..time {
                    let src_val = x[&[b, c, t]];
                    for i in 0..self.upsample_factor {
                        let dst_idx = b * channels * new_time + c * new_time + t * self.upsample_factor + i;
                        result[dst_idx] = src_val;
                    }
                }
            }
        }
        
        Tensor::from_data(result, vec![batch, channels, new_time])
    }
}

/// AdaINResBlk1d - EXACT PyTorch Implementation 
/// NO SILENT FALLBACKS - ALL ASSERTIONS ENFORCED
pub struct AdainResBlk1d {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: AdaIN1d,
    norm2: AdaIN1d,
    conv1x1: Option<Conv1d>,  // For learned shortcut
    upsample: Option<UpSample1d>,
    learned_sc: bool,
    dim_in: usize,
    dim_out: usize,
    negative_slope: f32,  // LeakyReLU negative slope (0.2 in PyTorch)
}

impl AdainResBlk1d {
    pub fn new(dim_in: usize, dim_out: usize, style_dim: usize, upsample: bool, _dropout: f32) -> Self {
        let learned_sc = (dim_in != dim_out) || upsample;
        
        let conv1 = Conv1d::new(dim_in, dim_out, 3, 1, 1, 1, 1, true);
        let conv2 = Conv1d::new(dim_out, dim_out, 3, 1, 1, 1, 1, true);
        let norm1 = AdaIN1d::new(dim_in, style_dim);
        let norm2 = AdaIN1d::new(dim_out, style_dim);
        
        let conv1x1 = if learned_sc {
            Some(Conv1d::new(dim_in, dim_out, 1, 1, 0, 1, 1, false))
        } else {
            None
        };
        
        let upsample = if upsample {
            Some(UpSample1d::new(2))  // 2x upsampling like PyTorch
        } else {
            None
        };
        
        Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            upsample,
            learned_sc,
            dim_in,
            dim_out,
            negative_slope: 0.2,  // PyTorch LeakyReLU negative slope
        }
    }
    
    /// EXACT PyTorch forward pattern - NO SILENT FALLBACKS
    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Validate input dimensions - NO TOLERANCE for mismatches
        assert_eq!(x.shape().len(), 3, 
            "STRICT: Input must be 3D [B, C, T], got: {:?}", x.shape());
        assert_eq!(s.shape().len(), 2, 
            "STRICT: Style must be 2D [B, style_dim], got: {:?}", s.shape());
        assert_eq!(x.shape()[0], s.shape()[0], 
            "STRICT: Batch dimensions must match exactly");
            
        // Process residual path - EXACT PyTorch pattern
        let residual_out = self._residual(x, s);
        
        // Process shortcut path - EXACT PyTorch pattern
        let shortcut_out = self._shortcut(x);
        
        // STRICT: Enforce exact shape matching - NO SILENT ADAPTATIONS
        assert_eq!(residual_out.shape(), shortcut_out.shape(),
            "CRITICAL ARCHITECTURAL ERROR: Residual {:?} and shortcut {:?} shapes MUST match exactly. \
            NO ADAPTATIONS ALLOWED - FIX THE ARCHITECTURE.",
            residual_out.shape(), shortcut_out.shape());
        
        // Combine paths with normalization - EXACT PyTorch pattern
        let norm_factor = 1.0 / (2.0f32).sqrt(); // torch.rsqrt(torch.tensor(2))
        let mut result_data = vec![0.0; residual_out.data().len()];
        
        for i in 0..residual_out.data().len() {
            result_data[i] = (residual_out.data()[i] + shortcut_out.data()[i]) * norm_factor;
        }
        
        Tensor::from_data(result_data, residual_out.shape().to_vec())
    }
    
    /// EXACT PyTorch _residual implementation
    fn _residual(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        // Step 1: norm1(x, s) - First AdaIN conditioning
        let mut h = self.norm1.forward(x, s);
        
        // Step 2: actv(x) - LeakyReLU activation
        let (batch, channels, time) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let mut h_data = h.data().to_vec();
        
        for i in 0..h_data.len() {
            h_data[i] = leaky_relu_activation(h_data[i], self.negative_slope);
        }
        h = Tensor::from_data(h_data, vec![batch, channels, time]);
        
        // Step 3: pool(x) - Apply upsampling if enabled
        if let Some(ref pool) = self.upsample {
            h = pool.forward(&h);
        }
        
        // Step 4: conv1(dropout(x)) - First convolution
        h = self.conv1.forward(&h);
        
        // Step 5: norm2(x, s) - Second AdaIN conditioning
        h = self.norm2.forward(&h, s);
        
        // Step 6: actv(x) - Second LeakyReLU activation
        let h_data = h.data();
        let mut h_activated = vec![0.0; h_data.len()];
        
        for i in 0..h_data.len() {
            h_activated[i] = leaky_relu_activation(h_data[i], self.negative_slope);
        }
        h = Tensor::from_data(h_activated, h.shape().to_vec());
        
        // Step 7: conv2(dropout(x)) - Second convolution
        self.conv2.forward(&h)
    }
    
    /// EXACT PyTorch _shortcut implementation - NO ADAPTATIONS
    fn _shortcut(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let mut shortcut = x.clone();
        
        // Apply upsampling if enabled
        if let Some(ref pool) = self.upsample {
            shortcut = pool.forward(&shortcut);
        }
        
        // Apply learned conv1x1 if needed
        if self.learned_sc {
            assert!(self.conv1x1.is_some(), 
                "CRITICAL: learned_sc=true but conv1x1 is None - ARCHITECTURAL BUG");
            shortcut = self.conv1x1.as_ref().unwrap().forward(&shortcut);
        }
        
        shortcut
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
        println!("Loading AdainResBlk1d weights for {}.{} (STRICT - NO FALLBACKS)", component, prefix);
        
        // STRICT: Load conv1/conv2 - MUST SUCCEED
        self.conv1.load_weights_binary(loader, component, &format!("{}.conv1", prefix))
            .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlk1d conv1 loading FAILED: {}", e)))?;
        
        self.conv2.load_weights_binary(loader, component, &format!("{}.conv2", prefix))
            .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlk1d conv2 loading FAILED: {}", e)))?;
        
        // STRICT: Load conv1x1 shortcut if required
        if self.learned_sc {
            assert!(self.conv1x1.is_some(), "CRITICAL: learned_sc=true but conv1x1 is None");
            self.conv1x1.as_mut().unwrap().load_weights_binary(loader, component, &format!("{}.conv1x1", prefix))
                .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlk1d conv1x1 loading FAILED: {}", e)))?;
        }
        
        // STRICT: Load AdaIN layers - MUST SUCCEED
        self.norm1.load_weights_binary(loader, component, &format!("{}.norm1", prefix))
            .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlk1d norm1 loading FAILED: {}", e)))?;
        
        self.norm2.load_weights_binary(loader, component, &format!("{}.norm2", prefix))
            .map_err(|e| FerroError::new(format!("CRITICAL: AdaINResBlk1d norm2 loading FAILED: {}", e)))?;
        
        println!("✅ AdaINResBlk1d: All weights loaded with STRICT validation (ZERO FALLBACKS)");
        Ok(())
    }
}

impl Forward for AdainResBlk1d {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        panic!("AdainResBlk1d requires style conditioning - use forward(input, style) instead");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adain_resblk1d_exact_dimensions() {
        let block = AdainResBlk1d::new(512, 256, 128, true, 0.0);
        
        let input = Tensor::from_data(vec![0.1; 1 * 512 * 30], vec![1, 512, 30]);
        let style = Tensor::from_data(vec![0.1; 1 * 128], vec![1, 128]);
        
        let output = block.forward(&input, &style);
        
        // STRICT: Verify exact output dimensions
        assert_eq!(output.shape(), &[1, 256, 60], 
            "Output must have exact dimensions [1, 256, 60] for 512→256 with 2x upsampling");
    }
    
    #[test] 
    fn test_leaky_relu_activation() {
        // Test positive input
        assert_eq!(leaky_relu_activation(1.0, 0.2), 1.0);
        
        // Test negative input
        assert_eq!(leaky_relu_activation(-1.0, 0.2), -0.2);
        
        // Test zero
        assert_eq!(leaky_relu_activation(0.0, 0.2), 0.0);
    }
}