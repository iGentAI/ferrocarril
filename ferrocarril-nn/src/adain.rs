//! AdaIN implementation with strict validation

use crate::{Parameter, Forward, linear::Linear};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

/// Instance Normalization 1D with strict validation
#[derive(Debug)]
pub struct InstanceNorm1d {
    pub num_features: usize,
    pub eps: f32,
    pub affine: bool,
    pub weight: Option<Parameter>, // [num_features]
    pub bias: Option<Parameter>,   // [num_features]
}

impl InstanceNorm1d {
    pub fn new(num_features: usize, eps: f32, affine: bool) -> Self {
        // STRICT: Validate parameters
        assert!(num_features > 0, "CRITICAL: num_features must be positive, got: {}", num_features);
        assert!(eps > 0.0 && eps < 1.0, "CRITICAL: eps must be in (0,1), got: {}", eps);
        
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
    
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let input_shape = input.shape();
        
        // STRICT: Input must be 3D
        assert_eq!(input_shape.len(), 3, 
            "CRITICAL: InstanceNorm1d expects 3D input [batch, channels, length], got: {:?}", input_shape);
        
        let (batch_size, input_channels, length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        // STRICT: No adaptations - dimensions must match exactly
        assert_eq!(input_channels, self.num_features,
            "CRITICAL: InstanceNorm1d channel mismatch: input has {} channels, layer configured for {}. \
            NO SILENT ADAPTATIONS - FIX YOUR LAYER CONFIGURATION.", 
            input_channels, self.num_features);
        
        let mut output = vec![0.0; batch_size * input_channels * length];
        
        for b in 0..batch_size {
            for c in 0..input_channels {
                // Calculate mean and variance for this channel
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                
                for l in 0..length {
                    let val = input[&[b, c, l]];
                    assert!(val.is_finite(), "CRITICAL: Non-finite value in InstanceNorm input");
                    sum += val;
                    sum_sq += val * val;
                }
                
                let mean = sum / length as f32;
                let variance = (sum_sq / length as f32) - (mean * mean);
                let std = (variance + self.eps).sqrt();
                
                // Apply normalization
                for l in 0..length {
                    let val = input[&[b, c, l]];
                    let normalized = (val - mean) / std;
                    
                    let final_val = if self.affine {
                        let gamma = self.weight.as_ref().unwrap().data()[&[c]];
                        let beta = self.bias.as_ref().unwrap().data()[&[c]];
                        normalized * gamma + beta
                    } else {
                        normalized
                    };
                    
                    output[b * input_channels * length + c * length + l] = final_val;
                }
            }
        }
        
        Tensor::from_data(output, input_shape.to_vec())
    }
}

/// Adaptive Instance Normalization 1D with strict validation
#[derive(Debug)]
pub struct AdaIN1d {
    pub fc: Linear,
    pub norm: InstanceNorm1d,
    pub num_features: usize,
}

impl AdaIN1d {
    pub fn new(num_features: usize, style_dim: usize) -> Self {
        // STRICT: Validate parameters
        assert!(num_features > 0, "CRITICAL: num_features must be positive, got: {}", num_features);
        assert!(style_dim > 0, "CRITICAL: style_dim must be positive, got: {}", style_dim);
        
        Self {
            fc: Linear::new(style_dim, num_features * 2, true),
            norm: InstanceNorm1d::new(num_features, 1e-5, false), // affine=false for AdaIN
            num_features,
        }
    }
    
    pub fn forward(&self, input: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Input validation
        assert_eq!(input.shape().len(), 3, "CRITICAL: AdaIN input must be 3D [batch, channels, time]");
        assert_eq!(style.shape().len(), 2, "CRITICAL: AdaIN style must be 2D [batch, style_dim]");
        
        let input_channels = input.shape()[1];
        
        // STRICT: No adaptations - channels must match exactly
        assert_eq!(input_channels, self.num_features,
            "CRITICAL: AdaIN channel mismatch: input has {} channels, layer configured for {}. \
            NO SILENT ADAPTATIONS - FIX YOUR LAYER CONFIGURATION OR WEIGHT LOADING.", 
            input_channels, self.num_features);
        
        // Apply instance normalization
        let normalized = self.norm.forward(input);
        
        // Get style parameters
        let style_params = self.fc.forward(style); // [batch, 2*num_features]
        
        // STRICT: Style params must have correct shape - no adaptations
        assert_eq!(style_params.shape()[1], self.num_features * 2,
            "CRITICAL: Style projection shape mismatch: expected {} features, got {}. \
            NO SILENT ADAPTATIONS - FIX YOUR FC LAYER CONFIGURATION.", 
            self.num_features * 2, style_params.shape()[1]);
        
        let batch_size = input.shape()[0];
        let length = input.shape()[2];
        let mut output = vec![0.0; normalized.data().len()];
        
        for b in 0..batch_size {
            for c in 0..input_channels {
                let gamma = style_params[&[b, c]];
                let beta = style_params[&[b, c + input_channels]];
                
                for l in 0..length {
                    let idx = b * input_channels * length + c * length + l;
                    output[idx] = normalized.data()[idx] * gamma + beta;
                }
            }
        }
        
        Tensor::from_data(output, input.shape().to_vec())
    }
}

impl LoadWeightsBinary for AdaIN1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Load the style projection layer
        self.fc.load_weights_binary(loader, component, &format!("{}.fc", prefix))?;
        Ok(())
    }
}