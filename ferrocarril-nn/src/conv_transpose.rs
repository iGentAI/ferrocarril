//! ConvTranspose1d implementation for upsampling in neural networks

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

#[derive(Debug)]
pub struct ConvTranspose1d {
    pub weight: Parameter,  // [in_channels, out_channels, kernel_size]
    pub bias: Option<Parameter>, // [out_channels]
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub groups: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
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
        // STRICT: Validate parameters
        assert!(in_channels > 0, "CRITICAL: in_channels must be positive, got: {}", in_channels);
        assert!(out_channels > 0, "CRITICAL: out_channels must be positive, got: {}", out_channels);
        assert!(kernel_size > 0, "CRITICAL: kernel_size must be positive, got: {}", kernel_size);
        assert!(stride > 0, "CRITICAL: stride must be positive, got: {}", stride);
        assert!(groups > 0, "CRITICAL: groups must be positive, got: {}", groups);
        
        // STRICT: Channel divisibility
        assert_eq!(in_channels % groups, 0,
            "CRITICAL: in_channels {} must be divisible by groups {}", in_channels, groups);
        assert_eq!(out_channels % groups, 0,
            "CRITICAL: out_channels {} must be divisible by groups {}", out_channels, groups);
        
        let weight = Parameter::new(Tensor::new(vec![in_channels, out_channels / groups, kernel_size]));
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
            output_padding,
            groups,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl Forward for ConvTranspose1d {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let input_shape = input.shape();
        
        // STRICT: Input must be 3D
        assert_eq!(input_shape.len(), 3, 
            "CRITICAL: ConvTranspose1d expects 3D input [batch, channels, length], got: {:?}", input_shape);
        
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        // STRICT: Channel count must match exactly
        assert_eq!(in_channels, self.in_channels,
            "CRITICAL: Input channels {} != expected {}. NO SILENT ADJUSTMENTS.",
            in_channels, self.in_channels);
        
        // Calculate output length for transposed convolution
        let out_length = (in_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding;
        
        // STRICT: Output length must be positive
        assert!(out_length > 0, "CRITICAL: Calculated output length is not positive");
        
        let mut output = vec![0.0; batch_size * self.out_channels * out_length];
        let weight_data = self.weight.data();
        
        // Transposed convolution computation
        for b in 0..batch_size {
            for ic in 0..self.in_channels {
                for il in 0..in_length {
                    for oc in 0..self.out_channels {
                        let group_id = oc / (self.out_channels / self.groups);
                        let group_in_channels = self.in_channels / self.groups;
                        
                        if ic / group_in_channels == group_id {
                            let ic_rel = ic % group_in_channels;
                            let oc_rel = oc % (self.out_channels / self.groups);
                            
                            for k in 0..self.kernel_size {
                                let ol = il * self.stride + k;
                                if ol >= self.padding && ol < out_length + self.padding {
                                    let ol_actual = ol - self.padding;
                                    let out_idx = b * self.out_channels * out_length + oc * out_length + ol_actual;
                                    output[out_idx] += input[&[b, ic, il]] * weight_data[&[ic_rel, oc_rel, k]];
                                }
                            }
                        }
                    }
                }
            }
            
            // Add bias
            if let Some(ref bias) = self.bias {
                let bias_data = bias.data();
                for oc in 0..self.out_channels {
                    for ol in 0..out_length {
                        let out_idx = b * self.out_channels * out_length + oc * out_length + ol;
                        output[out_idx] += bias_data[&[oc]];
                    }
                }
            }
        }
        
        Tensor::from_data(output, vec![batch_size, self.out_channels, out_length])
    }
}

impl LoadWeightsBinary for ConvTranspose1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Try weight normalization format first (most common for Generator)
        if let (Ok(weight_g), Ok(weight_v)) = (
            loader.load_component_parameter(component, &format!("{}.weight_g", prefix)),
            loader.load_component_parameter(component, &format!("{}.weight_v", prefix))
        ) {
            println!("Loading ConvTranspose1d with weight normalization for {}.{}", component, prefix);
            
            // Handle weight_g shape variations for transpose convolution
            let g_data = if weight_g.shape().len() == 1 {
                // For transpose convolution, g is per input channel
                weight_g.data().to_vec()
            } else if weight_g.shape().len() == 3 && weight_g.shape()[1] == 1 && weight_g.shape()[2] == 1 {
                let mut g_1d = Vec::new();
                for c in 0..self.in_channels {
                    g_1d.push(weight_g[&[c, 0, 0]]);
                }
                g_1d
            } else {
                return Err(FerroError::new(format!("Invalid weight_g shape for ConvTranspose1d: {:?}", weight_g.shape())));
            };
            
            // Reconstruct weight from weight normalization
            let expected_shape = vec![self.in_channels, self.out_channels / self.groups, self.kernel_size];
            if weight_v.shape() != &expected_shape {
                return Err(FerroError::new(format!(
                    "weight_v shape mismatch: expected {:?}, got {:?}",
                    expected_shape, weight_v.shape()
                )));
            }
            
            let mut normalized_weight = vec![0.0; self.in_channels * (self.out_channels / self.groups) * self.kernel_size];
            
            for ic in 0..self.in_channels {
                // Calculate L2 norm for this input channel
                let mut norm_sq = 0.0;
                for oc in 0..(self.out_channels / self.groups) {
                    for k in 0..self.kernel_size {
                        let val = weight_v[&[ic, oc, k]];
                        norm_sq += val * val;
                    }
                }
                let norm = norm_sq.sqrt() + 1e-8;
                let g_val = g_data[ic];
                
                for oc in 0..(self.out_channels / self.groups) {
                    for k in 0..self.kernel_size {
                        let idx = ic * (self.out_channels / self.groups) * self.kernel_size + oc * self.kernel_size + k;
                        normalized_weight[idx] = g_val * weight_v[&[ic, oc, k]] / norm;
                    }
                }
            }
            
            self.weight = Parameter::new(Tensor::from_data(normalized_weight, expected_shape));
        } else {
            // Try regular weight format
            let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))
                .map_err(|e| FerroError::new(format!("CRITICAL: Cannot load ConvTranspose1d weight for {}.{}: {}", component, prefix, e)))?;
            
            // STRICT: Weight shape must match layer configuration
            let expected_shape = vec![self.in_channels, self.out_channels / self.groups, self.kernel_size];
            assert_eq!(weight.shape(), &expected_shape,
                "CRITICAL: Weight shape mismatch for ConvTranspose1d layer {}.{}: expected {:?}, got {:?}",
                component, prefix, expected_shape, weight.shape());
            
            self.weight = Parameter::new(weight);
        }
        
        // Load bias if it exists
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            // STRICT: Bias shape must match
            assert_eq!(bias.shape(), &[self.out_channels],
                "CRITICAL: Bias shape mismatch for ConvTranspose1d layer {}.{}: expected [{}], got {:?}",
                component, prefix, self.out_channels, bias.shape());
            
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}