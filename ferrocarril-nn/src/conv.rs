//! Basic Conv1d implementation with strict validation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

#[derive(Debug)]
pub struct Conv1d {
    pub weight: Parameter,  // [out_channels, in_channels/groups, kernel_size]
    pub bias: Option<Parameter>, // [out_channels]
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

impl Conv1d {
    /// Load weight from reconstructed PyTorch weight normalization
    pub fn load_from_reconstructed_weight(&mut self, weight: &Tensor<f32>) -> Result<(), FerroError> {
        // STRICT: Validate weight shape matches our layer configuration
        let expected_shape = vec![self.out_channels, self.in_channels / self.groups, self.kernel_size];
        assert_eq!(weight.shape(), &expected_shape,
            "CRITICAL: Reconstructed weight shape mismatch: expected {:?}, got {:?}",
            expected_shape, weight.shape());
        
        self.weight = Parameter::new(weight.clone());
        Ok(())
    }
    
    /// Set bias parameter from loaded weights
    pub fn set_bias(&mut self, bias: &Tensor<f32>) -> Result<(), FerroError> {
        // STRICT: Validate bias shape
        assert_eq!(bias.shape(), &[self.out_channels],
            "CRITICAL: Bias shape mismatch: expected [{}], got {:?}", 
            self.out_channels, bias.shape());
        
        self.bias = Some(Parameter::new(bias.clone()));
        Ok(())
    }

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
        // STRICT: Validate all parameters
        assert!(in_channels > 0, "CRITICAL: in_channels must be positive, got: {}", in_channels);
        assert!(out_channels > 0, "CRITICAL: out_channels must be positive, got: {}", out_channels);
        assert!(kernel_size > 0, "CRITICAL: kernel_size must be positive, got: {}", kernel_size);
        assert!(stride > 0, "CRITICAL: stride must be positive, got: {}", stride);
        assert!(groups > 0, "CRITICAL: groups must be positive, got: {}", groups);
        assert!(dilation > 0, "CRITICAL: dilation must be positive, got: {}", dilation);
        
        // STRICT: Channel divisibility 
        assert_eq!(in_channels % groups, 0,
            "CRITICAL: in_channels {} must be divisible by groups {}", in_channels, groups);
        assert_eq!(out_channels % groups, 0,
            "CRITICAL: out_channels {} must be divisible by groups {}", out_channels, groups);
        
        let weight = Parameter::new(Tensor::new(vec![out_channels, in_channels / groups, kernel_size]));
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
            kernel_size,
        }
    }
}

impl Forward for Conv1d {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let input_shape = input.shape();
        
        // STRICT: Input must be 3D
        assert_eq!(input_shape.len(), 3, 
            "CRITICAL: Conv1d expects 3D input [batch, channels, length], got: {:?}", input_shape);
        
        let (batch_size, in_channels, in_length) = (input_shape[0], input_shape[1], input_shape[2]);
        
        // STRICT: Channel count must match exactly
        assert_eq!(in_channels, self.in_channels,
            "CRITICAL: Input channels {} != expected {}. NO SILENT ADJUSTMENTS.",
            in_channels, self.in_channels);
        
        // Calculate output length
        let dilated_kernel_size = (self.kernel_size - 1) * self.dilation + 1;
        let out_length = (in_length + 2 * self.padding - dilated_kernel_size) / self.stride + 1;
        
        // STRICT: Output length must be positive
        assert!(out_length > 0, "CRITICAL: Calculated output length is not positive");
        
        let mut output = vec![0.0; batch_size * self.out_channels * out_length];
        let weight_data = self.weight.data();
        
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                let group_id = oc / (self.out_channels / self.groups);
                let group_in_channels = self.in_channels / self.groups;
                let group_in_start = group_id * group_in_channels;
                
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    
                    for ic_rel in 0..group_in_channels {
                        let ic = group_in_start + ic_rel;
                        
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k * self.dilation;
                            if il < self.padding || il >= in_length + self.padding {
                                continue;
                            }
                            let il_actual = il - self.padding;
                            if il_actual < in_length {
                                sum += input[&[b, ic, il_actual]] * weight_data[&[oc, ic_rel, k]];
                            }
                        }
                    }
                    
                    if let Some(ref bias) = self.bias {
                        sum += bias.data()[&[oc]];
                    }
                    
                    output[b * self.out_channels * out_length + oc * out_length + ol] = sum;
                }
            }
        }
        
        Tensor::from_data(output, vec![batch_size, self.out_channels, out_length])
    }
}

impl LoadWeightsBinary for Conv1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // Try to load weight normalization format first (weight_g/weight_v)
        let weight_loaded = if let (Ok(weight_g), Ok(weight_v)) = (
            loader.load_component_parameter(component, &format!("{}.weight_g", prefix)),
            loader.load_component_parameter(component, &format!("{}.weight_v", prefix))
        ) {
            // Handle weight normalization format (PyTorch weight_norm)
            println!("Loading Conv1d weights using weight normalization format for {}.{}", component, prefix);
            
            // Handle weight_g shape variations  
            let g_data = if weight_g.shape().len() == 1 {
                weight_g.data().to_vec()
            } else if weight_g.shape().len() == 3 && weight_g.shape()[1] == 1 && weight_g.shape()[2] == 1 {
                let mut g_1d = Vec::new();
                for c in 0..self.out_channels {
                    g_1d.push(weight_g[&[c, 0, 0]]);
                }
                g_1d
            } else {
                return Err(FerroError::new(format!("Invalid weight_g shape: {:?}", weight_g.shape())));
            };
            
            // Reconstruct weight from normalization
            let mut normalized_weight = vec![0.0; self.out_channels * (self.in_channels / self.groups) * self.kernel_size];
            
            for oc in 0..self.out_channels {
                let mut norm_sq = 0.0;
                for ic in 0..(self.in_channels / self.groups) {
                    for k in 0..self.kernel_size {
                        let val = weight_v[&[oc, ic, k]];
                        norm_sq += val * val;
                    }
                }
                let norm = norm_sq.sqrt() + 1e-8;
                let g_val = g_data[oc];
                
                for ic in 0..(self.in_channels / self.groups) {
                    for k in 0..self.kernel_size {
                        let idx = oc * (self.in_channels / self.groups) * self.kernel_size + ic * self.kernel_size + k;
                        normalized_weight[idx] = g_val * weight_v[&[oc, ic, k]] / norm;
                    }
                }
            }
            
            self.weight = Parameter::new(Tensor::from_data(
                normalized_weight, 
                vec![self.out_channels, self.in_channels / self.groups, self.kernel_size]
            ));
            true
        } else {
            false
        };
        
        if !weight_loaded {
            // Try regular weight format
            println!("Trying regular weight format for {}.{}", component, prefix);
            match loader.load_component_parameter(component, &format!("{}.weight", prefix)) {
                Ok(weight) => {
                    println!("Successfully loaded regular weight for {}.{} with shape {:?}", 
                            component, prefix, weight.shape());
                    
                    // STRICT: Weight shape must match layer configuration
                    let expected_shape = vec![self.out_channels, self.in_channels / self.groups, self.kernel_size];
                    if weight.shape() != &expected_shape {
                        return Err(FerroError::new(format!(
                            "Conv1d weight shape mismatch for {}.{}: expected {:?}, got {:?}",
                            component, prefix, expected_shape, weight.shape()
                        )));
                    }
                    
                    self.weight = Parameter::new(weight);
                }
                Err(e) => {
                    return Err(FerroError::new(format!(
                        "CRITICAL: Cannot load Conv1d weight for {}.{} in any format: {}",
                        component, prefix, e
                    )));
                }
            }
        }
        
        // Load bias if it exists
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            // STRICT: Bias shape must match
            assert_eq!(bias.shape(), &[self.out_channels],
                "CRITICAL: Bias shape mismatch for Conv1d layer {}.{}: expected [{}], got {:?}",
                component, prefix, self.out_channels, bias.shape());
            
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}