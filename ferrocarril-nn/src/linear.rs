//! Basic Linear layer implementation with strict validation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;

#[derive(Debug)]
pub struct Linear {
    pub weight: Parameter,  // [out_features, in_features]
    pub bias: Option<Parameter>, // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // STRICT: Validate parameters
        assert!(in_features > 0, "CRITICAL: in_features must be positive, got: {}", in_features);
        assert!(out_features > 0, "CRITICAL: out_features must be positive, got: {}", out_features);
        
        let weight = Parameter::new(Tensor::new(vec![out_features, in_features]));
        let bias = if bias {
            Some(Parameter::new(Tensor::new(vec![out_features])))
        } else {
            None
        };
        
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
}

impl Forward for Linear {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let input_shape = input.shape();
        
        // Handle both 2D [batch, features] and 3D [batch, seq, features] inputs
        match input_shape.len() {
            2 => {
                let (batch_size, in_features) = (input_shape[0], input_shape[1]);
                
                // STRICT: Feature dimensions must match exactly
                assert_eq!(in_features, self.in_features,
                    "CRITICAL: Input features {} != expected {}. NO SILENT ADJUSTMENTS.",
                    in_features, self.in_features);
                
                let mut output = vec![0.0; batch_size * self.out_features];
                let weight_data = self.weight.data();
                
                for b in 0..batch_size {
                    for o in 0..self.out_features {
                        let mut sum = 0.0;
                        for i in 0..self.in_features {
                            sum += input[&[b, i]] * weight_data[&[o, i]];
                        }
                        
                        if let Some(ref bias) = self.bias {
                            sum += bias.data()[&[o]];
                        }
                        
                        output[b * self.out_features + o] = sum;
                    }
                }
                
                Tensor::from_data(output, vec![batch_size, self.out_features])
            },
            3 => {
                let (batch_size, seq_len, in_features) = (input_shape[0], input_shape[1], input_shape[2]);
                
                // STRICT: Feature dimensions must match exactly
                assert_eq!(in_features, self.in_features,
                    "CRITICAL: Input features {} != expected {}. NO SILENT ADJUSTMENTS.",
                    in_features, self.in_features);
                
                let mut output = vec![0.0; batch_size * seq_len * self.out_features];
                let weight_data = self.weight.data();
                
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        for o in 0..self.out_features {
                            let mut sum = 0.0;
                            for i in 0..self.in_features {
                                sum += input[&[b, s, i]] * weight_data[&[o, i]];
                            }
                            
                            if let Some(ref bias) = self.bias {
                                sum += bias.data()[&[o]];
                            }
                            
                            output[b * seq_len * self.out_features + s * self.out_features + o] = sum;
                        }
                    }
                }
                
                Tensor::from_data(output, vec![batch_size, seq_len, self.out_features])
            },
            _ => panic!("CRITICAL: Linear layer only supports 2D or 3D inputs, got: {:?}", input_shape)
        }
    }
}

impl LoadWeightsBinary for Linear {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        // STRICT: Weight must exist
        let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))
            .map_err(|e| FerroError::new(format!("CRITICAL: Cannot load Linear weight for {}.{}: {}", component, prefix, e)))?;
        
        // STRICT: NO ADAPTATIONS - Weight shape must match layer configuration exactly
        let actual_shape = weight.shape();
        if actual_shape.len() == 2 {
            let (actual_out, actual_in) = (actual_shape[0], actual_shape[1]);
            
            // ZERO TOLERANCE for dimension mismatches - fail loudly instead of adapting
            assert_eq!(actual_out, self.out_features,
                "CRITICAL: Linear layer {}.{} output dimension mismatch: layer configured for {}, weight has {}. \
                NO SILENT ADAPTATIONS - FIX YOUR LAYER CONFIGURATION.", 
                component, prefix, self.out_features, actual_out);
            
            assert_eq!(actual_in, self.in_features,
                "CRITICAL: Linear layer {}.{} input dimension mismatch: layer configured for {}, weight has {}. \
                NO SILENT ADAPTATIONS - FIX YOUR LAYER CONFIGURATION.", 
                component, prefix, self.in_features, actual_in);
        } else {
            return Err(FerroError::new(format!(
                "CRITICAL: Linear weight must be 2D, got shape {:?} for {}.{}", 
                actual_shape, component, prefix
            )));
        }
        
        self.weight = Parameter::new(weight);
        
        // Load bias if it exists
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            // STRICT: NO ADAPTATIONS for bias either
            assert_eq!(bias.shape()[0], self.out_features,
                "CRITICAL: Bias shape mismatch for Linear layer {}.{}: layer configured for {}, bias has {}. \
                NO SILENT ADAPTATIONS - FIX YOUR LAYER CONFIGURATION.",
                component, prefix, self.out_features, bias.shape()[0]);
                
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}