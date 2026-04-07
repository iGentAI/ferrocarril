//! Linear layer implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use ferrocarril_core::weights::{PyTorchWeightLoader, LoadWeights};
use ferrocarril_core::ops::matmul::linear_f32;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

#[derive(Debug)]
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
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
    
    /// Set the weight tensor
    pub fn set_weight(&mut self, weight: &Tensor<f32>) -> Result<(), FerroError> {
        // Validate shape
        if weight.shape().len() != 2 || 
           weight.shape()[0] != self.out_features || 
           weight.shape()[1] != self.in_features {
            return Err(FerroError::new(format!(
                "Invalid weight shape: {:?}, expected [{}, {}]", 
                weight.shape(), self.out_features, self.in_features
            )));
        }
        
        // Set weight
        self.weight = Parameter::new(weight.clone());
        
        Ok(())
    }
    
    /// Set the bias tensor
    pub fn set_bias(&mut self, bias: &Tensor<f32>) -> Result<(), FerroError> {
        // Validate shape
        if bias.shape().len() != 1 || bias.shape()[0] != self.out_features {
            return Err(FerroError::new(format!(
                "Invalid bias shape: {:?}, expected [{}]", 
                bias.shape(), self.out_features
            )));
        }
        
        // Update bias
        if self.bias.is_some() {
            self.bias = Some(Parameter::new(bias.clone()));
        }
        
        Ok(())
    }
    
    /// Load weight and bias directly from tensors (no prefix/component lookup)
    pub fn load_weight_bias(&mut self, weight: &Tensor<f32>, bias: Option<&Tensor<f32>>) -> Result<(), FerroError> {
        // Set weight
        self.set_weight(weight)?;
        
        // Set bias if provided and if this layer has bias
        if let Some(bias_tensor) = bias {
            if self.bias.is_some() {
                self.set_bias(bias_tensor)?;
            }
        }
        
        Ok(())
    }
    
}

impl Forward for Linear {
    type Output = Tensor<f32>;

    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let in_shape = input.shape().to_vec();
        let in_ndim = in_shape.len();

        let (m, out_shape) = match in_ndim {
            1 => {
                assert_eq!(
                    in_shape[0], self.in_features,
                    "Linear input feature dim {} does not match in_features {}",
                    in_shape[0], self.in_features
                );
                (1usize, vec![self.out_features])
            }
            2 => {
                assert_eq!(
                    in_shape[1], self.in_features,
                    "Linear input feature dim {} does not match in_features {}",
                    in_shape[1], self.in_features
                );
                (in_shape[0], vec![in_shape[0], self.out_features])
            }
            3 => {
                assert_eq!(
                    in_shape[2], self.in_features,
                    "Linear input feature dim {} does not match in_features {}",
                    in_shape[2], self.in_features
                );
                (
                    in_shape[0] * in_shape[1],
                    vec![in_shape[0], in_shape[1], self.out_features],
                )
            }
            _ => panic!(
                "Linear layer only supports 1D, 2D, or 3D input tensors, got shape: {:?}",
                in_shape
            ),
        };

        let k = self.in_features;
        let n = self.out_features;

        let bias_slice: Option<&[f32]> =
            self.bias.as_ref().map(|b| b.data().data());

        let mut out = vec![0.0f32; m * n];
        linear_f32(
            input.data(),
            self.weight.data().data(),
            bias_slice,
            &mut out,
            m,
            k,
            n,
        );

        Tensor::from_data(out, out_shape)
    }
}

impl LoadWeights for Linear {
    fn load_weights(
        &mut self, 
        loader: &PyTorchWeightLoader, 
        prefix: Option<&str>
    ) -> Result<(), FerroError> {
        // Load weight tensor
        loader.load_weight_into_parameter(&mut self.weight, "weight", prefix, None)?;
        
        // Load bias tensor if available
        if let Some(ref mut bias) = self.bias {
            loader.load_weight_into_parameter(bias, "bias", prefix, None)?;
        }
        
        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for Linear {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        // Load weight - fail immediately if missing
        let weight_path = format!("{}.weight", prefix);
        let weight = loader.load_component_parameter(component, &weight_path)?;
        self.set_weight(&weight)?;

        // Load bias if present
        if self.bias.is_some() {
            let bias_path = format!("{}.bias", prefix);
            let bias = loader.load_component_parameter(component, &bias_path)?;
            self.set_bias(&bias)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(4, 3, true);
        let input = Tensor::from_data(vec![0.0; 8], vec![2, 4]); // Batch size 2, 4 features
        let output = linear.forward(&input);
        
        assert_eq!(output.shape(), &[2, 3]);
    }
}