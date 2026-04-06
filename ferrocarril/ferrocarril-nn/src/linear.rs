//! Linear layer implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
use ferrocarril_core::weights::{PyTorchWeightLoader, LoadWeights};
// Import just matmul and transpose since the other functions are not found
use ferrocarril_core::ops::matmul::matmul;
// Import transpose directly from ferrocarril_core::ops
use ferrocarril_core::ops::transpose;
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
        // Handle different input shapes:
        // - 1D: [in_features]
        // - 2D: [batch_size, in_features]
        // - 3D: [batch_size, seq_len, in_features]
        
        let weight_transpose = transpose(&self.weight.data());
        
        // Get output based on input shape
        let output = match input.shape().len() {
            1 => {
                // 1D input: [in_features] -> [out_features]
                // Expand to 2D, perform matmul, then flatten back
                let input_2d = Tensor::from_data(
                    input.data().to_vec(),
                    vec![1, input.shape()[0]]
                );
                let result_2d = matmul(&input_2d, &weight_transpose);
                
                // Extract the single row
                let mut flat_data = Vec::with_capacity(self.out_features);
                for j in 0..self.out_features {
                    flat_data.push(result_2d[&[0, j]]);
                }
                Tensor::from_data(flat_data, vec![self.out_features])
            },
            2 => {
                // 2D input: [batch_size, in_features] -> [batch_size, out_features]
                matmul(&input, &weight_transpose)
            },
            3 => {
                // 3D input: [batch_size, seq_len, in_features] -> [batch_size, seq_len, out_features]
                // Implement manually for 3D case since matmul_3d_2d is not found
                let batch_size = input.shape()[0];
                let seq_len = input.shape()[1];
                let in_features = input.shape()[2];
                let out_features = self.out_features;
                
                // Initialize output tensor
                let mut result_data = vec![0.0; batch_size * seq_len * out_features];
                
                // Perform 3D tensor matrix multiplication manually
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        for o in 0..out_features {
                            // For each output feature, compute weighted sum of input features
                            let mut sum = 0.0;
                            for i in 0..in_features {
                                sum += input[&[b, s, i]] * weight_transpose[&[i, o]];
                            }
                            result_data[b * seq_len * out_features + s * out_features + o] = sum;
                        }
                    }
                }
                
                Tensor::from_data(result_data, vec![batch_size, seq_len, out_features])
            },
            _ => {
                panic!("Linear layer only supports 1D, 2D, or 3D input tensors, got shape: {:?}", input.shape());
            }
        };
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            match output.shape().len() {
                1 => {
                    // 1D output: [out_features]
                    // Add bias directly
                    let mut result_data = output.data().to_vec();
                    for i in 0..self.out_features {
                        result_data[i] += bias.data()[&[i]];
                    }
                    Tensor::from_data(result_data, output.shape().to_vec())
                },
                2 => {
                    // 2D output: [batch_size, out_features]
                    // Broadcast bias across batch dimension
                    let mut result_data = output.data().to_vec();
                    let batch_size = output.shape()[0];
                    for b in 0..batch_size {
                        for i in 0..self.out_features {
                            result_data[b * self.out_features + i] += bias.data()[&[i]];
                        }
                    }
                    Tensor::from_data(result_data, output.shape().to_vec())
                },
                3 => {
                    // 3D output: [batch_size, seq_len, out_features]
                    // Broadcast bias across batch and sequence dimensions
                    let mut result_data = output.data().to_vec();
                    let batch_size = output.shape()[0];
                    let seq_len = output.shape()[1];
                    for b in 0..batch_size {
                        for s in 0..seq_len {
                            for i in 0..self.out_features {
                                let idx = b * seq_len * self.out_features + s * self.out_features + i;
                                result_data[idx] += bias.data()[&[i]];
                            }
                        }
                    }
                    Tensor::from_data(result_data, output.shape().to_vec())
                },
                _ => unreachable!()  // We've already checked dimensions above
            }
        } else {
            output
        }
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