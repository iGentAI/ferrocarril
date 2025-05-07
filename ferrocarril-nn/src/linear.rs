//! Linear layer implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::ops::matmul;

pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize weight with small random-like values
        let mut weight_data = vec![0.0; out_features * in_features];
        for i in 0..weight_data.len() {
            // Use a deterministic pattern, but not zeros
            weight_data[i] = ((i % 10) as f32 - 5.0) * 0.01;
        }
        
        let weight = Parameter::new(Tensor::from_data(
            weight_data, 
            vec![out_features, in_features]
        ));
        
        // Initialize bias with small values
        let bias = if bias {
            let mut bias_data = vec![0.0; out_features];
            for i in 0..bias_data.len() {
                // Small non-zero bias values
                bias_data[i] = ((i % 5) as f32 - 2.0) * 0.01;
            }
            Some(Parameter::new(Tensor::from_data(
                bias_data, 
                vec![out_features]
            )))
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
    
    /// Get mutable reference to the weight parameter for testing
    pub fn weight_mut(&mut self) -> &mut Parameter {
        &mut self.weight
    }
}

impl Forward for Linear {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        // Input shape: [batch_size, in_features]
        // Weight shape: [out_features, in_features]
        // Output shape: [batch_size, out_features]
        
        let weight_transpose = ferrocarril_core::ops::transpose(&self.weight.data());
        
        // Batch matrix multiplication
        let output = if input.shape().len() == 1 {
            // Handle 1D input
            let mut expanded_data = Vec::with_capacity(input.shape()[0]);
            for i in 0..input.shape()[0] {
                expanded_data.push(input[&[i]]);
            }
            let input_2d = Tensor::from_data(expanded_data, vec![1, input.shape()[0]]);
            let result = matmul::matmul(&input_2d, &weight_transpose);
            
            // Extract the single row
            let mut flat_data = Vec::with_capacity(self.out_features);
            for j in 0..self.out_features {
                flat_data.push(result[&[0, j]]);
            }
            Tensor::from_data(flat_data, vec![self.out_features])
        } else {
            matmul::matmul(input, &weight_transpose)
        };
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Implement broadcasting for bias addition
            let mut result_data = output.data().to_vec();
            
            // If output is 1D, add bias directly
            if output.shape().len() == 1 {
                for i in 0..self.out_features {
                    result_data[i] += bias.data()[&[i]];
                }
                Tensor::from_data(result_data, output.shape().to_vec())
            } else {
                // If output is 2D, broadcast bias across batch dimension
                let batch_size = output.shape()[0];
                for b in 0..batch_size {
                    for i in 0..self.out_features {
                        result_data[b * self.out_features + i] += bias.data()[&[i]];
                    }
                }
                Tensor::from_data(result_data, output.shape().to_vec())
            }
        } else {
            output
        }
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