//! Adaptive Instance Normalization (AdaIN) implementation

use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use crate::linear::Linear;

/// Instance normalization for 1D data
pub struct InstanceNorm1d {
    num_features: usize,
    eps: f32,
    affine: bool,
    weight: Option<Parameter>,
    bias: Option<Parameter>,
}

impl InstanceNorm1d {
    pub fn new(num_features: usize, eps: f32, affine: bool) -> Self {
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
    
    /// Compute mean and variance for a channel
    fn compute_stats(&self, data: &[f32]) -> (f32, f32) {
        let len = data.len() as f32;
        
        // Compute mean
        let mean = data.iter().sum::<f32>() / len;
        
        // Compute variance
        let var = data.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>() / len;
        
        (mean, var)
    }
    
    /// Normalize data
    fn normalize(&self, data: &[f32], mean: f32, var: f32) -> Vec<f32> {
        let inv_std = 1.0 / (var + self.eps).sqrt();
        data.iter()
            .map(|&x| (x - mean) * inv_std)
            .collect()
    }
}

impl Forward for InstanceNorm1d {
    type Output = Tensor<f32>;
    
    fn forward(&self, input: &Tensor<f32>) -> Self::Output {
        let shape = input.shape();
        assert_eq!(shape.len(), 3, "Expected 3D input [batch, channels, length]");
        
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        assert_eq!(channels, self.num_features, "Channel mismatch");
        
        let mut output_data = Vec::with_capacity(batch * channels * length);
        
        for b in 0..batch {
            for c in 0..channels {
                // Extract channel data
                let mut channel_data = Vec::with_capacity(length);
                for l in 0..length {
                    channel_data.push(input[&[b, c, l]]);
                }
                
                // Compute stats and normalize
                let (mean, var) = self.compute_stats(&channel_data);
                let normalized = self.normalize(&channel_data, mean, var);
                
                // Apply affine transformation if enabled
                let transformed = if self.affine {
                    let weight = self.weight.as_ref().unwrap().data();
                    let bias = self.bias.as_ref().unwrap().data();
                    
                    normalized.iter()
                        .map(|&x| x * weight[&[c]] + bias[&[c]])
                        .collect()
                } else {
                    normalized
                };
                
                output_data.extend(transformed);
            }
        }
        
        Tensor::from_data(output_data, shape.to_vec())
    }
}

/// Adaptive Instance Normalization
pub struct AdaIN1d {
    style_dim: usize,
    num_features: usize,
    linear: Linear,
    norm: InstanceNorm1d,
}

impl AdaIN1d {
    pub fn new(style_dim: usize, num_features: usize) -> Self {
        Self {
            style_dim,
            num_features,
            linear: Linear::new(style_dim, num_features * 2, true),
            norm: InstanceNorm1d::new(num_features, 1e-5, false),
        }
    }
    
    /// Create a new AdaIN1d with a custom linear layer and normalization
    pub fn with_linear(style_dim: usize, num_features: usize, linear: Linear, norm: InstanceNorm1d) -> Self {
        Self {
            style_dim,
            num_features,
            linear,
            norm,
        }
    }
    
    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // Style transformation
        let h = self.linear.forward(style);
        
        // Debug output for style and transformed style
        println!("INFO: Style input shape: {:?}, first few values: {:?}", 
                 style.shape(), &style.data()[0..5.min(style.data().len())]);
        println!("INFO: Linear output shape: {:?}, values range: [{}, {}]", 
                 h.shape(), 
                 h.data().iter().fold(std::f32::INFINITY, |a, &b| a.min(b)),
                 h.data().iter().fold(std::f32::NEG_INFINITY, |a, &b| a.max(b)));
        
        // Reshape for gamma and beta
        let shape = h.shape();
        let (batch, features_x2) = (shape[0], shape[1]);
        
        // Split into gamma and beta
        let mut gamma_data = Vec::with_capacity(batch * self.num_features);
        let mut beta_data = Vec::with_capacity(batch * self.num_features);
        
        for b in 0..batch {
            for i in 0..self.num_features {
                gamma_data.push(h[&[b, i]]);
                beta_data.push(h[&[b, i + self.num_features]]);
            }
        }
        
        let gamma = Tensor::from_data(gamma_data, vec![batch, self.num_features, 1]);
        let beta = Tensor::from_data(beta_data, vec![batch, self.num_features, 1]);
        
        // Debug output for gamma and beta
        println!("INFO: Gamma range: [{}, {}], Beta range: [{}, {}]",
                 gamma.data().iter().fold(std::f32::INFINITY, |a, &b| a.min(b)),
                 gamma.data().iter().fold(std::f32::NEG_INFINITY, |a, &b| a.max(b)),
                 beta.data().iter().fold(std::f32::INFINITY, |a, &b| a.min(b)),
                 beta.data().iter().fold(std::f32::NEG_INFINITY, |a, &b| a.max(b)));
        println!("INFO: First few gamma values: {:?}", &gamma.data()[0..5.min(gamma.data().len())]);
        println!("INFO: First few beta values: {:?}", &beta.data()[0..5.min(beta.data().len())]);
        
        // Apply normalization
        let normalized = self.norm.forward(x);
        
        // Apply style conditioning
        let shape = normalized.shape();
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        let mut output_data = Vec::with_capacity(batch * channels * length);
        
        // Apply style directly: scale by gamma and shift by beta
        for b in 0..batch {
            for c in 0..channels {
                for l in 0..length {
                    let normalized_val = normalized[&[b, c, l]];
                    let gamma_val = gamma[&[b, c, 0]];
                    let beta_val = beta[&[b, c, 0]];
                    
                    output_data.push(gamma_val * normalized_val + beta_val);
                }
            }
        }
        
        Tensor::from_data(output_data, shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_instance_norm1d() {
        let norm = InstanceNorm1d::new(3, 1e-5, true);
        let input = Tensor::from_data(vec![0.0; 24], vec![2, 3, 4]); // [batch, channels, length]
        let output = norm.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }
    
    #[test]
    fn test_adain1d() {
        let adain = AdaIN1d::new(10, 5);
        let x = Tensor::from_data(vec![0.0; 40], vec![2, 5, 4]); // [batch, channels, length]
        let style = Tensor::from_data(vec![0.0; 20], vec![2, 10]); // [batch, style_dim]
        let output = adain.forward(&x, &style);
        assert_eq!(output.shape(), x.shape());
    }
}