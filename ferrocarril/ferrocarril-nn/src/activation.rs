//! Activation functions

use ferrocarril_core::tensor::Tensor;

/// ReLU activation function
pub fn relu(x: &Tensor<f32>) -> Tensor<f32> {
    let data: Vec<f32> = x.data().iter().map(|&v| v.max(0.0)).collect();
    Tensor::from_data(data, x.shape().to_vec())
}

/// Leaky ReLU activation function
pub fn leaky_relu(x: &Tensor<f32>, negative_slope: f32) -> Tensor<f32> {
    let data: Vec<f32> = x.data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { negative_slope * v })
        .collect();
    Tensor::from_data(data, x.shape().to_vec())
}

/// Sigmoid activation function
pub fn sigmoid(x: &Tensor<f32>) -> Tensor<f32> {
    let data: Vec<f32> = x.data().iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    Tensor::from_data(data, x.shape().to_vec())
}

/// Tanh activation function
pub fn tanh(x: &Tensor<f32>) -> Tensor<f32> {
    let data: Vec<f32> = x.data().iter().map(|&v| v.tanh()).collect();
    Tensor::from_data(data, x.shape().to_vec())
}

/// Snake activation function
pub fn snake(x: &Tensor<f32>, alpha: f32) -> Tensor<f32> {
    let data: Vec<f32> = x.data()
        .iter()
        .map(|&v| v + (1.0 / alpha) * (alpha * v).sin().powi(2))
        .collect();
    Tensor::from_data(data, x.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_relu() {
        let input = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = relu(&input);
        
        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_leaky_relu() {
        let input = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let output = leaky_relu(&input, 0.1);
        
        assert_eq!(output.data(), &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }
}