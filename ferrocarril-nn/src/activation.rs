//! Activation functions with strict validation

use ferrocarril_core::tensor::Tensor;

/// ReLU activation function - STRICT VERSION
pub fn relu(x: &Tensor<f32>) -> Tensor<f32> {
    // STRICT: Input validation
    assert!(!x.data().is_empty(), "CRITICAL: Cannot apply ReLU to empty tensor");
    
    let data: Vec<f32> = x.data().iter().map(|&v| {
        assert!(v.is_finite(), "CRITICAL: Non-finite value in ReLU input: {}", v);
        v.max(0.0)
    }).collect();
    
    Tensor::from_data(data, x.shape().to_vec())
}

/// Leaky ReLU activation function - STRICT VERSION
pub fn leaky_relu(x: &Tensor<f32>, negative_slope: f32) -> Tensor<f32> {
    // STRICT: Parameter validation
    assert!(negative_slope.is_finite() && negative_slope >= 0.0,
        "CRITICAL: Invalid negative_slope for LeakyReLU: {}", negative_slope);
    assert!(!x.data().is_empty(), "CRITICAL: Cannot apply LeakyReLU to empty tensor");
    
    let data: Vec<f32> = x.data().iter().map(|&v| {
        assert!(v.is_finite(), "CRITICAL: Non-finite value in LeakyReLU input: {}", v);
        if v > 0.0 { v } else { negative_slope * v }
    }).collect();
    
    Tensor::from_data(data, x.shape().to_vec())
}

/// Sigmoid activation function - STRICT VERSION
pub fn sigmoid(x: &Tensor<f32>) -> Tensor<f32> {
    // STRICT: Input validation
    assert!(!x.data().is_empty(), "CRITICAL: Cannot apply Sigmoid to empty tensor");
    
    let data: Vec<f32> = x.data().iter().map(|&v| {
        assert!(v.is_finite(), "CRITICAL: Non-finite value in Sigmoid input: {}", v);
        // Clamp input to prevent overflow in exp
        let clamped = v.clamp(-88.0, 88.0);
        1.0 / (1.0 + (-clamped).exp())
    }).collect();
    
    Tensor::from_data(data, x.shape().to_vec())
}

/// Tanh activation function - STRICT VERSION
pub fn tanh(x: &Tensor<f32>) -> Tensor<f32> {
    // STRICT: Input validation
    assert!(!x.data().is_empty(), "CRITICAL: Cannot apply Tanh to empty tensor");
    
    let data: Vec<f32> = x.data().iter().map(|&v| {
        assert!(v.is_finite(), "CRITICAL: Non-finite value in Tanh input: {}", v);
        v.tanh()
    }).collect();
    
    Tensor::from_data(data, x.shape().to_vec())
}

/// Snake activation function - STRICT VERSION
pub fn snake(x: &Tensor<f32>, alpha: f32) -> Tensor<f32> {
    // STRICT: Parameter validation
    assert!(alpha > 0.0 && alpha.is_finite(),
        "CRITICAL: Invalid alpha for Snake activation: {}", alpha);
    assert!(!x.data().is_empty(), "CRITICAL: Cannot apply Snake to empty tensor");
    
    let data: Vec<f32> = x.data().iter().map(|&v| {
        assert!(v.is_finite(), "CRITICAL: Non-finite value in Snake input: {}", v);
        v + (1.0 / alpha) * (alpha * v).sin().powi(2)
    }).collect();
    
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
    
    #[test]
    #[should_panic(expected = "CRITICAL: Non-finite value")]
    fn test_activation_rejects_nan() {
        let input = Tensor::from_data(vec![1.0, f32::NAN, 2.0], vec![3]);
        relu(&input);
    }
}