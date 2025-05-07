//! Matrix multiplication operations

use crate::tensor::Tensor;

/// Basic matrix multiplication implementation
pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.shape().len(), 2, "First tensor must be 2D");
    assert_eq!(b.shape().len(), 2, "Second tensor must be 2D");
    assert_eq!(a.shape()[1], b.shape()[0], "Incompatible dimensions for matrix multiplication");
    
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    assert_eq!(k1, k2);
    
    let mut result = Tensor::new(vec![m, n]);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += a[&[i, k]] * b[&[k, j]];
            }
            result[&[i, j]] = sum;
        }
    }
    
    result
}

/// SIMD-optimized matrix multiplication for x86_64
#[cfg(target_arch = "x86_64")]
pub fn matmul_simd(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    // TODO: Implement SIMD optimization
    matmul(a, b) // Fallback for now
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_matmul() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = matmul(&a, &b);
        
        assert_eq!(c[&[0, 0]], 19.0);
        assert_eq!(c[&[0, 1]], 22.0);
        assert_eq!(c[&[1, 0]], 43.0);
        assert_eq!(c[&[1, 1]], 50.0);
    }
}