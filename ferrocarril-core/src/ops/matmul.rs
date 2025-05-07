//! Matrix multiplication operations

use crate::tensor::Tensor;

/// Matrix multiplication for 2D tensors
pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    assert_eq!(a.shape().len(), 2, "First tensor must be 2D");
    assert_eq!(b.shape().len(), 2, "Second tensor must be 2D");
    assert_eq!(a.shape()[1], b.shape()[0], "Inner dimensions must match");
    
    let m = a.shape()[0];
    let n = a.shape()[1];
    let p = b.shape()[1];
    
    let mut result = Tensor::new(vec![m, p]);
    
    for i in 0..m {
        for j in 0..p {
            let mut sum = T::default();
            for k in 0..n {
                sum = sum + a[&[i, k]].clone() * b[&[k, j]].clone();
            }
            result[&[i, j]] = sum;
        }
    }
    
    result
}

/// Batch matrix multiplication
pub fn batch_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    assert_eq!(a.shape().len(), 3, "First tensor must be 3D");
    assert_eq!(b.shape().len(), 3, "Second tensor must be 3D");
    assert_eq!(a.shape()[0], b.shape()[0], "Batch sizes must match");
    assert_eq!(a.shape()[2], b.shape()[1], "Inner dimensions must match");
    
    let batch_size = a.shape()[0];
    let m = a.shape()[1];
    let n = a.shape()[2];
    let p = b.shape()[2];
    
    let mut result = Tensor::new(vec![batch_size, m, p]);
    
    for b_idx in 0..batch_size {
        for i in 0..m {
            for j in 0..p {
                let mut sum = T::default();
                for k in 0..n {
                    sum = sum + a[&[b_idx, i, k]].clone() * b[&[b_idx, k, j]].clone();
                }
                result[&[b_idx, i, j]] = sum;
            }
        }
    }
    
    result
}

/// Matrix multiplication for 3D tensor with a 2D tensor
/// [B, M, K] @ [K, N] -> [B, M, N]
pub fn matmul_3d_2d<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where 
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    assert_eq!(a.shape().len(), 3, "First tensor must be 3D");
    assert_eq!(b.shape().len(), 2, "Second tensor must be 2D");
    assert_eq!(a.shape()[2], b.shape()[0], "Inner dimensions must match");
    
    let batch_size = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[1];
    
    let mut result = Tensor::new(vec![batch_size, m, n]);
    
    for b_idx in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    sum = sum + a[&[b_idx, i, l]].clone() * b[&[l, j]].clone();
                }
                result[&[b_idx, i, j]] = sum;
            }
        }
    }
    
    result
}

/// Matrix multiplication for 2D tensor with a 3D tensor
/// [M, K] @ [B, K, N] -> [B, M, N]
pub fn matmul_2d_3d<T>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T>
where 
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    assert_eq!(a.shape().len(), 2, "First tensor must be 2D");
    assert_eq!(b.shape().len(), 3, "Second tensor must be 3D");
    assert_eq!(a.shape()[1], b.shape()[1], "Inner dimensions must match");
    
    let m = a.shape()[0];
    let k = a.shape()[1];
    let batch_size = b.shape()[0];
    let n = b.shape()[2];
    
    let mut result = Tensor::new(vec![batch_size, m, n]);
    
    for b_idx in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    sum = sum + a[&[i, l]].clone() * b[&[b_idx, l, j]].clone();
                }
                result[&[b_idx, i, j]] = sum;
            }
        }
    }
    
    result
}