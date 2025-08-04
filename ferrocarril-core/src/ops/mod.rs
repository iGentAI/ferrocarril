//! Operations module for Ferrocarril

pub mod matmul;

// Re-export matrix multiplication functions for easier access
pub use matmul::{matmul, batch_matmul, matmul_3d_2d, matmul_2d_3d};

use crate::tensor::Tensor;

/// Element-wise addition
pub fn add<T: Clone + std::ops::Add<Output = T> + Default>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");
    
    let data: Vec<T> = a.data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect();
    
    Tensor::from_data(data, a.shape().to_vec())
}

/// Element-wise multiplication
pub fn mul<T: Clone + std::ops::Mul<Output = T> + Default>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for multiplication");
    
    let data: Vec<T> = a.data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x.clone() * y.clone())
        .collect();
    
    Tensor::from_data(data, a.shape().to_vec())
}

/// Transpose a 2D tensor
pub fn transpose<T: Clone + Default>(tensor: &Tensor<T>) -> Tensor<T> {
    assert_eq!(tensor.shape().len(), 2, "Can only transpose 2D tensors");
    
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    let mut data = Vec::with_capacity(rows * cols);
    
    // Switch from row-major to column-major
    for j in 0..cols {
        for i in 0..rows {
            data.push(tensor[&[i, j]].clone());
        }
    }
    
    Tensor::from_data(data, vec![cols, rows])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = add(&a, &b);
        
        assert_eq!(c[&[0, 0]], 6.0);
        assert_eq!(c[&[0, 1]], 8.0);
        assert_eq!(c[&[1, 0]], 10.0);
        assert_eq!(c[&[1, 1]], 12.0);
    }
}