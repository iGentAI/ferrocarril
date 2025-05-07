//! Tensor implementation for Ferrocarril

use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

// Separate impl block for all generic T
impl<T> Tensor<T> {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get immutable access to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get mutable access to the underlying data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Convert indices to linear offset
    fn index_to_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        indices.iter().zip(&self.strides).map(|(&i, &s)| i * s).sum()
    }
}

// Specific impl block for T: Clone + Default
impl<T: Clone + Default> Tensor<T> {
    /// Create a new tensor with the given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        let size: usize = shape.iter().product();
        Tensor {
            data: vec![T::default(); size],
            shape,
            strides,
        }
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let total_elements: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), total_elements);
        
        let mut new_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        
        Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
        }
    }
}

// Specific impl block for T: Clone
impl<T: Clone> Tensor<T> {
    /// Create a tensor from data and shape
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(data.len(), total_elements);
        
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Tensor { data, shape, strides }
    }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        let offset = self.index_to_offset(indices);
        &self.data[offset]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        let offset = self.index_to_offset(indices);
        &mut self.data[offset]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t: Tensor<f32> = Tensor::new(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data().len(), 6);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut t: Tensor<f32> = Tensor::new(vec![2, 3]);
        t[&[0, 1]] = 1.0;
        assert_eq!(t[&[0, 1]], 1.0);
    }
}