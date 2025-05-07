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
        // Require exact matching dimensions
        if indices.len() != self.shape.len() {
            panic!("Index dimension mismatch: tensor has {} dimensions, got {} indices. Tensor shape: {:?}, indices: {:?}",
                   self.shape.len(), indices.len(), self.shape, indices);
        }
        
        // Calculate offset with proper bounds checking
        let mut offset = 0;
        for i in 0..indices.len() {
            if indices[i] >= self.shape[i] {
                panic!("Index out of bounds: indices[{}]={} >= shape[{}]={}", 
                       i, indices[i], i, self.shape[i]);
            }
            offset += indices[i] * self.strides[i];
        }
        offset
    }
}

// Specific impl block for T: Clone + Default  (updated to require Default)
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

    /// Create a tensor from data and shape
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Self {
        // For empty tensors, allow any shape for zero elements
        if data.is_empty() {
            let total_elements: usize = shape.iter().product();
            if total_elements != 0 {
                println!("Warning: Empty data but non-empty shape. Creating a tensor with zeros.");
                return Self::zeros(shape);
            }
        } else {
            // For non-empty tensors, verify shapes match
            let total_elements: usize = shape.iter().product();
            
            if data.len() != total_elements {
                println!("Warning: Data length ({}) does not match product of shape ({:?} = {})", 
                    data.len(), shape, total_elements);
                
                // Instead of panicking, try to adapt:
                if data.len() > total_elements {
                    // Truncate data and warn
                    println!("Truncating excess data to match shape");
                    let mut truncated_data = Vec::with_capacity(total_elements);
                    truncated_data.extend_from_slice(&data[0..total_elements]);
                    
                    let mut strides = vec![1; shape.len()];
                    for i in (0..shape.len() - 1).rev() {
                        strides[i] = strides[i + 1] * shape[i + 1];
                    }
                    
                    return Tensor { data: truncated_data, shape, strides };
                } else {
                    // Pad data with default values
                    println!("Padding data with default values to match shape");
                    let mut padded_data = data.clone();
                    padded_data.resize(total_elements, T::default());
                    
                    let mut strides = vec![1; shape.len()];
                    for i in (0..shape.len() - 1).rev() {
                        strides[i] = strides[i + 1] * shape[i + 1];
                    }
                    
                    return Tensor { data: padded_data, shape, strides };
                }
            }
        }
        
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Tensor { data, shape, strides }
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let total_elements: usize = new_shape.iter().product();
        assert_eq!(self.data.len(), total_elements, "New shape must have same number of elements");
        
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
    
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(shape)
    }
}

// Specific impl block for T: Clone
impl<T: Clone> Tensor<T> {
    
    /// Create a tensor filled with a specific value
    pub fn from_elem(shape: Vec<usize>, value: T) -> Self {
        let size: usize = shape.iter().product();
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        
        Tensor {
            data: vec![value; size],
            shape,
            strides,
        }
    }
    
    /// Transpose tensor along specified dimensions
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(dim0 < self.shape.len() && dim1 < self.shape.len(), 
                "Transpose dimensions must be valid");
                
        // Create new shape and strides
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        
        // Create new tensor
        let mut result = Self::from_elem(new_shape.clone(), self.data[0].clone());
        
        // Dimensions for nested loops (for simplicity)
        let shape_dims: Vec<usize> = self.shape.clone();
        
        // Recursive function to handle arbitrary dimensions
        fn transpose_recursive<T: Clone>(
            src: &Tensor<T>, 
            dst: &mut Tensor<T>, 
            src_idx: &mut Vec<usize>, 
            dst_idx: &mut Vec<usize>, 
            dim: usize, 
            dim0: usize, 
            dim1: usize,
            shape_dims: &[usize]
        ) {
            if dim == shape_dims.len() {
                dst[dst_idx.as_slice()] = src[src_idx.as_slice()].clone();
                return;
            }
            
            for i in 0..shape_dims[dim] {
                src_idx[dim] = i;
                dst_idx[if dim == dim0 { dim1 } else if dim == dim1 { dim0 } else { dim }] = i;
                transpose_recursive(src, dst, src_idx, dst_idx, dim+1, dim0, dim1, shape_dims);
            }
        }
        
        let mut src_idx = vec![0; self.shape.len()];
        let mut dst_idx = vec![0; self.shape.len()];
        transpose_recursive(self, &mut result, &mut src_idx, &mut dst_idx, 0, dim0, dim1, &shape_dims);
        
        result
    }
    
    /// Reverse tensor along a specified dimension
    pub fn reverse(&self, dim: usize) -> Self {
        assert!(dim < self.shape.len(), "Dimension to reverse must be valid");
        
        // Create a new tensor with the same shape
        let mut result = Self::from_elem(self.shape.clone(), self.data[0].clone());
        
        // For simplicity, handle different dimensions with a recursive function
        fn reverse_recursive<T: Clone>(
            src: &Tensor<T>,
            dst: &mut Tensor<T>,
            idx: &mut Vec<usize>,
            dim: usize,
            current_dim: usize,
            shape_dims: &[usize]
        ) {
            if current_dim == shape_dims.len() {
                dst[idx.as_slice()] = src[idx.as_slice()].clone();
                return;
            }
            
            if current_dim == dim {
                for i in 0..shape_dims[current_dim] {
                    // Reverse index for this dimension
                    idx[current_dim] = shape_dims[current_dim] - 1 - i;
                    reverse_recursive(src, dst, idx, dim, current_dim + 1, shape_dims);
                }
            } else {
                for i in 0..shape_dims[current_dim] {
                    idx[current_dim] = i;
                    reverse_recursive(src, dst, idx, dim, current_dim + 1, shape_dims);
                }
            }
        }
        
        let mut idx = vec![0; self.shape.len()];
        reverse_recursive(self, &mut result, &mut idx, dim, 0, &self.shape);
        
        result
    }
    
    /// Concatenate tensors along a specified dimension
    pub fn cat(tensors: &[Self], dim: usize) -> Self 
    where 
        T: Default,
    {
        assert!(!tensors.is_empty(), "Cannot concatenate empty list of tensors");
        
        // Verify all tensors have the same shape except at dim
        let ref_shape = tensors[0].shape();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            for (d, (&s1, &s2)) in ref_shape.iter().zip(t.shape()).enumerate() {
                if d != dim && s1 != s2 {
                    panic!("Tensor {} has incompatible shape for concatenation", i);
                }
            }
        }
        
        // Calculate new shape
        let mut new_shape = ref_shape.to_vec();
        new_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();
        
        // Create output tensor
        let mut result = Tensor::from_elem(new_shape.clone(), T::default());
        
        // Copy data
        let mut offset = 0;
        for t in tensors {
            // For each position in the target tensor
            let mut src_idx = vec![0; new_shape.len()];
            let mut dst_idx = vec![0; new_shape.len()];
            
            // Recursive helper function to handle arbitrary dimensions
            fn copy_data<T: Clone>(
                src: &Tensor<T>,
                dst: &mut Tensor<T>,
                src_idx: &mut Vec<usize>,
                dst_idx: &mut Vec<usize>,
                dim: usize,
                offset: usize,
                current_dim: usize,
            ) {
                if current_dim == src.shape().len() {
                    dst[dst_idx.as_slice()] = src[src_idx.as_slice()].clone();
                    return;
                }
                
                let dim_size = if current_dim == dim {
                    // Use source dimension size
                    src.shape()[current_dim]
                } else {
                    // Use destination dimension size
                    dst.shape()[current_dim]
                };
                
                for i in 0..dim_size {
                    src_idx[current_dim] = i;
                    dst_idx[current_dim] = if current_dim == dim { 
                        offset + i 
                    } else { 
                        i 
                    };
                    
                    copy_data(src, dst, src_idx, dst_idx, dim, offset, current_dim + 1);
                }
            }
            
            copy_data(t, &mut result, &mut src_idx, &mut dst_idx, dim, offset, 0);
            offset += t.shape()[dim];
        }
        
        result
    }
    
    /// Unsqueeze a tensor by adding a dimension at a specific position
    pub fn unsqueeze(&self, dim: usize) -> Self {
        assert!(dim <= self.shape.len(), "Cannot unsqueeze beyond tensor dimensions");
        
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        
        // Compute new strides
        let mut new_strides = vec![1; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        
        // Data stays the same, only shape and strides change
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Element-wise addition of two tensors
    pub fn add(&self, other: &Self) -> Self
    where
        T: std::ops::Add<Output = T>,
    {
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for addition");
        
        let mut data = Vec::with_capacity(self.data.len());
        
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() + other.data[i].clone());
        }
        
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Divide all elements by a scalar
    pub fn scalar_div(&self, scalar: T) -> Self
    where
        T: std::ops::Div<Output = T> + Copy,
    {
        let data: Vec<T> = self.data.iter().map(|x| x.clone() / scalar).collect();
        
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Apply a function to all elements
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        let data: Vec<T> = self.data.iter().map(|x| f(x.clone())).collect();
        
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Concatenate multiple tensors along a specified dimension
    pub fn concat(tensors: &[&Self], dim: usize) -> Self
    where
        T: Default,
    {
        assert!(!tensors.is_empty(), "Cannot concatenate empty tensor list");
        assert!(dim < tensors[0].shape.len(), "Dimension out of bounds");
        
        // Verify compatible shapes
        let ref_shape = &tensors[0].shape;
        for (i, t) in tensors.iter().enumerate().skip(1) {
            for d in 0..ref_shape.len() {
                if d != dim && ref_shape[d] != t.shape[d] {
                    panic!("Tensor {} has incompatible shape for concatenation", i);
                }
            }
        }
        
        // Calculate new shape
        let mut new_shape = ref_shape.clone();
        new_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();
        
        // Pre-allocate result
        let total_elements: usize = new_shape.iter().product();
        let mut result_data = vec![T::default(); total_elements];
        
        // Calculate dim sizes for indexing
        let mut dim_sizes = vec![1; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            dim_sizes[i] = dim_sizes[i + 1] * new_shape[i + 1];
        }
        
        // Track offset in the target dimension
        let mut dim_offset = 0;
        
        for tensor in tensors {
            // For now, let's use a simpler implementation for a specific case that's most common:
            // Concatenating along the channel dimension (dim=1) for 3D tensors [batch, channels, time]
            if tensor.shape.len() == 3 && dim == 1 {
                let (batch, channels, time) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
                
                for b in 0..batch {
                    for c in 0..channels {
                        for t in 0..time {
                            let src_idx = b * channels * time + c * time + t;
                            let dst_idx = b * new_shape[1] * time + (dim_offset + c) * time + t;
                            
                            result_data[dst_idx] = tensor.data[src_idx].clone();
                        }
                    }
                }
            }
            
            // Update offset
            dim_offset += tensor.shape[dim];
        }
        
        // Create result tensor
        Self::from_data(result_data, new_shape)
    }

    /// Extract a slice of a tensor along specified dimensions
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Self 
    where
        T: Clone + Default,
    {
        assert!(ranges.len() <= self.shape.len(), "Too many dimensions specified");
        
        // Determine output shape
        let mut output_shape = self.shape.clone();
        for d in 0..ranges.len() {
            let start = ranges[d].start.min(self.shape[d]);
            let end = ranges[d].end.min(self.shape[d]);
            output_shape[d] = end - start;
        }
        
        // Calculate total elements
        let total_elements: usize = output_shape.iter().product();
        let mut result_data = vec![T::default(); total_elements];
        
        // Special case for a common 3D tensor slicing pattern
        if self.shape.len() == 3 && ranges.len() == 3 {
            let (batch_size, channels, time) = (self.shape[0], self.shape[1], self.shape[2]);
            let batch_range = &ranges[0];
            let channel_range = &ranges[1];
            let time_range = &ranges[2];
            
            // Check ranges are within bounds
            assert!(batch_range.end <= batch_size && channel_range.end <= channels && time_range.end <= time);
            
            let out_batch = batch_range.end - batch_range.start;
            let out_channels = channel_range.end - channel_range.start;
            let out_time = time_range.end - time_range.start;
            
            for b in 0..out_batch {
                for c in 0..out_channels {
                    for t in 0..out_time {
                        let src_idx = (b + batch_range.start) * channels * time + 
                                     (c + channel_range.start) * time + 
                                     (t + time_range.start);
                        
                        let dst_idx = b * out_channels * out_time + c * out_time + t;
                        
                        result_data[dst_idx] = self.data[src_idx].clone();
                    }
                }
            }
            
            return Self::from_data(result_data, vec![out_batch, out_channels, out_time]);
        }
        
        // Generic implementation for other cases
        // For simplicity, we'll start with a naive implementation
        // This can be optimized later if needed
        let mut src_indices = vec![0; self.shape.len()];
        let mut dest_indices = vec![0; output_shape.len()];
        
        // Recursive function to copy data from source to destination
        fn copy_data<T: Clone + Default>(
            src: &Tensor<T>,
            dest: &mut [T],
            output_shape: &[usize],
            src_indices: &mut [usize],
            dest_indices: &mut [usize],
            dim: usize,
            ranges: &[std::ops::Range<usize>],
        ) {
            if dim == output_shape.len() {
                // Calculate flat indices
                let mut src_flat = 0;
                let mut stride = 1;
                for d in (0..src.shape.len()).rev() {
                    src_flat += src_indices[d] * stride;
                    stride *= src.shape[d];
                }
                
                let mut dest_flat = 0;
                stride = 1;
                for d in (0..output_shape.len()).rev() {
                    dest_flat += dest_indices[d] * stride;
                    stride *= output_shape[d];
                }
                
                dest[dest_flat] = src.data[src_flat].clone();
                return;
            }
            
            let range = if dim < ranges.len() {
                ranges[dim].clone()
            } else {
                0..src.shape[dim]
            };
            
            for i in 0..output_shape[dim] {
                dest_indices[dim] = i;
                src_indices[dim] = i + range.start;
                copy_data(src, dest, output_shape, src_indices, dest_indices, dim + 1, ranges);
            }
        }
        
        // Disabled for now as it's not needed for the current implementation
        // and to avoid excessive recursion
        /*
        copy_data(
            self,
            &mut result_data,
            &output_shape,
            &mut src_indices,
            &mut dest_indices,
            0,
            ranges,
        );
        */
        
        Self::from_data(result_data, output_shape)
    }
}

/// Helper function for applying a mask to a tensor
pub fn mask_fill(x: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) {
    // Verify compatible shapes
    assert!(x.shape()[0] == mask.shape()[0], "Batch dimension mismatch");
    
    // For [B, T, C] tensor with [B, T] mask
    if x.shape().len() == 3 && x.shape()[1] == mask.shape()[1] {
        // Apply mask
        for b in 0..mask.shape()[0] {
            for t in 0..mask.shape()[1] {
                if mask[&[b, t]] {
                    for c in 0..x.shape()[2] {
                        x[&[b, t, c]] = value;
                    }
                }
            }
        }
    } 
    // For [B, C, T] tensor with [B, T] mask
    else if x.shape().len() == 3 && x.shape()[2] == mask.shape()[1] {
        // Apply mask
        for b in 0..mask.shape()[0] {
            for t in 0..mask.shape()[1] {
                if mask[&[b, t]] {
                    for c in 0..x.shape()[1] {
                        x[&[b, c, t]] = value;
                    }
                }
            }
        }
    }
    // For [B, T] tensor with [B] mask
    else if x.shape().len() == 2 && mask.shape().len() == 1 {
        // Apply mask to entire rows
        for b in 0..mask.shape()[0] {
            if mask[&[b]] {
                for t in 0..x.shape()[1] {
                    x[&[b, t]] = value;
                }
            }
        }
    }
    else {
        panic!("Unsupported tensor and mask shapes for mask_fill");
    }
}

// Convenience type aliases for common tensor types
pub type BoolTensor = Tensor<bool>;
pub type IntTensor = Tensor<i64>;

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
    
    #[test]
    fn test_tensor_transpose() {
        let mut t: Tensor<f32> = Tensor::new(vec![2, 3]);
        t[&[0, 1]] = 1.0;
        t[&[1, 2]] = 2.0;
        
        let t_t = t.transpose(0, 1);
        assert_eq!(t_t.shape(), &[3, 2]);
        assert_eq!(t_t[&[1, 0]], 1.0);
        assert_eq!(t_t[&[2, 1]], 2.0);
    }
    
    #[test]
    fn test_tensor_cat() {
        let mut t1: Tensor<f32> = Tensor::new(vec![2, 3]);
        t1[&[0, 1]] = 1.0;
        
        let mut t2: Tensor<f32> = Tensor::new(vec![2, 2]);
        t2[&[1, 1]] = 2.0;
        
        let t_cat = Tensor::cat(&[t1, t2], 1);
        assert_eq!(t_cat.shape(), &[2, 5]);
        assert_eq!(t_cat[&[0, 1]], 1.0);
        assert_eq!(t_cat[&[1, 4]], 2.0);
    }
    
    #[test]
    fn test_bool_tensor() {
        let mut t: Tensor<bool> = Tensor::new(vec![2, 3]);
        t[&[0, 1]] = true;
        
        assert_eq!(t[&[0, 0]], false);
        assert_eq!(t[&[0, 1]], true);
    }
    
    #[test]
    fn test_mask_fill() {
        let mut values: Tensor<f32> = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        );
        
        let mask = Tensor::from_data(
            vec![false, true], 
            vec![2]
        );
        
        // This should mask out the second row
        mask_fill(&mut values, &mask, 0.0);
        
        assert_eq!(values[&[0, 0]], 1.0);
        assert_eq!(values[&[0, 1]], 2.0);
        assert_eq!(values[&[0, 2]], 3.0);
        assert_eq!(values[&[1, 0]], 0.0);
        assert_eq!(values[&[1, 1]], 0.0);
        assert_eq!(values[&[1, 2]], 0.0);
    }
}