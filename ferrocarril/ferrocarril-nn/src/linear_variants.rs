//! Specialized Linear implementations for different Kokoro TTS components
//! 
//! Different usage patterns found in Kokoro:
//! - BERTLinear: Attention projections [768,768], embeddings [178,128]
//! - ProjectionLinear: BERT→Hidden [512,768], Duration [50,512]  
//! - EmbeddingLinear: Token embeddings [178,128], position [512,128]

use crate::Parameter;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::{FerroError, LoadWeightsBinary};
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_core::ops::matmul;

/// BERT Linear layers: attention projections, feed-forward layers
/// 
/// Optimized for the 768-dimension BERT processing with efficient attention patterns
#[derive(Debug)]
pub struct BERTLinear {
    weight: Parameter,     // [out_features, in_features]
    bias: Option<Parameter>, // [out_features]
    in_features: usize,
    out_features: usize,
}

impl BERTLinear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            weight: Parameter::new(Tensor::new(vec![out_features, in_features])),
            bias: if bias {
                Some(Parameter::new(Tensor::new(vec![out_features])))
            } else {
                None
            },
            in_features,
            out_features,
        }
    }
    
    /// Forward pass optimized for BERT attention patterns
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // input: [batch_size, seq_len, in_features] for BERT
        let input_shape = input.shape();
        assert_eq!(input_shape.len(), 3, "BERT Linear expects 3D input [batch, seq, features]");
        assert_eq!(input_shape[2], self.in_features, "Input features mismatch");
        
        let (batch_size, seq_len, _) = (input_shape[0], input_shape[1], input_shape[2]);
        
        // Reshape to 2D for matrix multiplication
        let mut input_2d = vec![0.0; batch_size * seq_len * self.in_features];
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..self.in_features {
                    input_2d[b * seq_len * self.in_features + s * self.in_features + f] = input[&[b, s, f]];
                }
            }
        }
        let input_2d_tensor = Tensor::from_data(input_2d, vec![batch_size * seq_len, self.in_features]);
        
        // Apply linear transformation
        let weight_transpose = transpose_2d(&self.weight.data());
        let mut result = matmul::matmul(&input_2d_tensor, &weight_transpose);
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            let mut result_data = result.data().to_vec();
            for i in 0..result_data.len() {
                let feature_idx = i % self.out_features;
                result_data[i] += bias.data()[&[feature_idx]];
            }
            result = Tensor::from_data(result_data, result.shape().to_vec());
        }
        
        // Reshape back to 3D
        let mut output_3d = vec![0.0; batch_size * seq_len * self.out_features];
        let result_data = result.data();
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..self.out_features {
                    output_3d[b * seq_len * self.out_features + s * self.out_features + f] = 
                        result_data[b * seq_len * self.out_features + s * self.out_features + f];
                }
            }
        }
        
        Tensor::from_data(output_3d, vec![batch_size, seq_len, self.out_features])
    }
}

impl LoadWeightsBinary for BERTLinear {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))?;
        self.weight = Parameter::new(weight);
        
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}

/// Projection Linear layers: BERT→Hidden, Duration projections
/// 
/// Optimized for dimension transformation between major pipeline components
#[derive(Debug)]
pub struct ProjectionLinear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
    projection_type: ProjectionType,
}

#[derive(Debug)]
pub enum ProjectionType {
    BertToHidden,    // 768 → 512 BERT encoder projection
    DurationOut,     // 512 → 50 duration projection
    StyleProjection, // 128 → variable style conditioning
}

impl ProjectionLinear {
    pub fn new(in_features: usize, out_features: usize, bias: bool, projection_type: ProjectionType) -> Self {
        Self {
            weight: Parameter::new(Tensor::new(vec![out_features, in_features])),
            bias: if bias {
                Some(Parameter::new(Tensor::new(vec![out_features])))
            } else {
                None
            },
            in_features,
            out_features,
            projection_type,
        }
    }
    
    /// Forward pass optimized for projection usage patterns
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        match self.projection_type {
            ProjectionType::BertToHidden => self.forward_bert_projection(input),
            ProjectionType::DurationOut => self.forward_duration_projection(input),
            ProjectionType::StyleProjection => self.forward_style_projection(input),
        }
    }
    
    fn forward_bert_projection(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // BERT→Hidden: [B, T, 768] → [B, T, 512]
        assert_eq!(input.shape().len(), 3, "BERT projection expects 3D input");
        assert_eq!(input.shape()[2], 768, "BERT projection expects 768 input features");
        
        self.apply_linear_3d(input)
    }
    
    fn forward_duration_projection(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Duration: [B, T, 512] → [B, T, 50]
        assert_eq!(input.shape().len(), 3, "Duration projection expects 3D input");
        assert_eq!(input.shape()[2], 512, "Duration projection expects 512 input features");
        
        self.apply_linear_3d(input)
    }
    
    fn forward_style_projection(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // Style: [B, 128] → [B, out_features]
        assert_eq!(input.shape().len(), 2, "Style projection expects 2D input");
        assert_eq!(input.shape()[1], 128, "Style projection expects 128 input features");
        
        self.apply_linear_2d(input)
    }
    
    fn apply_linear_3d(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        let (batch_size, seq_len, in_features) = (shape[0], shape[1], shape[2]);
        
        // Reshape to 2D for matrix multiplication
        let mut input_2d = vec![0.0; batch_size * seq_len * in_features];
        for b in 0..batch_size {
            for s in 0..seq_len {
                for f in 0..in_features {
                    input_2d[b * seq_len * in_features + s * in_features + f] = input[&[b, s, f]];
                }
            }
        }
        
        let input_2d_tensor = Tensor::from_data(input_2d, vec![batch_size * seq_len, in_features]);
        let weight_transpose = transpose_2d(&self.weight.data());
        let result = matmul::matmul(&input_2d_tensor, &weight_transpose);
        
        // Add bias and reshape back to 3D
        let mut output_data = result.data().to_vec();
        if let Some(ref bias) = self.bias {
            for i in 0..output_data.len() {
                let feature_idx = i % self.out_features;
                output_data[i] += bias.data()[&[feature_idx]];
            }
        }
        
        Tensor::from_data(output_data, vec![batch_size, seq_len, self.out_features])
    }
    
    fn apply_linear_2d(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let weight_transpose = transpose_2d(&self.weight.data());
        let mut result = matmul::matmul(input, &weight_transpose);
        
        // Add bias
        if let Some(ref bias) = self.bias {
            let mut result_data = result.data().to_vec();
            for i in 0..result_data.len() {
                let feature_idx = i % self.out_features;
                result_data[i] += bias.data()[&[feature_idx]];
            }
            result = Tensor::from_data(result_data, result.shape().to_vec());
        }
        
        result
    }
}

impl LoadWeightsBinary for ProjectionLinear {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))?;
        self.weight = Parameter::new(weight);
        
        if let Ok(bias) = loader.load_component_parameter(component, &format!("{}.bias", prefix)) {
            self.bias = Some(Parameter::new(bias));
        }
        
        Ok(())
    }
}

/// Embedding Linear: token and position embeddings
#[derive(Debug)]
pub struct EmbeddingLinear {
    weight: Parameter,   // [vocab_size, embedding_dim] or [max_pos, embedding_dim]
    vocab_size: usize,
    embedding_dim: usize,
}

impl EmbeddingLinear {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            weight: Parameter::new(Tensor::new(vec![vocab_size, embedding_dim])),
            vocab_size,
            embedding_dim,
        }
    }
    
    /// Forward pass: lookup embeddings by index
    pub fn forward(&self, indices: &Tensor<i64>) -> Tensor<f32> {
        let indices_shape = indices.shape();
        let weight_data = self.weight.data();
        
        let mut output_data = vec![0.0; indices_shape.iter().product::<usize>() * self.embedding_dim];
        let mut output_idx = 0;
        
        for i in 0..indices.data().len() {
            let idx = indices.data()[i] as usize;
            assert!(idx < self.vocab_size, "Index {} out of vocabulary bounds", idx);
            
            for e in 0..self.embedding_dim {
                output_data[output_idx * self.embedding_dim + e] = weight_data[&[idx, e]];
            }
            output_idx += 1;
        }
        
        let mut output_shape = indices_shape.to_vec();
        output_shape.push(self.embedding_dim);
        Tensor::from_data(output_data, output_shape)
    }
}

impl LoadWeightsBinary for EmbeddingLinear {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        let weight = loader.load_component_parameter(component, &format!("{}.weight", prefix))?;
        self.weight = Parameter::new(weight);
        Ok(())
    }
}

/// Helper function to transpose a 2D tensor
fn transpose_2d(tensor: &Tensor<f32>) -> Tensor<f32> {
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "transpose_2d requires 2D tensor");
    let (rows, cols) = (shape[0], shape[1]);
    
    let mut transposed = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = tensor[&[r, c]];
        }
    }
    
    Tensor::from_data(transposed, vec![cols, rows])
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bert_linear() {
        let linear = BERTLinear::new(768, 768, true);
        let input = Tensor::new(vec![2, 10, 768]); // [batch, seq, features]
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[2, 10, 768]);
    }
    
    #[test]
    fn test_projection_linear() {
        let linear = ProjectionLinear::new(768, 512, true, ProjectionType::BertToHidden);
        let input = Tensor::new(vec![1, 5, 768]);
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[1, 5, 512]);
    }
    
    #[test]
    fn test_embedding_linear() {
        let embedding = EmbeddingLinear::new(178, 128);
        let indices = Tensor::from_data(vec![0i64, 1, 2], vec![3]);
        let output = embedding.forward(&indices);
        assert_eq!(output.shape(), &[3, 128]);
    }
}