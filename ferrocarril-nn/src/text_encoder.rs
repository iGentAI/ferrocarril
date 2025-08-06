//! TextEncoder – port of kokoro/modules.py::TextEncoder
//! Uses only specialized implementations for Kokoro TTS pipeline

use crate::lstm_variants::TextEncoderLSTM;
use crate::conv1d_variants::TextEncoderConv1d;
use crate::{Parameter, Forward};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use std::sync::Arc;

/// Helper function for applying a mask to a tensor - STRICT VERSION
fn mask_fill_strict(x: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) {
    // STRICT: Verify compatible shapes exactly - no adaptive behavior
    assert_eq!(x.shape()[0], mask.shape()[0], 
        "STRICT: Batch dimension mismatch - tensor: {}, mask: {}", x.shape()[0], mask.shape()[0]);
    
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
    } else {
        panic!("STRICT: Incompatible tensor shapes for mask_fill - tensor: {:?}, mask: {:?}", 
               x.shape(), mask.shape());
    }
}

// -------------------------------------------------
// Helper: embedding (gather along 0-th dim)
// -------------------------------------------------
#[derive(Debug)]
pub struct Embedding {
    pub(crate) weight: Parameter, // [n_symbols, channels] float32
}

impl Embedding {
    pub fn new(n_symbols: usize, channels: usize) -> Self {
        Self {
            weight: Parameter::new(Tensor::new(vec![n_symbols, channels])),
        }
    }

    /// x : [B, T]  (i64 indices)
    /// returns [B, T, C] float32
    pub fn forward(&self, x: &Tensor<i64>) -> Tensor<f32> {
        // STRICT: Validate input shape exactly
        assert_eq!(x.shape().len(), 2, 
            "STRICT: Embedding input must be 2D [batch, time], got: {:?}", x.shape());
            
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let c = self.weight.data().shape()[1];
        let mut out = vec![0.0f32; b * t * c];
        let w = self.weight.data();

        for batch in 0..b {
            for pos in 0..t {
                let idx = x[&[batch, pos]] as usize;
                
                // STRICT: Validate vocab index bounds
                assert!(idx < w.shape()[0], 
                    "STRICT: Vocab index {} out of bounds [0, {})", idx, w.shape()[0]);
                
                for ch in 0..c {
                    out[batch * t * c + pos * c + ch] = w[&[idx, ch]];
                }
            }
        }
        Tensor::from_data(out, vec![b, t, c])
    }
}

// -------------------------------------------------
// Helper: LayerNorm over channel dim (like PyTorch impl)
// -------------------------------------------------
#[derive(Debug)]
pub struct LayerNorm {
    pub(crate) gamma: Parameter, // [C]
    pub(crate) beta:  Parameter, // [C]
    eps: f32,
}

impl LayerNorm {
    pub fn new(channels: usize, eps: f32) -> Self {
        Self {
            gamma: Parameter::new(Tensor::from_data(vec![1.0; channels], vec![channels])),
            beta:  Parameter::new(Tensor::from_data(vec![0.0; channels], vec![channels])),
            eps,
        }
    }

    // Input  : [B, C, T]
    // Output : same shape
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Validate input shape exactly
        assert_eq!(input.shape().len(), 3,
            "STRICT: LayerNorm input must be 3D [batch, channels, time], got: {:?}", input.shape());
            
        let (b, c, t) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut out = vec![0.0; b * c * t];

        // STRICT: Validate channels match layer configuration
        assert_eq!(c, self.gamma.data().shape()[0],
            "STRICT: Input channels {} don't match LayerNorm channels {}", 
            c, self.gamma.data().shape()[0]);

        for batch in 0..b {
            for ch in 0..c {
                // mean & variance across time dimension
                let mut mean = 0.0;
                let mut var  = 0.0;
                let base_idx = batch * c * t + ch * t;
                for pos in 0..t {
                    let v = input[&[batch, ch, pos]];
                    mean += v;
                    var  += v * v;
                }
                mean /= t as f32;
                var  = var / t as f32 - mean * mean;

                let denom = (var + self.eps).sqrt();
                let g = self.gamma.data()[&[ch]];
                let b_ = self.beta.data()[&[ch]];

                for pos in 0..t {
                    let v = input[&[batch, ch, pos]];
                    out[base_idx + pos] = (v - mean) / denom * g + b_;
                }
            }
        }
        Tensor::from_data(out, vec![b, c, t])
    }
}

// -------------------------------------------------
// Helper: LeakyReLU
// -------------------------------------------------
#[inline]
fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * x }
}

// -------------------------------------------------
// Convolutional residual block
// (Conv1d / LayerNorm / LeakyReLU)
// -------------------------------------------------
#[derive(Debug)]
pub struct ConvBlock {
    pub(crate) conv: TextEncoderConv1d,
    pub(crate) ln:   LayerNorm,
    negative_slope: f32,
}

impl ConvBlock {
    pub fn new(channels: usize, kernel: usize) -> Self {
        let padding = (kernel - 1) / 2;
        Self {
            conv: TextEncoderConv1d::new(
                channels, channels, kernel,
                1,                 // stride
                padding,           // padding
                1,                 // dilation
                1,                 // groups
                true,              // bias
            ),
            ln: LayerNorm::new(channels, 1e-5),
            negative_slope: 0.2,
        }
    }

    /// x : [B, C, T]
    pub fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        // STRICT: Validate input shape
        assert_eq!(x.shape().len(), 3,
            "STRICT: ConvBlock input must be 3D [batch, channels, time], got: {:?}", x.shape());
            
        let y = self.conv.forward(x);              // [B, C, T] - using specialized Conv1d
        
        // STRICT: Verify conv output shape unchanged
        assert_eq!(y.shape(), x.shape(),
            "STRICT: ConvBlock changed shape unexpectedly: {:?} -> {:?}", x.shape(), y.shape());
            
        let y = self.ln.forward(&y);               // LN
        
        // Apply LeakyReLU
        let mut output_data = y.data().to_vec();
        for v in output_data.iter_mut() {
            *v = leaky_relu(*v, self.negative_slope);
        }
        
        Tensor::from_data(output_data, y.shape().to_vec())
    }
}

// -------------------------------------------------
// TextEncoder
// -------------------------------------------------
#[derive(Debug)]
pub struct TextEncoder {
    pub(crate) embedding: Embedding,          // nn.Embedding  
    pub(crate) cnn:       Vec<Arc<ConvBlock>>, // depth x ConvBlock
    pub(crate) lstm:      TextEncoderLSTM,     // Specialized TextEncoder LSTM
    channels:  usize,
}

impl TextEncoder {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        depth: usize,
        n_symbols: usize,
    ) -> Self {
        // CNN blocks using specialized TextEncoderConv1d
        let mut cnn = Vec::with_capacity(depth);
        for _ in 0..depth {
            cnn.push(Arc::new(ConvBlock::new(channels, kernel_size)));
        }

        // Specialized bidirectional LSTM for TextEncoder: 512→512
        let lstm = TextEncoderLSTM::new(
            channels,               // input_size = 512
            channels / 2,           // hidden_size = 256 (bidirectional doubles to 512)
        );

        Self {
            embedding: Embedding::new(n_symbols, channels),
            cnn,
            lstm,
            channels,
        }
    }

    /// Forward pass with specialized bidirectional LSTM
    pub fn forward(
        &self,
        x_tok: &Tensor<i64>,
        input_lengths: &Vec<usize>,  
        mask: &Tensor<bool>,
    ) -> Tensor<f32> {
        // STRICT INPUT VALIDATION
        assert_eq!(x_tok.shape().len(), 2,
            "STRICT: Input tokens must be 2D [batch, time], got: {:?}", x_tok.shape());
        assert_eq!(mask.shape(), x_tok.shape(),
            "STRICT: Mask shape {} must exactly match input shape {:?}", 
            format!("{:?}", mask.shape()), format!("{:?}", x_tok.shape()));
        assert_eq!(input_lengths.len(), x_tok.shape()[0],
            "STRICT: input_lengths count {} must match batch size {}", 
            input_lengths.len(), x_tok.shape()[0]);
        
        // Validate all input lengths are within sequence bounds
        for (i, &length) in input_lengths.iter().enumerate() {
            assert!(length <= x_tok.shape()[1], 
                "STRICT: input_lengths[{}]={} exceeds sequence length {}", 
                i, length, x_tok.shape()[1]);
        }
        
        // 1. Text embedding: [B, T] → [B, T, C]
        let mut x = self.embedding.forward(x_tok);      // [B, T, channels]
        
        // Zero out masked positions
        self.apply_mask_btc(&mut x, mask);

        // 2. Transpose for CNN: [B, T, C] → [B, C, T]
        let mut x_bct = self.transpose_btc_to_bct_strict(&x);        

        // 3. CNN processing with masking after each layer
        for (i, blk) in self.cnn.iter().enumerate() {
            x_bct = blk.forward(&x_bct);
            
            // Apply mask after each CNN block
            self.apply_mask_bct_strict(&mut x_bct, mask); 
        }

        // 4. Transpose back for LSTM: [B, C, T] → [B, T, C]
        let x_btc = self.transpose_bct_to_btc_strict(&x_bct);

        // 5. Use specialized TextEncoderLSTM with bidirectional processing
        let (lstm_out, _) = self.lstm.forward_batch_first_with_lengths(&x_btc, input_lengths, None, None);  // [B, T, 512]
        
        // 6. Transpose to final format: [B, T, C] → [B, C, T]
        let mut final_output = self.transpose_btc_to_bct_strict(&lstm_out);
        
        // 7. Final mask application
        self.apply_mask_bct_strict(&mut final_output, mask);
        
        final_output  // [B, 512, T]
    }

    /// Apply mask to [B, T, C] tensor
    fn apply_mask_btc(&self, tensor: &mut Tensor<f32>, mask: &Tensor<bool>) {
        assert_eq!(tensor.shape()[0], mask.shape()[0], "Batch dimension mismatch");
        assert_eq!(tensor.shape()[1], mask.shape()[1], "Time dimension mismatch");
        
        for b in 0..mask.shape()[0] {
            for t in 0..mask.shape()[1] {
                if mask[&[b, t]] {
                    for c in 0..tensor.shape()[2] {
                        tensor[&[b, t, c]] = 0.0;
                    }
                }
            }
        }
    }

    /// Transpose implementation for [B, T, C] → [B, C, T]
    fn transpose_btc_to_bct_strict(&self, x: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(x.shape().len(), 3, "Expected 3D tensor [B, T, C]");
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut out = vec![0.0; b * c * t];
        
        for batch in 0..b {
            for time in 0..t {
                for chan in 0..c {
                    out[batch * c * t + chan * t + time] = x[&[batch, time, chan]];
                }
            }
        }
        
        Tensor::from_data(out, vec![b, c, t])
    }

    fn transpose_bct_to_btc_strict(&self, x: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(x.shape().len(), 3, "Expected 3D tensor [B, C, T]");
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut out = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for chan in 0..c {
                for time in 0..t {
                    out[batch * t * c + time * c + chan] = x[&[batch, chan, time]];
                }
            }
        }
        
        Tensor::from_data(out, vec![b, t, c])
    }

    /// Apply mask to [B, C, T] tensor
    fn apply_mask_bct_strict(&self, tensor: &mut Tensor<f32>, mask: &Tensor<bool>) {
        assert_eq!(tensor.shape()[0], mask.shape()[0], "Batch dimension mismatch");
        assert_eq!(tensor.shape()[2], mask.shape()[1], "Time dimension mismatch");

        for b in 0..mask.shape()[0] {
            for t in 0..mask.shape()[1] {
                if mask[&[b, t]] {
                    for c in 0..tensor.shape()[1] {
                        tensor[&[b, c, t]] = 0.0;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------
// LoadWeights trait implementation for TextEncoder
// -------------------------------------------------
impl ferrocarril_core::weights::LoadWeights for TextEncoder {
    fn load_weights(
        &mut self,
        _loader: &ferrocarril_core::weights::PyTorchWeightLoader,
        _prefix: Option<&str>,
    ) -> Result<(), ferrocarril_core::FerroError> {
        // This is only a stub for the PyTorchWeightLoader
        // The real implementation is in load_weights_binary
        Ok(())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for TextEncoder {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading TextEncoder weights for {}.{}", component, prefix);
        
        // Load embedding weights
        let embedding_weight = loader.load_component_parameter(component, "module.embedding.weight")?;
        self.embedding.weight = Parameter::new(embedding_weight);
        
        // Load CNN blocks using specialized TextEncoderConv1d
        for i in 0..self.cnn.len().min(3) {
            let block = Arc::get_mut(&mut self.cnn[i])
                .ok_or_else(|| FerroError::new(format!("Cannot get mutable reference to CNN block {}", i)))?;

            // Load specialized TextEncoderConv1d weights
            block.conv.load_weights_binary(loader, component, &format!("{}.cnn.{}.0", prefix, i))?;
            
            // Load LayerNorm weights
            let gamma = loader.load_component_parameter(component, &format!("{}.cnn.{}.1.gamma", prefix, i))?;
            let beta = loader.load_component_parameter(component, &format!("{}.cnn.{}.1.beta", prefix, i))?;
            
            block.ln.gamma = Parameter::new(gamma);
            block.ln.beta = Parameter::new(beta);
        }
        
        // Load specialized TextEncoderLSTM weights
        self.lstm.load_weights_binary(loader, component, &format!("{}.lstm", prefix))?;
        
        println!("✅ TextEncoder loaded successfully with specialized implementations only");
        Ok(())
    }
}