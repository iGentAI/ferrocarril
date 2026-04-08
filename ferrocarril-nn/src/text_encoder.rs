//! TextEncoder – port of kokoro/modules.py::TextEncoder
//! Inference-only; numerically identical to the original model
//! (assuming the PyTorch weights are exported verbatim).
//! 
//! STRICT VALIDATION: No silent fallbacks, all dimension mismatches cause immediate failure

use crate::{conv::Conv1d, lstm::LSTM, Forward, Parameter};
use ferrocarril_core::tensor::Tensor;
use std::sync::Arc;

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

        let g_data = self.gamma.data();
        let b_data = self.beta.data();
        let inv_c = 1.0_f32 / c as f32;

        for batch in 0..b {
            for pos in 0..t {
                let mut mean = 0.0_f32;
                let mut sq_sum = 0.0_f32;
                for ch in 0..c {
                    let v = input[&[batch, ch, pos]];
                    mean += v;
                    sq_sum += v * v;
                }
                mean *= inv_c;
                let var = sq_sum * inv_c - mean * mean;
                let denom = (var + self.eps).sqrt();

                for ch in 0..c {
                    let v = input[&[batch, ch, pos]];
                    let g = g_data[&[ch]];
                    let bb = b_data[&[ch]];
                    out[batch * c * t + ch * t + pos] = (v - mean) / denom * g + bb;
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
    pub(crate) conv: Conv1d,
    pub(crate) ln:   LayerNorm,
    negative_slope: f32,
}

impl ConvBlock {
    pub fn new(channels: usize, kernel: usize) -> Self {
        let padding = (kernel - 1) / 2;
        Self {
            conv: Conv1d::new(
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
            
        let y = self.conv.forward(x);              // [B, C, T]
        
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
    pub(crate) lstm:      LSTM,               // Single bidirectional LSTM
    channels:  usize,
}

impl TextEncoder {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        depth: usize,
        n_symbols: usize,
    ) -> Self {
        // CNN blocks
        let mut cnn = Vec::with_capacity(depth);
        for _ in 0..depth {
            cnn.push(Arc::new(ConvBlock::new(channels, kernel_size)));
        }

        // Single bidirectional LSTM
        let lstm = LSTM::new(
            channels,               // input_size
            channels / 2,           // hidden_size (bidirectional doubles this back to channels)
            1,                      // num_layers
            true,                   // batch_first
            true,                   // bidirectional
        );

        Self {
            embedding: Embedding::new(n_symbols, channels),
            cnn,
            lstm,
            channels,
        }
    }

    /// Forward pass with bidirectional LSTM
    /// Tensor flow: x: [B, T] → embedding: [B, T, C] → transpose: [B, C, T] → CNN → transpose: [B, T, C] → LSTM → transpose: [B, C, T]
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
        for (_i, blk) in self.cnn.iter().enumerate() {
            x_bct = blk.forward(&x_bct);
            
            // Apply mask after each CNN block
            self.apply_mask_bct_strict(&mut x_bct, mask); 
        }

        // 4. Transpose back for LSTM: [B, C, T] → [B, T, C]
        let x_btc = self.transpose_bct_to_btc_strict(&x_bct);

        // 5. Single bidirectional LSTM
        let (lstm_out, _) = self.lstm.forward_batch_first(&x_btc, None, None);  // [B, T, channels]
        
        // 6. Transpose to final format: [B, T, C] → [B, C, T]
        let mut final_output = self.transpose_btc_to_bct_strict(&lstm_out);
        
        // 7. Final mask application
        self.apply_mask_bct_strict(&mut final_output, mask);
        
        final_output  // [B, channels, T]
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

impl TextEncoder {
    /// Load weights from a binary weight loader - STRICT VERSION
    pub fn load_weights_binary(
        &mut self, 
        loader: &ferrocarril_core::weights_binary::BinaryWeightLoader
    ) -> Result<(), ferrocarril_core::FerroError> {
        // Component name must be "text_encoder" to match the output structure from the weight converter
        let component = "text_encoder";
        
        // STRICT: Load embedding weights - fail immediately if missing
        let embedding_weight_path = "module.embedding.weight";
        let embedding_weight = loader.load_component_parameter(component, embedding_weight_path)
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load embedding weights: {}", e)))?;
        
        // STRICT: Validate embedding weight shape
        assert_eq!(embedding_weight.shape(), &[self.embedding.weight.data().shape()[0], self.channels],
            "STRICT: Embedding weight shape mismatch");
            
        self.embedding.weight = Parameter::new(embedding_weight);
        
        // STRICT: Load CNN blocks - fail immediately if any missing
        for i in 0..self.cnn.len() {
            // Skip blocks beyond what exists in the weights - but be explicit about this
            if i >= 3 {
                return Err(ferrocarril_core::FerroError::new(format!(
                    "STRICT: CNN block {} requested but only 3 blocks exist in weights", i)));
            }
            
            // Get mutable access to the block
            let block = match Arc::get_mut(&mut self.cnn[i]) {
                Some(b) => b,
                None => {
                    return Err(ferrocarril_core::FerroError::new(format!(
                        "STRICT: Cannot get mutable reference to CNN block {}", i)));
                }
            };

            // STRICT: Load Conv1d weights - fail immediately if missing
            let weight_g_path = format!("module.cnn.{}.0.weight_g", i);
            let weight_v_path = format!("module.cnn.{}.0.weight_v", i);
            let bias_path = format!("module.cnn.{}.0.bias", i);
            
            let weight_g = loader.load_component_parameter(component, &weight_g_path)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load {}: {}", weight_g_path, e)))?;
            let weight_v = loader.load_component_parameter(component, &weight_v_path)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load {}: {}", weight_v_path, e)))?;
            
            block.conv.set_weight_norm(&weight_g, &weight_v)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("STRICT: Failed to set weight norm for block {}: {}", i, e)))?;
            
            // Load bias
            let bias = loader.load_component_parameter(component, &bias_path)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load {}: {}", bias_path, e)))?;
            block.conv.set_bias(&bias)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("STRICT: Failed to set bias for block {}: {}", i, e)))?;
            
            // STRICT: Load LayerNorm weights - fail immediately if missing
            let gamma_path = format!("module.cnn.{}.1.gamma", i);
            let beta_path = format!("module.cnn.{}.1.beta", i);
            
            let gamma = loader.load_component_parameter(component, &gamma_path)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load {}: {}", gamma_path, e)))?;
            let beta = loader.load_component_parameter(component, &beta_path)
                .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load {}: {}", beta_path, e)))?;
            
            block.ln.gamma = Parameter::new(gamma);
            block.ln.beta = Parameter::new(beta);
        }
        
        // STRICT: Load bidirectional LSTM weights - fail immediately if missing
        
        // Forward direction weights
        let forward_weight_ih = loader.load_component_parameter(component, "module.lstm.weight_ih_l0")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM forward weight_ih: {}", e)))?;
        let forward_weight_hh = loader.load_component_parameter(component, "module.lstm.weight_hh_l0")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM forward weight_hh: {}", e)))?;
        let forward_bias_ih = loader.load_component_parameter(component, "module.lstm.bias_ih_l0")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM forward bias_ih: {}", e)))?;
        let forward_bias_hh = loader.load_component_parameter(component, "module.lstm.bias_hh_l0")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM forward bias_hh: {}", e)))?;
        
        // Backward direction weights
        let backward_weight_ih = loader.load_component_parameter(component, "module.lstm.weight_ih_l0_reverse")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM backward weight_ih: {}", e)))?;
        let backward_weight_hh = loader.load_component_parameter(component, "module.lstm.weight_hh_l0_reverse")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM backward weight_hh: {}", e)))?;
        let backward_bias_ih = loader.load_component_parameter(component, "module.lstm.bias_ih_l0_reverse")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM backward bias_ih: {}", e)))?;
        let backward_bias_hh = loader.load_component_parameter(component, "module.lstm.bias_hh_l0_reverse")
            .map_err(|e| ferrocarril_core::FerroError::new(format!("CRITICAL: Failed to load LSTM backward bias_hh: {}", e)))?;
        
        // Load into single bidirectional LSTM
        self.lstm.weight_ih_l0 = Parameter::new(forward_weight_ih);
        self.lstm.weight_hh_l0 = Parameter::new(forward_weight_hh);
        self.lstm.bias_ih_l0 = Parameter::new(forward_bias_ih);
        self.lstm.bias_hh_l0 = Parameter::new(forward_bias_hh);
        self.lstm.weight_ih_l0_reverse = Parameter::new(backward_weight_ih);
        self.lstm.weight_hh_l0_reverse = Parameter::new(backward_weight_hh);
        self.lstm.bias_ih_l0_reverse = Parameter::new(backward_bias_ih);
        self.lstm.bias_hh_l0_reverse = Parameter::new(backward_bias_hh);
        
        Ok(())
    }
}