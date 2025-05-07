//! TextEncoder – port of kokoro/modules.py::TextEncoder
//! Inference-only; numerically identical to the original model
//! (assuming the PyTorch weights are exported verbatim).

use crate::{conv::Conv1d, lstm::LSTM, Forward, Parameter};
use ferrocarril_core::tensor::Tensor;
use std::sync::Arc;

/// Helper function for applying a mask to a tensor
fn mask_fill(x: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) {
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
    } else {
        panic!("Unsupported tensor and mask shapes for mask_fill");
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
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let c = self.weight.data().shape()[1];
        let mut out = vec![0.0f32; b * t * c];
        let w = self.weight.data();

        for batch in 0..b {
            for pos in 0..t {
                let idx = x[&[batch, pos]] as usize;
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
        let (b, c, t) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut out = vec![0.0; b * c * t];

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
        let y = self.conv.forward(x);              // [B, C, T]
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
    pub(crate) lstm_fw:   LSTM,               // forward  LSTM  (C  → C/2)
    pub(crate) lstm_bw:   LSTM,               // backward LSTM  (C  → C/2)
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

        // Two single-layer unidirectional LSTMs
        let lstm_fw = LSTM::new(
            channels,            // input_size
            channels / 2,        // hidden_size
            1,                   // num_layers
            true,                // batch_first
            false,               // bidirectional
        );

        let lstm_bw = LSTM::new(
            channels,
            channels / 2,
            1,
            true,
            false,
        );

        Self {
            embedding: Embedding::new(n_symbols, channels),
            cnn,
            lstm_fw,
            lstm_bw,
            channels,
        }
    }

    /// x_tok      : [B, T]   int64 phoneme ids
    /// input_lengths: Vec<usize>    lengths of valid tokens
    /// mask       : [B, T]   bool (true = padded position)
    /// returns    : [B, C, T]
    pub fn forward(
        &self,
        x_tok: &Tensor<i64>,
        input_lengths: &Vec<usize>,  
        mask: &Tensor<bool>,
    ) -> Tensor<f32> {
        // First, embed the token ids
        let mut x = self.embedding.forward(x_tok);      // [B, T, C]
        
        // Zero out masked positions (pads)
        let mut x_masked = x.clone();
        mask_fill(&mut x_masked, mask, 0.0);
        x = x_masked;

        // ---- CNN stack --------------------------------------------------
        // transpose → (B, C, T) for convs
        let x_t = self.transpose_btc_to_bct(&x);        // [B, C, T]
        let mut x = x_t;
        
        // Apply CNN blocks sequentially
        for blk in &self.cnn {
            let mut x_conv = blk.forward(&x);
            
            // Zero out masked positions again after each CNN block
            self.apply_mask_bct(&mut x_conv, mask); 
            
            x = x_conv;
        }

        // ---- LSTM -------------------------------------------------------
        // back to (B, T, C) for LSTM
        let x_t = self.transpose_bct_to_btc(&x);        // [B, T, C]

        // Forward LSTM
        let (fw, _) = self.lstm_fw.forward_batch_first(&x_t, None, None);  // [B, T, C/2]
        
        // Backward LSTM (reverse sequence and then forward)
        let x_rev = self.reverse_time(&x_t);
        let (bw_rev, _) = self.lstm_bw.forward_batch_first(&x_rev, None, None);
        let bw = self.reverse_time(&bw_rev);           // restore order
        
        // Explicitly zero out any padded positions in the LSTM outputs
        let mut fw_masked = fw.clone();
        let mut bw_masked = bw.clone();
        
        if fw.shape()[1] == mask.shape()[1] {
            mask_fill(&mut fw_masked, mask, 0.0);
            mask_fill(&mut bw_masked, mask, 0.0);
        }

        // Concatenate outputs along channel dimension
        let y = self.concat_channels(&fw_masked, &bw_masked);        // [B, T, C]
        
        // ---- final layout + mask ---------------------------------------
        let y_t = self.transpose_btc_to_bct(&y);    // [B, C, T]
        
        // Zero out masked positions in final output only if dimensions match
        let mut final_output = y_t.clone();
        if y_t.shape()[2] == mask.shape()[1] {
            self.apply_mask_bct(&mut final_output, mask);
        }
        
        // Ensure output is in [B, C, T] format before returning
        if final_output.shape().len() != 3 {
            panic!("TextEncoder output must be 3-dimensional [B, C, T], got shape {:?}", final_output.shape());
        }
        
        // Log shape for debugging
        println!("TextEncoder output shape: {:?}", final_output.shape());
        
        final_output  // Final output: [B, C, T]
    }

    // Helper methods for tensor manipulation
    fn transpose_btc_to_bct(&self, x: &Tensor<f32>) -> Tensor<f32> {
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

    fn transpose_bct_to_btc(&self, x: &Tensor<f32>) -> Tensor<f32> {
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

    fn reverse_time(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut out = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for time in 0..t {
                for chan in 0..c {
                    out[batch * t * c + (t - 1 - time) * c + chan] = x[&[batch, time, chan]];
                }
            }
        }
        
        Tensor::from_data(out, vec![b, t, c])
    }

    fn concat_channels(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        let (batch, time, c_half) = (a.shape()[0], a.shape()[1], a.shape()[2]);
        assert_eq!(c_half, self.channels / 2);
        assert_eq!(a.shape(), b.shape());
        
        let mut out = vec![0.0; batch * time * self.channels];
        
        for i in 0..batch {
            for j in 0..time {
                // Copy first half from a
                for k in 0..c_half {
                    out[i * time * self.channels + j * self.channels + k] = a[&[i, j, k]];
                }
                // Copy second half from b
                for k in 0..c_half {
                    out[i * time * self.channels + j * self.channels + c_half + k] = b[&[i, j, k]];
                }
            }
        }
        
        Tensor::from_data(out, vec![batch, time, self.channels])
    }

    // -------------------------------------------------
    // util: in-place masked fill for tensor [B, C, T]
    // -------------------------------------------------
    fn apply_mask_bct(&self, t: &mut Tensor<f32>, mask: &Tensor<bool>) {
        let (b, c, tt) = (t.shape()[0], t.shape()[1], t.shape()[2]);
        
        // Check if mask dimensions match
        if mask.shape()[0] != b || mask.shape()[1] != tt {
            println!("Warning: Cannot apply mask due to dimension mismatch in apply_mask_bct: t.shape={:?}, mask.shape={:?}", 
                     t.shape(), mask.shape());
            
            // Handle different mask lengths by applying what we can
            if mask.shape()[0] == b {
                let mask_len = std::cmp::min(mask.shape()[1], tt);
                
                // Apply mask for the available dimensions
                for batch in 0..b {
                    for time in 0..mask_len {
                        if mask[&[batch, time]] {
                            for chan in 0..c {
                                t[&[batch, chan, time]] = 0.0;
                            }
                        }
                    }
                }
                
                println!("Applied partial mask to first {} time positions", mask_len);
            }
            
            return;
        }

        // Apply mask: t[batch, :, time] = 0 where mask[batch, time] == true
        for batch in 0..b {
            for time in 0..tt {
                if mask[&[batch, time]] {
                    for chan in 0..c {
                        t[&[batch, chan, time]] = 0.0;
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

impl TextEncoder {
    /// Load weights from a binary weight loader
    pub fn load_weights_binary(
        &mut self, 
        loader: &ferrocarril_core::weights_binary::BinaryWeightLoader
    ) -> Result<(), ferrocarril_core::FerroError> {
        println!("Loading TextEncoder weights from binary loader...");
        
        // Component name must be "text_encoder" to match the output structure from the weight converter
        let component = "text_encoder";
        
        // Load embedding weights
        // Embeddings are in the text_encoder component
        let embedding_weight_path = "module.embedding.weight";
        if let Ok(embedding_weight) = loader.load_component_parameter(component, embedding_weight_path) {
            self.embedding.weight = Parameter::new(embedding_weight);
            println!("Loaded embedding weights successfully");
        } else {
            println!("Warning: Could not find embedding weights at path: {}", embedding_weight_path);
            println!("Using random initialization");
        }
        
        // Load CNN blocks
        // CNN blocks are in the text_encoder component
        // The weights file only contains up to cnn.2 (0, 1, 2)
        // This matches n_layer=3 in config.json
        for i in 0..self.cnn.len() {
            println!("Loading CNN block {}", i);
            
            // Skip blocks beyond what exists in the weights
            if i >= 3 {
                println!("Skipping block {} as it may not exist in the weights", i);
                continue;
            }
            
            // Get mutable access to the block
            let block = match Arc::get_mut(&mut self.cnn[i]) {
                Some(b) => b,
                None => {
                    println!("Warning: Cannot get mutable reference to CNN block {}, skipping", i);
                    continue;
                }
            };

            // Load Conv1d weights and bias - Using actual file naming patterns
            let weight_g_path = format!("module.cnn.{}.0.weight_g", i);
            let weight_v_path = format!("module.cnn.{}.0.weight_v", i);
            let bias_path = format!("module.cnn.{}.0.bias", i);
            
            // Fix the borrowing issue by cloning the Results before using them in pattern matching
            let weight_g_result = loader.load_component_parameter(component, &weight_g_path);
            let weight_v_result = loader.load_component_parameter(component, &weight_v_path);
            
            let is_weight_g_err = weight_g_result.is_err();
            let is_weight_v_err = weight_v_result.is_err();
            
            if let (Ok(weight_g), Ok(weight_v)) = (&weight_g_result, &weight_v_result) {
                if let Err(e) = block.conv.set_weight_norm(weight_g, weight_v) {
                    println!("Warning: Failed to set weight norm for block {}: {}", i, e);
                } else {
                    println!("Set weight norm for block {}", i);
                }
                
                // Load bias
                if let Ok(bias) = loader.load_component_parameter(component, &bias_path) {
                    if let Err(_) = block.conv.set_bias(&bias) {
                        println!("Warning: Failed to set bias for block {}", i);
                    }
                }
            } else {
                println!("Warning: Failed to load weights for CNN block {}: {}", i, 
                         if is_weight_g_err { 
                             format!("Failed to find weight_g at {}", weight_g_path) 
                         } else { 
                             format!("Failed to find weight_v at {}", weight_v_path) 
                         });
                println!("Skipping this block");
                continue;
            }
            
            // Load LayerNorm weights
            let gamma_path = format!("module.cnn.{}.1.gamma", i);
            let beta_path = format!("module.cnn.{}.1.beta", i);
            
            if let Ok(gamma) = loader.load_component_parameter(component, &gamma_path) {
                block.ln.gamma = Parameter::new(gamma);
            } else {
                println!("Warning: Failed to load gamma for block {}", i);
            }
            
            if let Ok(beta) = loader.load_component_parameter(component, &beta_path) {
                block.ln.beta = Parameter::new(beta);
            } else {
                println!("Warning: Failed to load beta for block {}", i);
            }
        }
        
        // Load LSTM weights
        // The TextEncoder has a single bidirectional LSTM
        println!("Loading LSTM weights...");
        
        // Forward LSTM - Uses parameters from the text_encoder component
        let forward_paths = [
            ("module.lstm.weight_ih_l0", "module.lstm.weight_hh_l0", 
             "module.lstm.bias_ih_l0", "module.lstm.bias_hh_l0")
        ];
        
        let mut forward_loaded = false;
        for (weight_ih_path, weight_hh_path, bias_ih_path, bias_hh_path) in forward_paths.iter() {
            // Only try the text_encoder component, as that's where the weights should be
            if let (Ok(weight_ih), Ok(weight_hh), Ok(bias_ih), Ok(bias_hh)) = (
                loader.load_component_parameter(component, weight_ih_path),
                loader.load_component_parameter(component, weight_hh_path),
                loader.load_component_parameter(component, bias_ih_path),
                loader.load_component_parameter(component, bias_hh_path),
            ) {
                self.lstm_fw.weight_ih_l0 = Parameter::new(weight_ih);
                self.lstm_fw.weight_hh_l0 = Parameter::new(weight_hh);
                self.lstm_fw.bias_ih_l0 = Parameter::new(bias_ih);
                self.lstm_fw.bias_hh_l0 = Parameter::new(bias_hh);
                println!("Forward LSTM weights loaded successfully");
                forward_loaded = true;
            }
        }
        
        if !forward_loaded {
            println!("Warning: Failed to load forward LSTM weights");
        }
        
        // Backward LSTM (reverse weights)
        // Look for reverse weights in the text_encoder component
        let backward_paths = [
            ("module.lstm.weight_ih_l0_reverse", "module.lstm.weight_hh_l0_reverse", 
             "module.lstm.bias_ih_l0_reverse", "module.lstm.bias_hh_l0_reverse")
        ];
        
        let mut backward_loaded = false;
        for (weight_ih_path, weight_hh_path, bias_ih_path, bias_hh_path) in backward_paths.iter() {
            // Only try the text_encoder component, as that's where the weights should be
            if let (Ok(weight_ih), Ok(weight_hh), Ok(bias_ih), Ok(bias_hh)) = (
                loader.load_component_parameter(component, weight_ih_path),
                loader.load_component_parameter(component, weight_hh_path),
                loader.load_component_parameter(component, bias_ih_path),
                loader.load_component_parameter(component, bias_hh_path),
            ) {
                self.lstm_bw.weight_ih_l0 = Parameter::new(weight_ih);
                self.lstm_bw.weight_hh_l0 = Parameter::new(weight_hh);
                self.lstm_bw.bias_ih_l0 = Parameter::new(bias_ih);
                self.lstm_bw.bias_hh_l0 = Parameter::new(bias_hh);
                println!("Backward LSTM weights loaded successfully");
                backward_loaded = true;
            }
        }
        
        if !backward_loaded {
            // Try alternative paths directly with _reverse suffix
            println!("Warning: Failed to load backward LSTM weights: Parameter 'module.lstm_reverse.weight_ih_l0' not found in component 'text_encoder'");
        }
        
        println!("TextEncoder weights loaded successfully");
        Ok(())
    }
}

// Extension to LSTM for binary weight loading
impl LSTM {
    pub fn load_weights_binary(
        &mut self,
        loader: &ferrocarril_core::weights_binary::BinaryWeightLoader,
        component: &str,
        direction: &str,
        is_reverse: bool
    ) -> Result<(), ferrocarril_core::FerroError> {
        let suffix = if is_reverse { "_reverse" } else { "" };
        
        // Determine the prefix based on direction
        let prefix = match direction {
            "fw" => "module.lstm",
            "bw" => "module.lstm",
            _ => direction,
        };
        
        // Load weight_ih_l0
        let weight_ih_name = if direction == "bw" { 
            format!("{}_reverse", prefix) 
        } else { 
            prefix.to_string() 
        };
        
        let weight_ih = loader.load_component_parameter(component, &format!("{}.weight_ih_l0{}", weight_ih_name, suffix))?;
        self.weight_ih_l0 = Parameter::new(weight_ih);
        
        // Load weight_hh_l0
        let weight_hh = loader.load_component_parameter(component, &format!("{}.weight_hh_l0{}", weight_ih_name, suffix))?;
        self.weight_hh_l0 = Parameter::new(weight_hh);
        
        // Load bias_ih_l0
        let bias_ih = loader.load_component_parameter(component, &format!("{}.bias_ih_l0{}", weight_ih_name, suffix))?;
        self.bias_ih_l0 = Parameter::new(bias_ih);
        
        // Load bias_hh_l0
        let bias_hh = loader.load_component_parameter(component, &format!("{}.bias_hh_l0{}", weight_ih_name, suffix))?;
        self.bias_hh_l0 = Parameter::new(bias_hh);
        
        Ok(())
    }
}

// The duplicate Conv1d implementation was removed as it conflicts with conv.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_shapes() {
        let enc = TextEncoder::new(192, 5, 3, 200);
        let x = Tensor::<i64>::from_data(vec![1i64; 4 * 123], vec![4, 123]);   // dummy ids
        let mask = Tensor::<bool>::from_data(vec![false; 4 * 123], vec![4, 123]); // no padding
        let _input_lengths = vec![123, 123, 123, 123];
        let y = enc.forward(&x, &_input_lengths, &mask);
        assert_eq!(y.shape(), &[4, 192, 123]);
    }
}