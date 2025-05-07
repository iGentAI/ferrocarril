//! Prosody predictor (duration, F0, noise) – Rust port of kokoro implementation
//!
//! Entry point: ProsodyPredictor::forward()
//!
//! During inference we typically work on one utterance per call, therefore all
//! tensors are assumed `batch = 1`.
//! If you want batched inference just wrap the outer loop; all operators are
//! shape-agnostic w.r.t. batch dimension.

use crate::{
    lstm::LSTM,
    linear::Linear,
    conv::Conv1d,
    Forward,
};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

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
        panic!("Unsupported tensor and mask shapes for mask_fill. Expected compatible dimensions for x: {:?}, mask: {:?}", 
               x.shape(), mask.shape());
    }
}

// Forward declaration of submodules
mod duration_encoder;
mod resblk1d;

use duration_encoder::DurationEncoder;
use resblk1d::AdainResBlk1d;

/// Main prosody predictor
pub struct ProsodyPredictor {
    // sub-modules -------------------------------------------------------------
    pub(crate) txt_enc: DurationEncoder,
    pub(crate) dur_lstm: LSTM,
    pub(crate) dur_proj: Linear, // → categorical duration logits
    pub(crate) shared_lstm: LSTM,
    pub(crate) f0_blocks: Vec<AdainResBlk1d>,
    pub(crate) noise_blocks: Vec<AdainResBlk1d>,
    pub(crate) f0_proj: Conv1d,
    pub(crate) noise_proj: Conv1d,
    // configuration -----------------------------------------------------------
    pub max_dur: usize,
    d_model: usize,
    style_dim: usize,
}

impl ProsodyPredictor {
    pub fn new(style_dim: usize, d_hid: usize, n_layers: usize,
               max_dur: usize, dropout: f32) -> Self {

        let txt_enc = DurationEncoder::new(style_dim, d_hid, n_layers, dropout);

        // bidirectional = true doubles the hidden size
        // we feed d_hid+d_style in and get d_hid out (fw + bw)
        let dur_lstm  = LSTM::new(d_hid + style_dim,
                                  d_hid / 2, /*hidden*/
                                  1, /*layers*/
                                  true,/*batch_first*/
                                  true /*bidir*/);

        let dur_proj  = Linear::new(d_hid, max_dur, true);

        let shared_lstm = LSTM::new(d_hid + style_dim,
                                    d_hid / 2,
                                    1, true, true);

        // ------------- F0 / Noise residual towers ---------------------------
        let mut f0_blocks = Vec::new();
        f0_blocks.push(AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout));
        f0_blocks.push(AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout));
        f0_blocks.push(AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout));

        let mut noise_blocks = Vec::new();
        noise_blocks.push(AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout));
        noise_blocks.push(AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout));
        noise_blocks.push(AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout));

        let f0_proj = Conv1d::new(d_hid / 2, 1, 1, 1, 0, 1, 1, true);
        let noise_proj = Conv1d::new(d_hid / 2, 1, 1, 1, 0, 1, 1, true);

        Self {
            txt_enc, dur_lstm, dur_proj, shared_lstm,
            f0_blocks, noise_blocks, f0_proj, noise_proj,
            max_dur, d_model: d_hid, style_dim
        }
    }

    /// Forward pass that returns duration logits (= categorical dist) and the
    /// encoded hidden sequence `en` consumed by the mel-decoder.
    ///
    /// txt_feat      – [B, C, T]  phoneme id's **already embedded** and transposed
    /// style       – [B, style_dim]
    /// text_mask   – [B, T]  bool (true = padded position)
    /// alignment   – [T, S]  attention matrix used for energy pooling
    pub fn forward(&self,
                   txt_feat: &Tensor<f32>,
                   style   : &Tensor<f32>,
                   text_mask: &Tensor<bool>,
                   alignment: &Tensor<f32>)
        -> (Tensor<f32>, Tensor<f32>) {

        // Verify basic input shapes
        if txt_feat.shape().len() != 3 {
            panic!("txt_feat must be 3-dimensional, got shape {:?}", txt_feat.shape());
        }
        
        // Log the input dimensions for debugging
        println!("ProsodyPredictor.forward input: txt_feat shape={:?}, style shape={:?}, mask shape={:?}", 
                 txt_feat.shape(), style.shape(), text_mask.shape());
        
        // In Kokoro, txt_feat is expected to be in shape [B, C, T] (batch, channels, time)
        let (batch_size, channels, seq_len) = (txt_feat.shape()[0], txt_feat.shape()[1], txt_feat.shape()[2]);
        
        // Verify alignment shape
        if alignment.shape().len() != 2 {
            panic!("Alignment tensor must have 2 dimensions, got {:?}", alignment.shape());
        }
        
        // Alignment first dimension must match sequence length
        if alignment.shape()[0] != seq_len {
            panic!("Alignment tensor first dimension ({}) must match sequence length ({})",
                   alignment.shape()[0], seq_len);
        }
        
        // 1) Duration encoder -------------------------------------------------
        // Pass directly to duration encoder which expects [B, C, T] format
        let d_enc = self.txt_enc.forward(txt_feat, style, text_mask);

        // Verify output dimensions from duration encoder
        if d_enc.shape().len() != 3 {
            panic!("Duration encoder output must have 3 dimensions [B, T, C], got shape {:?}", 
                   d_enc.shape());
        }
        
        let (b, t, c) = (d_enc.shape()[0], d_enc.shape()[1], d_enc.shape()[2]);
        let style_dim = style.shape()[1];
        
        // Verify style tensor dimensionality
        if style.shape().len() != 2 || style.shape()[0] != b {
            panic!("Style tensor must have shape [batch, style_dim], got shape {:?}", style.shape());
        }
        
        // Verify style dimension is as expected
        if style_dim != self.style_dim {
            panic!("Style tensor dimension mismatch: expected style_dim={}, got {}", 
                   self.style_dim, style_dim);
        }
        
        // Expand style to match the d_enc time dimension
        // Create [B, T, style_dim] from [B, style_dim]
        let mut style_expanded = vec![0.0; b * t * style_dim];
        
        for batch in 0..b {
            for time in 0..t {
                for s in 0..style_dim {
                    style_expanded[batch * t * style_dim + time * style_dim + s] = style[&[batch, s]];
                }
            }
        }
        let style_exp = Tensor::from_data(style_expanded, vec![b, t, style_dim]);
        
        // Concatenate d_enc and style_exp along feature dimension
        let mut d_concat_data = vec![0.0; b * t * (c + style_dim)];
        
        for batch in 0..b {
            for time in 0..t {
                // First copy d_enc values
                for chan in 0..c {
                    d_concat_data[batch * t * (c + style_dim) + time * (c + style_dim) + chan] = 
                        d_enc[&[batch, time, chan]];
                }
                
                // Then copy style_exp values for this batch and time
                for s in 0..style_dim {
                    d_concat_data[batch * t * (c + style_dim) + time * (c + style_dim) + c + s] = 
                        style_exp[&[batch, time, s]];
                }
            }
        }
        
        // Create tensor with correct shape
        let d_concat = Tensor::from_data(d_concat_data, vec![b, t, c + style_dim]);
        
        // Verify LSTM input dimensions
        let lstm_input_size = self.dur_lstm.get_input_size();
        if d_concat.shape()[2] != lstm_input_size {
            panic!("LSTM input size mismatch: expected {}, got {}", 
                  lstm_input_size, d_concat.shape()[2]);
        }
        
        // 3) LSTM for duration logits -----------------------------------------
        let (dur_out, _) = self.dur_lstm.forward_batch_first(&d_concat, None, None); // [B, T, d_hid]
        let dur_logits = self.dur_proj.forward(&dur_out); // [B, T, max_dur]
        
        // 4) Energy pooling for decoder --------------------------------------
        // Convert d_concat to [B, C+S, T] format for matrix multiplication with alignment
        // In Kokoro: Equivalent to (d.transpose(-1, -2) @ alignment)
        let mut d_concat_bct_data = vec![0.0; b * (c + style_dim) * t];
        for batch in 0..b {
            for time in 0..t {
                for chan in 0..(c + style_dim) {
                    d_concat_bct_data[batch * (c + style_dim) * t + chan * t + time] = 
                        d_concat[&[batch, time, chan]];
                }
            }
        }
        
        // Create tensor with shape [B, C+S, T]
        let d_concat_bct = Tensor::from_data(d_concat_bct_data, vec![b, c + style_dim, t]);
        
        // Perform matrix multiplication: [B, C+S, T] @ [T, S] -> [B, C+S, S]
        let frames = alignment.shape()[1];
        let en = self.matmul_bct_ts(&d_concat_bct, alignment);
        
        // Verify final output shape - should be [B, C+S, frames]
        if en.shape()[0] != b || en.shape()[1] != c + style_dim || en.shape()[2] != frames {
            panic!("Energy pooling output shape mismatch: expected [B={}, C+S={}, frames={}], got {:?}", 
                   b, c + style_dim, frames, en.shape());
        }
        
        (dur_logits, en)
    }

/// Called during inference to predict F0 and noise
    pub fn predict_f0_noise(&self, en: &Tensor<f32>, style: &Tensor<f32>)
        -> (Tensor<f32>, Tensor<f32>) {
    
        // Verify input shapes: en should be [B, F, H]
        if en.shape().len() != 3 {
            panic!("Input tensor 'en' must have 3 dimensions [B, F, H], got: {:?}", en.shape());
        }
        
        // Extract dimensions for clarity
        let (batch_size, frames, hidden_dim) = (en.shape()[0], en.shape()[1], en.shape()[2]);
        println!("EN shape before transpose: {:?}", en.shape());
        
        // Ensure style shape and batch size are correct
        if style.shape().len() != 2 {
            panic!("Style tensor must have 2 dimensions [B, style_dim], got: {:?}", style.shape());
        }
        
        if style.shape()[1] != self.style_dim {
            panic!("Style tensor dimension mismatch: expected style_dim={}, got {}",
                 self.style_dim, style.shape()[1]);
        }
        
        if style.shape()[0] != batch_size {
            panic!("Style tensor batch size mismatch: expected batch_size={}, got {}",
                 batch_size, style.shape()[0]);
        }
        
        // In Kokoro:
        // 1. Input is [B, F, H]
        // 2. We transpose to [B, H, F] for the LSTM
        // 3. We process through LSTM with the correct input shape
        // 4. We transpose back to [B, C, F] for the F0 and noise blocks
        
        // Step 1: Transpose from [B, F, H] to [B, H, F]
        let mut bht_data = vec![0.0; batch_size * hidden_dim * frames];
        for b in 0..batch_size {
            for f in 0..frames {
                for h in 0..hidden_dim {
                    bht_data[b * hidden_dim * frames + h * frames + f] = en[&[b, f, h]];
                }
            }
        }
        
        // Create transposed tensor
        let en_bht = Tensor::from_data(bht_data, vec![batch_size, hidden_dim, frames]);
        println!("EN shape after transpose: {:?}", en_bht.shape());
        
        // Step 2: Convert back to [B, F, H] for LSTM processing (batch-first format)
        let mut en_bfh_data = vec![0.0; batch_size * frames * hidden_dim];
        for b in 0..batch_size {
            for h in 0..hidden_dim {
                for f in 0..frames {
                    en_bfh_data[b * frames * hidden_dim + f * hidden_dim + h] = en_bht[&[b, h, f]];
                }
            }
        }
        
        let en_bfh = Tensor::from_data(en_bfh_data, vec![batch_size, frames, hidden_dim]);
        
        // Step 3: Expand style to match sequence dimension
        let mut style_expanded = vec![0.0; batch_size * frames * self.style_dim];
        for b in 0..batch_size {
            for f in 0..frames {
                for s in 0..self.style_dim {
                    style_expanded[b * frames * self.style_dim + f * self.style_dim + s] = style[&[b, s]];
                }
            }
        }
        
        let style_expanded_tensor = Tensor::from_data(
            style_expanded, 
            vec![batch_size, frames, self.style_dim]
        );
        
        // Step 4: Concatenate along feature dimension for LSTM input
        // Verify LSTM input size is correctly set
        let lstm_input_size = self.shared_lstm.get_input_size();
        
        // Expecting input size to be hidden_dim + style_dim
        if lstm_input_size != hidden_dim + self.style_dim {
            panic!(
                "LSTM input size ({}) doesn't match hidden_dim ({}) + style_dim ({})", 
                lstm_input_size, hidden_dim, self.style_dim
            );
        }
        
        // Concatenate along feature dimension
        let mut concat_data = vec![0.0; batch_size * frames * lstm_input_size];
        
        for b in 0..batch_size {
            for f in 0..frames {
                // Copy hidden features
                for h in 0..hidden_dim {
                    concat_data[b * frames * lstm_input_size + f * lstm_input_size + h] = 
                        en_bfh[&[b, f, h]];
                }
                
                // Copy style features
                for s in 0..self.style_dim {
                    concat_data[b * frames * lstm_input_size + f * lstm_input_size + hidden_dim + s] = 
                        style_expanded_tensor[&[b, f, s]];
                }
            }
        }
        
        let lstm_input = Tensor::from_data(
            concat_data, 
            vec![batch_size, frames, lstm_input_size]
        );
        
        println!("LSTM input shape: {:?}", lstm_input.shape());
        
        // Step 5: Process through shared LSTM
        let (shared_out, _) = self.shared_lstm.forward_batch_first(&lstm_input, None, None);
        
        // Step 6: Transpose output back to [B, C, F] for F0/noise blocks
        let lstm_out_dim = shared_out.shape()[2];
        let mut shared_out_bct_data = vec![0.0; batch_size * lstm_out_dim * frames];
        
        for b in 0..batch_size {
            for f in 0..frames {
                for c in 0..lstm_out_dim {
                    shared_out_bct_data[b * lstm_out_dim * frames + c * frames + f] = 
                        shared_out[&[b, f, c]];
                }
            }
        }
        
        let shared_out_bct = Tensor::from_data(
            shared_out_bct_data, 
            vec![batch_size, lstm_out_dim, frames]
        );
        
        // Step 7: Process through F0 and noise blocks
        let mut f0 = shared_out_bct.clone();
        let mut noise = shared_out_bct.clone();
        
        // Apply F0 blocks
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, style);
        }
        
        // Apply noise blocks
        for block in &self.noise_blocks {
            noise = block.forward(&noise, style);
        }
        
        // Apply projections
        let f0_proj_out = self.f0_proj.forward(&f0);
        let noise_proj_out = self.noise_proj.forward(&noise);
        
        // Squeeze to get final outputs [B, 1, F] -> [B, F]
        let f0_squeezed = self.squeeze_dim1(&f0_proj_out);
        let noise_squeezed = self.squeeze_dim1(&noise_proj_out);
        
        println!("F0 output shape: {:?}, Noise output shape: {:?}", 
                 f0_squeezed.shape(), noise_squeezed.shape());
        
        (f0_squeezed, noise_squeezed)
    }
    
    // Helper functions for tensor operations
    
    // Transpose from [B, T, C] to [B, C, T]
    fn transpose_btc_to_bct(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * c * t];
        
        for batch in 0..b {
            for time in 0..t {
                for chan in 0..c {
                    let src_idx = batch * t * c + time * c + chan;
                    let dst_idx = batch * c * t + chan * t + time;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, c, t])
    }
    
    // Transpose from [B, C, T] to [B, T, C]
    fn transpose_bct_to_btc(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for chan in 0..c {
                for time in 0..t {
                    let src_idx = batch * c * t + chan * t + time;
                    let dst_idx = batch * t * c + time * c + chan;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, t, c])
    }
    
    // Matrix multiplication: [B, C, T] @ [T, S] -> [B, C, S]
    // Now with dimension adaptation for mismatched inner dimensions
    fn matmul_bct_ts(&self, x: &Tensor<f32>, y: &Tensor<f32>) -> Tensor<f32> {
        let (b, c, t1) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (t2, s) = (y.shape()[0], y.shape()[1]);
        
        // Check for dimension mismatch and assert instead of adapting
        // In PyTorch's matmul, inner dimensions must match exactly
        assert_eq!(t1, t2, 
            "Inner dimensions must match for matrix multiplication: x has inner dimension {}, y has inner dimension {}",
            t1, t2);
        
        // Normal matrix multiplication with shapes that match Kokoro exactly
        let mut result = vec![0.0; b * c * s];
        
        // For each batch and channel index
        for batch in 0..b {
            for chan in 0..c {
                // For each column in y
                for i in 0..s {
                    let mut sum = 0.0;
                    // Inner dimension multiply-add
                    for j in 0..t1 {
                        let x_idx = batch * c * t1 + chan * t1 + j;
                        let y_idx = j * s + i;
                        
                        if x_idx < x.data().len() && y_idx < y.data().len() {
                            sum += x.data()[x_idx] * y.data()[y_idx];
                        }
                    }
                    
                    let result_idx = batch * c * s + chan * s + i;
                    if result_idx < result.len() {
                        result[result_idx] = sum;
                    }
                }
            }
        }
        
        // Add validation for result dimensions to match expected shape in Kokoro
        let result_tensor = Tensor::from_data(
            result,
            vec![b, c, s]
        );
        
        // In Kokoro: en = (d.transpose(-1, -2) @ pred_aln_trg)
        // Verify our output matches this shape exactly
        assert_eq!(result_tensor.shape()[0], b,
            "Output batch dimension should be {}, got {}", b, result_tensor.shape()[0]);
        assert_eq!(result_tensor.shape()[1], c,
            "Output channel dimension should be {}, got {}", c, result_tensor.shape()[1]);
        assert_eq!(result_tensor.shape()[2], s,
            "Output sequence dimension should be {}, got {}", s, result_tensor.shape()[2]);
            
        result_tensor
    }
    
    // Remove dimension 1 from a [B, 1, T] tensor to get [B, T]
    fn squeeze_dim1(&self, x: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(x.shape()[1], 1, "Can only squeeze dimension 1 if it has size 1");
        
        let b = x.shape()[0];
        let t = x.shape()[2];
        
        let mut result = vec![0.0; b * t];
        
        for batch in 0..b {
            for time in 0..t {
                result[batch * t + time] = x[&[batch, 0, time]];
            }
        }
        
        Tensor::from_data(result, vec![b, t])
    }
    
    // Inherent method load_weights_binary is replaced with trait implementation
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for ProsodyPredictor {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading ProsodyPredictor weights for {}.{}", component, prefix);
        
        // Load Duration Encoder weights
        println!("Loading Duration Encoder weights...");
        self.txt_enc.load_weights_binary(loader, component, &format!("{}.text_encoder", prefix))?;
        
        // Load LSTM weights for both directions (forward and reverse)
        println!("Loading LSTM weights for durations...");
        
        // Load both forward and reverse weights for dur_lstm
        // Forward weights
        self.dur_lstm.load_weights_binary_with_reverse(
            loader, 
            component, 
            &format!("{}.lstm", prefix), 
            false // forward direction
        )?;
        
        // Reverse weights for bidirectional LSTM
        self.dur_lstm.load_weights_binary_with_reverse(
            loader, 
            component, 
            &format!("{}.lstm", prefix), 
            true // reverse direction
        )?;
        
        // Load both forward and reverse weights for shared_lstm
        println!("Loading LSTM weights for shared network...");
        
        // Forward weights
        self.shared_lstm.load_weights_binary_with_reverse(
            loader, 
            component, 
            &format!("{}.shared", prefix), 
            false // forward direction
        )?;
        
        // Reverse weights for bidirectional LSTM
        self.shared_lstm.load_weights_binary_with_reverse(
            loader, 
            component, 
            &format!("{}.shared", prefix), 
            true // reverse direction
        )?;
        
        // Load Linear projection
        println!("Loading Linear projection weights...");
        self.dur_proj.load_weights_binary(loader, component, &format!("{}.duration_proj.linear_layer", prefix))?;
        
        // Load F0 blocks
        println!("Loading F0 blocks...");
        for i in 0..self.f0_blocks.len() {
            let f0_block_prefix = format!("{}.F0.{}", prefix, i);
            // Get a mutable reference and load weights
            if let Some(block) = self.f0_blocks.get_mut(i) {
                block.load_weights_binary(loader, component, &f0_block_prefix)?;
            }
        }
        
        // Load Noise blocks
        println!("Loading Noise blocks...");
        for i in 0..self.noise_blocks.len() {
            let noise_block_prefix = format!("{}.N.{}", prefix, i);
            // Get a mutable reference and load weights
            if let Some(block) = self.noise_blocks.get_mut(i) {
                block.load_weights_binary(loader, component, &noise_block_prefix)?;
            }
        }
        
        // Load Conv1d projections
        println!("Loading F0 and Noise projections...");
        self.f0_proj.load_weights_binary(loader, component, &format!("{}.F0_proj", prefix))?;
        self.noise_proj.load_weights_binary(loader, component, &format!("{}.N_proj", prefix))?;
        
        println!("ProsodyPredictor weights loaded successfully");
        Ok(())
    }
}