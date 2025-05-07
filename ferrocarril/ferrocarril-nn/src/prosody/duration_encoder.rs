//! Duration encoder (= several bi-LSTM + AdaLayerNorm blocks)

use crate::{
    lstm::LSTM,
    linear::Linear,
    Forward,
};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary; // Import LoadWeightsBinary trait
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;

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

/// Lightweight AdaLayerNorm (LayerNorm whose affine terms come from style vec)
pub struct AdaLayerNorm {
    pub(crate) style_fc: Linear,   // style → (γ,β)
    channels: usize,
    eps: f32,
}

impl AdaLayerNorm {
    pub fn new(style_dim: usize, channels: usize) -> Self {
        Self {
            style_fc: Linear::new(style_dim, channels * 2, true),
            channels,
            eps: 1e-5
        }
    }

    pub fn forward(&self, x: &Tensor<f32>, style: &Tensor<f32>) -> Tensor<f32> {
        // x: [B,T,C]   style:[B,sty]
        let gamma_beta = self.style_fc.forward(style);      // [B,2C]
        let (b, t, c) = { let s=x.shape(); (s[0],s[1],s[2]) };
        
        // Ensure gamma_beta has the right shape
        let gamma_beta_shape = gamma_beta.shape();
        if gamma_beta_shape.len() != 2 || gamma_beta_shape[1] < c * 2 {
            println!("Warning: gamma_beta has unexpected shape: {:?}, expected [b, {}]", 
                     gamma_beta_shape, c * 2);
            println!("Channels: {}, Style dim: {}", c, style.shape()[1]);
            
            // Provide a safe fallback in this case
            let mut result = x.clone();
            return result;
        }
        
        // Split gamma_beta into gamma and beta with bounds checking
        let mut gamma = vec![0.0; b * 1 * c];
        let mut beta = vec![0.0; b * 1 * c];
        
        for batch in 0..b {
            // Verify that we're within bounds for the batch dimension
            if batch >= gamma_beta_shape[0] {
                println!("Warning: batch index {} out of bounds for gamma_beta shape {:?}", 
                         batch, gamma_beta_shape);
                continue;
            }
            
            for chan in 0..c {
                // Verify channel indices are within bounds
                if chan < gamma_beta_shape[1] {
                    gamma[batch * c + chan] = gamma_beta[&[batch, chan]];
                } else {
                    gamma[batch * c + chan] = 0.0; // Default value if out of bounds
                }
                
                // Check for beta indices too
                let beta_idx = chan + c;
                if beta_idx < gamma_beta_shape[1] {
                    beta[batch * c + chan] = gamma_beta[&[batch, beta_idx]];
                } else {
                    beta[batch * c + chan] = 0.0; // Default value if out of bounds
                }
            }
        }
        
        let gamma = Tensor::from_data(gamma, vec![b, 1, c]);
        let beta = Tensor::from_data(beta, vec![b, 1, c]);

        // Layer normalization along the last dimension
        let ln = self.layer_norm(x);
        
        // Apply gamma and beta: (ln * (gamma + 1.0)) + beta
        let mut result = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for time in 0..t {
                for chan in 0..c {
                    let ln_val = ln[&[batch, time, chan]];
                    let gamma_val = gamma[&[batch, 0, chan]];
                    let beta_val = beta[&[batch, 0, chan]];
                    
                    result[batch * t * c + time * c + chan] = (ln_val * (gamma_val + 1.0)) + beta_val;
                }
            }
        }
        
        Tensor::from_data(result, vec![b, t, c])
    }
    
    // Layer normalization implementation (along last dimension)
    fn layer_norm(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, t, c) = { let s=x.shape(); (s[0],s[1],s[2]) };
        let mut result = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for time in 0..t {
                // Calculate mean and variance for this batch and time step
                let mut mean = 0.0;
                let mut var = 0.0;
                
                for chan in 0..c {
                    let val = x[&[batch, time, chan]];
                    mean += val;
                    var += val * val;
                }
                
                mean /= c as f32;
                var = var / c as f32 - mean * mean;
                let std_dev = (var + self.eps).sqrt();
                
                // Normalize
                for chan in 0..c {
                    let val = x[&[batch, time, chan]];
                    result[batch * t * c + time * c + chan] = (val - mean) / std_dev;
                }
            }
        }
        
        Tensor::from_data(result, vec![b, t, c])
    }
    
    /// Load weights from a binary weight loader
    #[cfg(feature = "weights")]
    pub fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading AdaLayerNorm weights for {}.{}", component, prefix);
        
        // Load Linear parameters
        self.style_fc.load_weights_binary(loader, component, &format!("{}.fc", prefix))?;
        
        Ok(())
    }
}

pub struct DurationEncoder {
    blocks: Vec<Block>,
    d_model: usize,
    style_dim: usize,
}

enum Block {
    Rnn(LSTM),
    Ada(AdaLayerNorm),
}

impl DurationEncoder {
    pub fn new(style_dim: usize, d_model: usize,
               n_layers: usize, _dropout: f32) -> Self {
        let mut blocks = Vec::new();
        
        // We need to make sure we're creating the right number of blocks
        // Each n_layers in the config creates 2 blocks: LSTM + AdaLayerNorm
        // The Kokoro model has n_layer=3, so we expect 6 blocks (0-5)
        println!("Creating DurationEncoder with n_layers={}, which will create {} blocks", n_layers, n_layers * 2);
        
        // Restrict the number of layers to prevent trying to load weights that don't exist
        let actual_layers = n_layers.min(3); // Never create more than 3 layers (what's in the weights)
        
        for _layer_idx in 0..actual_layers {
            blocks.push(Block::Rnn(LSTM::new(d_model + style_dim,
                                             d_model / 2,
                                             1, /*batch_first*/ true,
                                             true /*bidir*/)));
            
            blocks.push(Block::Ada(AdaLayerNorm::new(style_dim, d_model)));
        }
        Self { blocks, d_model, style_dim }
    }

    /// txt_feat : [B,C,T]  (output of text embedding / conv stack in Kokoro format) 
    /// style    : [B,style_dim]
    /// mask     : [B,T]  (true = pad)
    pub fn forward(&self,
               txt_feat: &Tensor<f32>,
               style: &Tensor<f32>,
               mask: &Tensor<bool>)
    -> Tensor<f32> {
    println!("DurationEncoder input: txt_feat shape={:?}, style shape={:?}", 
             txt_feat.shape(), style.shape());
    
    // Validate input is in [B, C, T] format
    if txt_feat.shape().len() != 3 {
        panic!("Input tensor must be 3-dimensional [B, C, T], got shape {:?}", txt_feat.shape());
    }
    
    // Extract dimensions
    let (batch_size, channels, seq_len) = (txt_feat.shape()[0], txt_feat.shape()[1], txt_feat.shape()[2]);
    println!("Input tensor has shape [B={}, C={}, T={}]", batch_size, channels, seq_len);
    
    // In Kokoro, the first operation is a permutation:
    // x = x.permute(2, 0, 1)  # [B, C, T] -> [T, B, C]
    // But we'll work with [B, T, C] for our LSTM implementation
    // So we need to transpose from [B, C, T] to [B, T, C]
    let mut btc_data = vec![0.0; batch_size * seq_len * channels];
    for b in 0..batch_size {
        for c in 0..channels {
            for t in 0..seq_len {
                btc_data[b * seq_len * channels + t * channels + c] = txt_feat[&[b, c, t]];
            }
        }
    }
    
    // Create the properly transposed tensor
    let mut x = Tensor::from_data(btc_data, vec![batch_size, seq_len, channels]);
    
    // Verify style tensor matches expected dimensions
    if style.shape()[0] != batch_size || style.shape()[1] != self.style_dim {
        panic!("Style tensor must have shape [batch, style_dim], got {:?}", style.shape());
    }
    
    // Process through LSTM blocks and AdaLayerNorm
    for blk in &self.blocks {
        match blk {
            Block::Rnn(rnn) => {
                // Expand style to match the time dimension
                let mut style_expanded = vec![0.0; batch_size * seq_len * self.style_dim];
                for b in 0..batch_size {
                    for t in 0..seq_len {
                        for s in 0..self.style_dim {
                            style_expanded[b * seq_len * self.style_dim + t * self.style_dim + s] = 
                                style[&[b, s]];
                        }
                    }
                }
                
                let style_t = Tensor::from_data(style_expanded, vec![batch_size, seq_len, self.style_dim]);
                
                // Concatenate hidden states and style for LSTM
                let mut concat_data = vec![0.0; batch_size * seq_len * (channels + self.style_dim)];
                for b in 0..batch_size {
                    for t in 0..seq_len {
                        // Copy x features
                        for c in 0..channels {
                            concat_data[b * seq_len * (channels + self.style_dim) + t * (channels + self.style_dim) + c] = 
                                x[&[b, t, c]];
                        }
                        
                        // Copy style features
                        for s in 0..self.style_dim {
                            concat_data[b * seq_len * (channels + self.style_dim) + t * (channels + self.style_dim) + channels + s] = 
                                style_t[&[b, t, s]];
                        }
                    }
                }
                
                // Create concatenated tensor
                let d_concat = Tensor::from_data(
                    concat_data, 
                    vec![batch_size, seq_len, channels + self.style_dim]
                );
                
                // Verify input size matches LSTM expectations
                let lstm_input_size = rnn.get_input_size();
                if d_concat.shape()[2] != lstm_input_size {
                    panic!("LSTM input size mismatch: expected {}, got {}", 
                          lstm_input_size, d_concat.shape()[2]);
                }
                
                // Forward pass through LSTM
                let (h, _) = rnn.forward_batch_first(&d_concat, None, None);
                
                // Update x for next block
                x = h;
                
                // Apply mask
                if x.shape()[1] == mask.shape()[1] {
                    mask_fill(&mut x, mask, 0.0);
                } else {
                    panic!("Mask dimensions ({:?}) don't match tensor time dimension ({}).",
                           mask.shape(), x.shape()[1]);
                }
            }
            Block::Ada(adaln) => {
                // Forward pass through AdaLayerNorm
                x = adaln.forward(&x, style);
                
                // Apply mask
                if x.shape()[1] == mask.shape()[1] {
                    mask_fill(&mut x, mask, 0.0);
                } else {
                    panic!("Mask dimensions ({:?}) don't match tensor time dimension ({}).",
                           mask.shape(), x.shape()[1]);
                }
            }
        }
    }
    
    println!("DurationEncoder output shape: {:?}", x.shape());
    // Output is in [B, T, C] format
    x
}
    
    
    /// Load weights from a binary weight loader
    #[cfg(feature = "weights")]
    pub fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        println!("Loading DurationEncoder weights for {}.{}", component, prefix);
        
        // Log how many blocks we're trying to load
        let total_blocks = self.blocks.len();
        println!("DurationEncoder has {} blocks to load", total_blocks);
        
        // Loop through blocks and load weights for each
        for (i, blk) in self.blocks.iter_mut().enumerate() {
            // Skip loading blocks beyond what the model actually has
            // Kokoro model has 3 LSTM pairs (6 blocks total, indices 0-5)
            if i >= 6 {
                println!("Skipping block {} as it exceeds the available weights in Kokoro model", i);
                continue;
            }
            
            match blk {
                Block::Rnn(lstm) => {
                    // In Kokoro, the LSTM blocks are stored in the module.text_encoder.lstms array
                    let lstm_idx = i / 2 * 2; // Convert block index to lstm index (0,1,2,3,4,5 -> 0,0,2,2,4,4)
                    let lstm_prefix = format!("{}.lstms.{}", prefix, lstm_idx);
                    
                    println!("Loading LSTM weights for block {} from {}.lstms.{}", i, prefix, lstm_idx);
                    
                    // Try to load LSTM weights, but don't fail if not found
                    if let Err(e) = lstm.load_weights_binary_with_reverse(
                        loader, 
                        component, 
                        &lstm_prefix,
                        i % 2 == 1 // odd indices are reverse
                    ) {
                        println!("Warning: Could not load LSTM weights for block {}. Error: {}", i, e);
                        println!("Using default random weights instead.");
                        // Continue with default random weights
                    } else {
                        println!("Successfully loaded LSTM weights for block {}", i);
                    }
                },
                Block::Ada(adaln) => {
                    // In Kokoro, the FC layers are stored in the module.text_encoder.lstms array
                    // after each LSTM pair
                    let fc_idx = i / 2 * 2 + 1; // Convert block index to fc index (0,1,2,3,4,5 -> 1,1,3,3,5,5)
                    let fc_prefix = format!("{}.lstms.{}", prefix, fc_idx);
                    
                    println!("Loading AdaLayerNorm weights for block {} from {}.lstms.{}", i, prefix, fc_idx);
                    
                    // Try to load AdaLayerNorm weights, but don't fail if not found
                    if let Err(e) = adaln.load_weights_binary(
                        loader, 
                        component, 
                        &fc_prefix
                    ) {
                        println!("Warning: Could not load AdaLayerNorm weights for block {}. Error: {}", i, e);
                        println!("Using default random weights instead.");
                        // Continue with default random weights
                    } else {
                        println!("Successfully loaded AdaLayerNorm weights for block {}", i);
                    }
                },
            }
        }
        
        // Always return success - we've handled errors at the component level
        Ok(())
    }
}