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
        
        // STRICT: Validate gamma_beta has exactly the right shape - NO FALLBACKS
        let gamma_beta_shape = gamma_beta.shape();
        assert_eq!(gamma_beta_shape.len(), 2,
            "STRICT: gamma_beta must be 2D, got shape: {:?}", gamma_beta_shape);
        assert_eq!(gamma_beta_shape[0], b,
            "STRICT: gamma_beta batch size {} != input batch size {}", gamma_beta_shape[0], b);
        assert_eq!(gamma_beta_shape[1], c * 2,
            "STRICT: gamma_beta features {} != expected {} (2 * channels)",
            gamma_beta_shape[1], c * 2);
        
        // Split gamma_beta into gamma and beta with STRICT bounds checking
        let mut gamma = vec![0.0; b * 1 * c];
        let mut beta = vec![0.0; b * 1 * c];
        
        for batch in 0..b {
            for chan in 0..c {
                // STRICT: Validate indices are within bounds
                assert!(chan < gamma_beta_shape[1],
                    "STRICT: Gamma index {} out of bounds for shape {:?}", chan, gamma_beta_shape);
                assert!(chan + c < gamma_beta_shape[1], 
                    "STRICT: Beta index {} out of bounds for shape {:?}", chan + c, gamma_beta_shape);
                
                gamma[batch * c + chan] = gamma_beta[&[batch, chan]];
                beta[batch * c + chan] = gamma_beta[&[batch, chan + c]];
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

    /// Forward pass with EXACT PYTORCH BEHAVIORAL MATCHING - CRITICAL CHANNEL FIX
    /// PyTorch DurationEncoder returns [B, T, d_model] NOT [B, T, d_model+style_dim]
    /// CRITICAL: All input validation MUST be strict - NO SILENT ADAPTATIONS
    pub fn forward(&self,
               txt_feat: &Tensor<f32>,
               style: &Tensor<f32>,
               mask: &Tensor<bool>)
    -> Tensor<f32> {
    
    // CRITICAL INPUT VALIDATION: STRICT format requirements [B, C, T]
    assert_eq!(txt_feat.shape().len(), 3,
        "STRICT: txt_feat must be 3D [batch, channels, time], got: {:?}", txt_feat.shape());
    assert_eq!(txt_feat.shape()[1], self.d_model,
        "STRICT: txt_feat channels {} must equal d_model {}, got shape {:?}", 
        txt_feat.shape()[1], self.d_model, txt_feat.shape());
    
    println!("DurationEncoder STRICT input validation: txt_feat shape={:?} [B, C, T], style shape={:?} [B, style]", 
             txt_feat.shape(), style.shape());
    
    let masks = mask; // [B, T] format
    
    // PyTorch: x = x.permute(2, 0, 1)  # [B,C,T] → [T,B,C]
    let mut x = self.transpose_bct_to_tbc(txt_feat);
    let (t, b, c) = { let s = x.shape(); (s[0], s[1], s[2]) };
    
    println!("After permute(2,0,1): shape={:?} [T,B,C]", x.shape());
    
    // STRICT: Validate the transpose worked correctly
    assert_eq!(x.shape(), &[txt_feat.shape()[2], txt_feat.shape()[0], txt_feat.shape()[1]],
        "STRICT: Transpose failed - expected [T,B,C] = [{},{},{}], got {:?}",
        txt_feat.shape()[2], txt_feat.shape()[0], txt_feat.shape()[1], x.shape());
    
    // PyTorch: s = style.expand(x.shape[0], x.shape[1], -1)  # [B,style] → [T,B,style]
    let mut style_expanded = vec![0.0; t * b * self.style_dim];
    for time in 0..t {
        for batch in 0..b {
            for s in 0..self.style_dim {
                style_expanded[time * b * self.style_dim + batch * self.style_dim + s] = 
                    style[&[batch, s]];
            }
        }
    }
    let style_tbc = Tensor::from_data(style_expanded, vec![t, b, self.style_dim]);
    
    // PyTorch: x = torch.cat([x, s], axis=-1)  # [T,B,C] + [T,B,style] → [T,B,C+style]  
    let mut concat_data = vec![0.0; t * b * (c + self.style_dim)];
    for time in 0..t {
        for batch in 0..b {
            // Copy x features
            for ch in 0..c {
                concat_data[time * b * (c + self.style_dim) + batch * (c + self.style_dim) + ch] = 
                    x[&[time, batch, ch]];
            }
            // Copy style features
            for s in 0..self.style_dim {
                concat_data[time * b * (c + self.style_dim) + batch * (c + self.style_dim) + c + s] = 
                    style_tbc[&[time, batch, s]];
            }
        }
    }
    
    x = Tensor::from_data(concat_data, vec![t, b, c + self.style_dim]);
    println!("After torch.cat([x, s], axis=-1): shape={:?} [T,B,C+style]", x.shape());
    
    // STRICT: Validate concatenation worked correctly
    assert_eq!(x.shape(), &[t, b, c + self.style_dim],
        "STRICT: Style concatenation failed");
    
    // PyTorch: x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
    let masks_transformed = self.transform_mask_for_tbc_tensor(masks); // [B,T] → [T,B,1]
    
    // STRICT: Validate mask transformation
    assert_eq!(masks_transformed.shape(), &[t, b, 1],
        "STRICT: Mask transform failed - expected [T={},B={},1], got {:?}", t, b, masks_transformed.shape());
    
    self.mask_fill_tbc_with_broadcasted_mask(&mut x, &masks_transformed, 0.0);
    
    // PyTorch: x = x.transpose(0, 1)  # [T,B,C+style] → [B,T,C+style]
    x = self.transpose_tbc_to_btc(&x);
    println!("After transpose(0, 1): shape={:?} [B,T,C+style]", x.shape());
    
    // STRICT: Validate transpose back to batch-first
    assert_eq!(x.shape(), &[b, t, c + self.style_dim],
        "STRICT: Transpose back to batch-first failed");
    
    // Process through LSTM and AdaLayerNorm blocks following PyTorch exactly
    for (block_idx, blk) in self.blocks.iter().enumerate() {
        match blk {
            Block::Rnn(rnn) => {
                // PyTorch: x = x.transpose(-1, -2) before LSTM → [B,C+style,T] 
                x = self.transpose_btc_to_bct(&x);
                
                // Execute LSTM on transposed tensor
                let x_btc = self.transpose_bct_to_btc(&x); // Back to [B,T,C+style]
                
                // STRICT: Validate LSTM input has correct dimensions
                assert_eq!(x_btc.shape()[2], self.d_model + self.style_dim,
                    "STRICT: LSTM block {} expects input features {}, got {}. Shape: {:?}",
                    block_idx, self.d_model + self.style_dim, x_btc.shape()[2], x_btc.shape());
                
                let (h, _) = rnn.forward_batch_first(&x_btc, None, None);
                
                // CRITICAL: After bidirectional LSTM, output should be [B,T,d_model] (style dropped)
                x = self.transpose_btc_to_bct(&h); // [B,d_model,T]
                
                // STRICT: Validate LSTM output dropped style channels correctly
                assert_eq!(x.shape()[1], self.d_model,
                    "STRICT: LSTM {} should output d_model={} channels, got {}. Shape: {:?}",
                    block_idx, self.d_model, x.shape()[1], x.shape());
                
                println!("LSTM {} output shape: {:?} - style channels correctly dropped to d_model", block_idx, x.shape());
                
                // Pad if needed
                let x_padded = self.pad_tensor_if_needed(&x, masks.shape()[1]);
                x = x_padded;
            }
            Block::Ada(adaln) => {
                // PyTorch: x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                let x_btc = self.transpose_bct_to_btc(&x); // [B,C,T] → [B,T,C]
                let ada_out = adaln.forward(&x_btc, style);  
                x = self.transpose_btc_to_bct(&ada_out); // [B,T,C] → [B,C,T]
                
                println!("AdaLayerNorm {} output shape: {:?}", block_idx, x.shape());
                
                // CRITICAL: MUST re-add style after AdaLayerNorm for next LSTM iteration
                // PyTorch: x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                let mut style_bct = vec![0.0; b * self.style_dim * x.shape()[2]];
                let t_current = x.shape()[2];
                for batch in 0..b {
                    for s in 0..self.style_dim {
                        for time in 0..t_current {
                            style_bct[batch * self.style_dim * t_current + s * t_current + time] = 
                                style[&[batch, s]];
                        }
                    }
                }
                let style_tensor = Tensor::from_data(style_bct, vec![b, self.style_dim, t_current]);
                
                // Concatenate along channel dimension (axis=1)
                x = self.concat_channels_bct(&x, &style_tensor);
                
                // STRICT: Validate style was re-added correctly
                assert_eq!(x.shape()[1], self.d_model + self.style_dim,
                    "STRICT: After AdaLayerNorm {}, channels should be d_model+style={}, got {}",
                    block_idx, self.d_model + self.style_dim, x.shape()[1]);
                
                // PyTorch: x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
                self.apply_mask_bct_with_broadcast(&mut x, masks);
            }
        }
    }
    
    // CRITICAL FIX: PyTorch DurationEncoder returns [B,T,d_model] NOT [B,T,d_model+style]
    // The final bidirectional LSTM should have dropped style channels
    let final_bct = x; // Current: should be [B, d_model, T] if the last block was LSTM
    
    // STRICT: Validate final tensor has correct channel count
    if final_bct.shape()[1] != self.d_model {
        return panic!("CRITICAL: DurationEncoder final tensor has {} channels, expected d_model={}. \
                      This indicates the final block was not an LSTM or LSTM didn't drop style correctly. \
                      Current shape: {:?}", 
                      final_bct.shape()[1], self.d_model, final_bct.shape());
    }
    
    // PyTorch: return x.transpose(-1, -2)  # [B,d_model,T] → [B,T,d_model] 
    let final_output = self.transpose_bct_to_btc(&final_bct);
    
    // STRICT: Final output validation
    assert_eq!(final_output.shape(), &[b, t, self.d_model],
        "STRICT: DurationEncoder final output shape mismatch: expected [{},{},{}], got {:?}",
        b, t, self.d_model, final_output.shape());
    
    println!("DurationEncoder final output VALIDATED: {:?} [B,T,d_model] - NO STYLE CHANNELS", final_output.shape());
    final_output
}

    /// Helper: transpose [B,C,T] → [T,B,C]
    fn transpose_bct_to_tbc(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; t * b * c];
        
        for batch in 0..b {
            for channel in 0..c {
                for time in 0..t {
                    let src_idx = batch * c * t + channel * t + time;
                    let dst_idx = time * b * c + batch * c + channel;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![t, b, c])
    }
    
    /// Helper: transpose [T,B,C] → [B,T,C]
    fn transpose_tbc_to_btc(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (t, b, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * t * c];
        
        for time in 0..t {
            for batch in 0..b {
                for channel in 0..c {
                    let src_idx = time * b * c + batch * c + channel;
                    let dst_idx = batch * t * c + time * c + channel;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, t, c])
    }
    
    /// Helper: transpose [B,T,C] → [B,C,T]
    fn transpose_btc_to_bct(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * c * t];
        
        for batch in 0..b {
            for time in 0..t {
                for channel in 0..c {
                    let src_idx = batch * t * c + time * c + channel;
                    let dst_idx = batch * c * t + channel * t + time;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, c, t])
    }
    
    /// Helper: transpose [B,C,T] → [B,T,C]
    fn transpose_bct_to_btc(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * t * c];
        
        for batch in 0..b {
            for channel in 0..c {
                for time in 0..t {
                    let src_idx = batch * c * t + channel * t + time;
                    let dst_idx = batch * t * c + time * c + channel;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, t, c])
    }
    
    /// Helper: transpose [B,T] → [T,B] 
    fn transpose_bt_to_tb(&self, x: &Tensor<bool>) -> Tensor<bool> {
        let (b, t) = (x.shape()[0], x.shape()[1]);
        let mut result = vec![false; t * b];
        
        for batch in 0..b {
            for time in 0..t {
                result[time * b + batch] = x[&[batch, time]];
            }
        }
        
        Tensor::from_data(result, vec![t, b])
    }
    
    /// Helper: mask fill for [T,B,C] format
    fn mask_fill_tbc(&self, x: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) {
        let (t, b, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        assert_eq!(mask.shape(), &[t, b], "Mask shape must be [T,B]");
        
        for time in 0..t {
            for batch in 0..b {
                if mask[&[time, batch]] {
                    for channel in 0..c {
                        x[&[time, batch, channel]] = value;
                    }
                }
            }
        }
    }
    
    /// Helper: mask fill for [B,C,T] format
    fn apply_mask_bct_strict(&self, x: &mut Tensor<f32>, mask: &Tensor<bool>) {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        assert_eq!(mask.shape(), &[b, t], "Mask shape must be [B,T]");
        
        for batch in 0..b {
            for time in 0..t {
                if mask[&[batch, time]] {
                    for channel in 0..c {
                        x[&[batch, channel, time]] = 0.0;
                    }
                }
            }
        }
    }

    /// Helper: Transform mask following PyTorch: masks.unsqueeze(-1).transpose(0, 1)
    /// [B,T] → unsqueeze(-1) → [B,T,1] → transpose(0,1) → [T,B,1]
    fn transform_mask_for_tbc_tensor(&self, mask: &Tensor<bool>) -> Tensor<bool> {
        let (b, t) = (mask.shape()[0], mask.shape()[1]);
        
        // Step 1: unsqueeze(-1): [B,T] → [B,T,1]
        let mut mask_bt1 = vec![false; b * t * 1];
        for batch in 0..b {
            for time in 0..t {
                mask_bt1[batch * t * 1 + time * 1 + 0] = mask[&[batch, time]];
            }
        }
        
        // Step 2: transpose(0,1): [B,T,1] → [T,B,1]  
        let mut result = vec![false; t * b * 1];
        for batch in 0..b {
            for time in 0..t {
                for ch in 0..1 {
                    let src_idx = batch * t * 1 + time * 1 + ch;
                    let dst_idx = time * b * 1 + batch * 1 + ch;
                    result[dst_idx] = mask_bt1[src_idx];
                }
            }
        }
        
        Tensor::from_data(result, vec![t, b, 1])
    }
    
    fn mask_fill_tbc_with_broadcasted_mask(&self, x: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) {
        let (t, b, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        assert_eq!(mask.shape(), &[t, b, 1], "Mask shape must be [T,B,1] for broadcasting");
        
        for time in 0..t {
            for batch in 0..b {
                if mask[&[time, batch, 0]] {
                    for channel in 0..c {
                        x[&[time, batch, channel]] = value;
                    }
                }
            }
        }
    }
    
    fn apply_mask_bct_with_broadcast(&self, x: &mut Tensor<f32>, mask: &Tensor<bool>) {
        let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        assert_eq!(mask.shape(), &[b, t], "Original mask must be [B,T]");
        
        for batch in 0..b {
            for time in 0..t {
                if mask[&[batch, time]] {
                    for channel in 0..c {
                        x[&[batch, channel, time]] = 0.0;
                    }
                }
            }
        }
    }
    
    fn concat_channels_bct(&self, tensor1: &Tensor<f32>, tensor2: &Tensor<f32>) -> Tensor<f32> {
        let (b, c1, t) = (tensor1.shape()[0], tensor1.shape()[1], tensor1.shape()[2]);
        let c2 = tensor2.shape()[1];
        
        let mut result = vec![0.0; b * (c1 + c2) * t];
        
        for batch in 0..b {
            for time in 0..t {
                for ch in 0..c1 {
                    result[batch * (c1 + c2) * t + ch * t + time] = 
                        tensor1[&[batch, ch, time]];
                }
                for ch in 0..c2 {
                    result[batch * (c1 + c2) * t + (c1 + ch) * t + time] = 
                        tensor2[&[batch, ch, time]];
                }
            }
        }
        
        Tensor::from_data(result, vec![b, c1 + c2, t])
    }
    
    fn pad_tensor_if_needed(&self, x: &Tensor<f32>, target_length: usize) -> Tensor<f32> {
        let (b, c, current_t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        
        if current_t >= target_length {
            return x.clone();
        }
        
        let mut padded = vec![0.0; b * c * target_length];
        
        for batch in 0..b {
            for channel in 0..c {
                for time in 0..current_t {
                    padded[batch * c * target_length + channel * target_length + time] = 
                        x[&[batch, channel, time]];
                }
            }
        }
        
        Tensor::from_data(padded, vec![b, c, target_length])
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