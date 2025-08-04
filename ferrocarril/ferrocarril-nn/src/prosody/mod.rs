//! Prosody predictor using ONLY specialized implementations

use crate::lstm_variants::ProsodyLSTM;  // Specialized LSTM only
use crate::linear_variants::ProjectionLinear; // Specialized Linear only
use crate::conv1d_variants::PredictorConv1d;  // Specialized Conv1d only
use crate::Forward;
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

/// Helper function for applying a mask to a tensor with STRICT validation
fn apply_mask_strict(tensor: &mut Tensor<f32>, mask: &Tensor<bool>, value: f32) -> Result<(), FerroError> {
    // STRICT: Validate exact broadcasting compatibility
    if tensor.shape().len() != 3 || mask.shape().len() != 2 {
        return Err(FerroError::new(format!(
            "STRICT: Invalid tensor dimensions for masking: tensor={:?}, mask={:?}",
            tensor.shape(), mask.shape()
        )));
    }
    
    let (tensor_b, tensor_c, tensor_t) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
    let (mask_b, mask_t) = (mask.shape()[0], mask.shape()[1]);
    
    // STRICT: Exact dimension validation
    assert_eq!(tensor_b, mask_b, 
        "STRICT: Batch dimension mismatch: tensor={}, mask={}", tensor_b, mask_b);
    assert_eq!(tensor_t, mask_t,
        "STRICT: Time dimension mismatch: tensor={}, mask={}", tensor_t, mask_t);
    
    // Apply mask with explicit broadcasting
    for b in 0..tensor_b {
        for t in 0..tensor_t {
            if mask[&[b, t]] {
                for c in 0..tensor_c {
                    tensor[&[b, c, t]] = value;
                }
            }
        }
    }
    
    Ok(())
}

// Forward declaration of submodules
mod duration_encoder;
mod resblk1d;

use duration_encoder::DurationEncoder;
use resblk1d::AdainResBlk1d;

/// Main prosody predictor with specialized implementations only
pub struct ProsodyPredictor {
    // sub-modules
    pub(crate) txt_enc: DurationEncoder,
    pub(crate) dur_lstm: ProsodyLSTM,        // Specialized for duration prediction
    pub(crate) dur_proj: ProjectionLinear,   // Specialized duration projection
    pub(crate) shared_lstm: ProsodyLSTM,     // Specialized for F0/noise prediction
    pub(crate) f0_blocks: Vec<AdainResBlk1d>,
    pub(crate) noise_blocks: Vec<AdainResBlk1d>,
    pub(crate) f0_proj: PredictorConv1d,     // Specialized F0 projection
    pub(crate) noise_proj: PredictorConv1d,  // Specialized noise projection
    
    // configuration
    pub max_dur: usize,
    d_model: usize,
    style_dim: usize,
}

impl ProsodyPredictor {
    pub fn new(style_dim: usize, d_hid: usize, n_layers: usize,
               max_dur: usize, dropout: f32) -> Self {

        let txt_enc = DurationEncoder::new(style_dim, d_hid, n_layers, dropout);

        // Specialized ProsodyLSTMs for 640→512 processing
        let dur_lstm = ProsodyLSTM::new(d_hid + style_dim, d_hid / 2);
        let shared_lstm = ProsodyLSTM::new(d_hid + style_dim, d_hid / 2);

        // Specialized duration projection
        let dur_proj = ProjectionLinear::new(
            d_hid, 
            max_dur, 
            true, 
            crate::linear_variants::ProjectionType::DurationOut
        );

        // F0 / Noise residual towers
        let mut f0_blocks = Vec::new();
        f0_blocks.push(AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout));
        f0_blocks.push(AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout));
        f0_blocks.push(AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout));

        let mut noise_blocks = Vec::new();
        noise_blocks.push(AdainResBlk1d::new(d_hid, d_hid, style_dim, false, dropout));
        noise_blocks.push(AdainResBlk1d::new(d_hid, d_hid / 2, style_dim, true, dropout));
        noise_blocks.push(AdainResBlk1d::new(d_hid / 2, d_hid / 2, style_dim, false, dropout));

        // Specialized Conv1d projections
        let f0_proj = PredictorConv1d::new(d_hid / 2, 1, 1);
        let noise_proj = PredictorConv1d::new(d_hid / 2, 1, 1);

        Self {
            txt_enc, dur_lstm, dur_proj, shared_lstm,
            f0_blocks, noise_blocks, f0_proj, noise_proj,
            max_dur, d_model: d_hid, style_dim
        }
    }

    /// Forward pass with specialized LSTMs
    /// 
    /// FIXED TENSOR SHAPE FLOW (STRICT VALIDATION):
    /// - txt_feat: [B, C, T] → DurationEncoder → [B, T, d_model] (NOT C+style_dim)
    /// - style: [B, style_dim] → expanded to [B, T, style_dim]
    /// - LSTM input: [B, T, d_model+style_dim] → LSTM → [B, T, d_model]
    /// - Duration: [B, T, d_model] → Projection → [B, T, max_dur]
    /// - Energy pooling: [B, T, d_model+style_dim] @ [T, S] → [B, d_model+style_dim, S]
    pub fn forward(&self,
                   txt_feat: &Tensor<f32>,
                   style: &Tensor<f32>,
                   text_mask: &Tensor<bool>,
                   alignment: &Tensor<f32>)
        -> Result<(Tensor<f32>, Tensor<f32>), FerroError> {

        // STRICT INPUT VALIDATION
        assert_eq!(txt_feat.shape().len(), 3,
            "STRICT: txt_feat must be 3D [batch, channels, time], got: {:?}", txt_feat.shape());
        assert_eq!(style.shape().len(), 2,
            "STRICT: style must be 2D [batch, style_dim], got: {:?}", style.shape());
        assert_eq!(text_mask.shape().len(), 2,
            "STRICT: text_mask must be 2D [batch, time], got: {:?}", text_mask.shape());
        assert_eq!(alignment.shape().len(), 2,
            "STRICT: alignment must be 2D [time_in, time_out], got: {:?}", alignment.shape());

        let (batch_size, channels, seq_len) = (txt_feat.shape()[0], txt_feat.shape()[1], txt_feat.shape()[2]);
        
        // STRICT: Validate style dimensions
        assert_eq!(style.shape()[0], batch_size,
            "STRICT: Style batch size {} != input batch size {}", style.shape()[0], batch_size);
        assert_eq!(style.shape()[1], self.style_dim,
            "STRICT: Style dim {} != expected {}", style.shape()[1], self.style_dim);

        // STRICT: Validate text mask compatibility
        assert_eq!(text_mask.shape()[0], batch_size,
            "STRICT: Mask batch size {} != input batch size {}", text_mask.shape()[0], batch_size);
        assert_eq!(text_mask.shape()[1], seq_len,
            "STRICT: Mask time dim {} != input time dim {}", text_mask.shape()[1], seq_len);

        // STRICT: Validate alignment matrix
        assert_eq!(alignment.shape()[0], seq_len,
            "STRICT: Alignment input time {} != txt_feat time {}", alignment.shape()[0], seq_len);

        // 1) Duration encoder processing
        println!("🔍 ProsodyPredictor calling DurationEncoder with STRICT validation...");
        let d_enc = self.txt_enc.forward(txt_feat, style, text_mask);

        // STRICT: Validate DurationEncoder output - MUST be [B, T, d_model] per PyTorch
        let expected_d_enc_shape = vec![batch_size, seq_len, self.d_model];
        assert_eq!(d_enc.shape(), expected_d_enc_shape,
            "CRITICAL: DurationEncoder returned WRONG shape: expected {:?} [B,T,d_model], got {:?}. \
            This indicates DurationEncoder is still leaking style channels.",
            expected_d_enc_shape, d_enc.shape());
        
        println!("✅ DurationEncoder output STRICTLY validated: {:?} [B,T,d_model] - no style channels", d_enc.shape());
        
        // 2) Duration LSTM processing - ADD STYLE BACK FOR LSTM INPUT
        // Expand style to match sequence length: [B, style_dim] → [B, T, style_dim]
        let mut style_expanded = vec![0.0; batch_size * seq_len * self.style_dim];
        for b in 0..batch_size {
            for t in 0..seq_len {
                for s in 0..self.style_dim {
                    style_expanded[b * seq_len * self.style_dim + t * self.style_dim + s] = 
                        style[&[b, s]];
                }
            }
        }
        let style_btc = Tensor::from_data(style_expanded, vec![batch_size, seq_len, self.style_dim]);
        
        // Concatenate d_enc [B,T,d_model] + style [B,T,style_dim] → [B,T,d_model+style_dim]
        let mut dur_lstm_input = vec![0.0; batch_size * seq_len * (self.d_model + self.style_dim)];
        for b in 0..batch_size {
            for t in 0..seq_len {
                // Copy d_model features
                for c in 0..self.d_model {
                    dur_lstm_input[b * seq_len * (self.d_model + self.style_dim) + t * (self.d_model + self.style_dim) + c] = 
                        d_enc[&[b, t, c]];
                }
                // Copy style features  
                for s in 0..self.style_dim {
                    dur_lstm_input[b * seq_len * (self.d_model + self.style_dim) + t * (self.d_model + self.style_dim) + self.d_model + s] = 
                        style_btc[&[b, t, s]];
                }
            }
        }
        let dur_lstm_input_tensor = Tensor::from_data(dur_lstm_input, vec![batch_size, seq_len, self.d_model + self.style_dim]);
        
        // STRICT: Validate LSTM input has exactly the expected dimensions
        assert_eq!(dur_lstm_input_tensor.shape(), &[batch_size, seq_len, self.d_model + self.style_dim],
            "STRICT: Duration LSTM input shape validation failed");
        assert_eq!(dur_lstm_input_tensor.shape()[2], 640,
            "STRICT: Duration LSTM input must have 640 features (512+128), got {}",
            dur_lstm_input_tensor.shape()[2]);
        
        println!("✅ Duration LSTM input STRICTLY validated: {:?} [B,T,d_model+style=640]", dur_lstm_input_tensor.shape());
        
        // Create input_lengths from actual sequence lengths  
        let input_lengths = vec![seq_len; batch_size]; // For now use full length, could be optimized
        
        // Use specialized ProsodyLSTM
        let (dur_out, _) = self.dur_lstm.forward_batch_first_with_lengths(&dur_lstm_input_tensor, &input_lengths, None, None);
        
        // STRICT: Validate LSTM output shape
        assert_eq!(dur_out.shape(), &[batch_size, seq_len, self.d_model],
            "STRICT: Duration LSTM output shape mismatch: expected [{},{},{}], got {:?}",
            batch_size, seq_len, self.d_model, dur_out.shape());

        println!("✅ Duration LSTM output validated: {:?} [B,T,d_model]", dur_out.shape());

        // 3) Duration projection - PYTORCH ALIGNED
        let dur_logits = self.dur_proj.forward(&dur_out);
        
        // STRICT: Validate duration projection  
        assert_eq!(dur_logits.shape(), &[batch_size, seq_len, self.max_dur],
            "STRICT: Duration projection shape mismatch");

        println!("✅ Duration projection output: {:?} [B,T,max_dur]", dur_logits.shape());

        // 4) Energy pooling - USE d_enc WITH STYLE RE-ADDED FOR POOLING
        let d_enc_with_style = dur_lstm_input_tensor.clone(); // [B,T,d_model+style_dim] 
        
        let d_transposed = self.transpose_btc_to_bct_strict(&d_enc_with_style)?;
        println!("Energy pooling input: d.transpose(-1, -2): {:?} [B,d_model+style,T]", d_transposed.shape());
        
        // STRICT: Validate transpose for energy pooling
        assert_eq!(d_transposed.shape(), &[batch_size, self.d_model + self.style_dim, seq_len],
            "STRICT: Energy pooling transpose failed");
        
        // Matrix multiplication: [B, d_model+style, T] @ [T, S] → [B, d_model+style, S]  
        let frames = alignment.shape()[1];
        let en = self.matmul_bct_ts_strict(&d_transposed, alignment)?;
        
        // STRICT: Validate energy pooling output
        assert_eq!(en.shape(), &[batch_size, self.d_model + self.style_dim, frames],
            "STRICT: Energy pooling output shape mismatch: expected [{}, {}, {}], got {:?}",
            batch_size, self.d_model + self.style_dim, frames, en.shape());

        println!("✅ Energy pooling output STRICTLY validated: {:?} [B,d_model+style,S]", en.shape());
        println!("🎯 ProsodyPredictor CORRECTED output shapes - dur_logits: {:?}, en: {:?}", 
                 dur_logits.shape(), en.shape());

        Ok((dur_logits, en))
    }

    /// Predict F0 and noise from encoded features and style using specialized shared LSTM
    pub fn predict_f0_noise(&self, en: &Tensor<f32>, style: &Tensor<f32>)
        -> Result<(Tensor<f32>, Tensor<f32>), FerroError> {
    
        // STRICT: Input validation
        assert_eq!(en.shape().len(), 3,
            "STRICT: Input 'en' must have 3 dimensions [B, C, F], got: {:?}", en.shape());
        assert_eq!(style.shape().len(), 2,
            "STRICT: Style must have 2 dimensions [B, style_dim], got: {:?}", style.shape());

        let (batch_size, channels, frames) = (en.shape()[0], en.shape()[1], en.shape()[2]);
        
        // STRICT: Validate style consistency
        assert_eq!(style.shape()[0], batch_size,
            "STRICT: Style batch size {} != input batch size {}", style.shape()[0], batch_size);
        assert_eq!(style.shape()[1], self.style_dim,
            "STRICT: Style dimension {} != expected {}", style.shape()[1], self.style_dim);

        // Convert from [B, C, F] to [B, F, C] for LSTM processing
        let en_bfc = self.transpose_bcf_to_bfc_strict(en)?;

        // STRICT: Validate transpose result
        assert_eq!(en_bfc.shape(), &[batch_size, frames, channels],
            "STRICT: Transpose BCF→BFC failed");

        // Create input_lengths for LSTM processing
        let input_lengths = vec![frames; batch_size];

        // Process through specialized shared LSTM (FIXED METHOD CALL)
        let (shared_out, _) = self.shared_lstm.forward_batch_first_with_lengths(&en_bfc, &input_lengths, None, None);
        
        // STRICT: Validate LSTM output
        assert_eq!(shared_out.shape(), &[batch_size, frames, self.d_model],
            "STRICT: Shared LSTM output shape mismatch");

        // Convert back to [B, C, F] for F0/noise blocks
        let shared_bct = self.transpose_bfc_to_bcf_strict(&shared_out)?;

        // F0 prediction branch
        let mut f0 = shared_bct.clone();
        for (i, block) in self.f0_blocks.iter().enumerate() {
            f0 = block.forward(&f0, style);
            println!("F0 block {} output shape: {:?}", i, f0.shape());
        }
        
        // Noise prediction branch
        let mut noise = shared_bct.clone();
        for (i, block) in self.noise_blocks.iter().enumerate() {
            noise = block.forward(&noise, style);
            println!("Noise block {} output shape: {:?}", i, noise.shape());
        }

        // Apply projections
        let f0_proj_out = self.f0_proj.forward(&f0);
        let noise_proj_out = self.noise_proj.forward(&noise);

        // STRICT: Validate projection outputs
        // Note: F0 block 1 upsamples by 2x, so final frame count is upsampled_frames
        let upsampled_frames = frames * 2; // After AdainResBlk1d block 1 upsampling
        assert_eq!(f0_proj_out.shape(), &[batch_size, 1, upsampled_frames],
            "STRICT: F0 projection output shape mismatch: expected [batch={}, 1, upsampled_frames={}], got {:?}",
            batch_size, upsampled_frames, f0_proj_out.shape());
        assert_eq!(noise_proj_out.shape(), &[batch_size, 1, upsampled_frames], 
            "STRICT: Noise projection output shape mismatch: expected [batch={}, 1, upsampled_frames={}], got {:?}",
            batch_size, upsampled_frames, noise_proj_out.shape());

        // Squeeze dimension 1: [B, 1, F] → [B, F]
        let f0_squeezed = self.squeeze_dim1_strict(&f0_proj_out)?;
        let noise_squeezed = self.squeeze_dim1_strict(&noise_proj_out)?;

        Ok((f0_squeezed, noise_squeezed))
    }
    
    // STRICT tensor operation helpers

    /// STRICT transpose: [B, C, T] → [B, T, C]
    fn transpose_bct_to_btc_strict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3,
            "STRICT: transpose_bct_to_btc requires 3D tensor, got: {:?}", x.shape());
            
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
        
        Ok(Tensor::from_data(result, vec![b, t, c]))
    }

    /// STRICT transpose: [B, T, C] → [B, C, T]  
    fn transpose_btc_to_bct_strict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3,
            "STRICT: transpose_btc_to_bct requires 3D tensor, got: {:?}", x.shape());
            
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
        
        Ok(Tensor::from_data(result, vec![b, c, t]))
    }

    /// STRICT transpose: [B, C, F] → [B, F, C] for LSTM input
    fn transpose_bcf_to_bfc_strict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3,
            "STRICT: transpose_bcf_to_bfc requires 3D tensor, got: {:?}", x.shape());
            
        let (b, c, f) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * f * c];
        
        for batch in 0..b {
            for chan in 0..c {
                for frame in 0..f {
                    let src_idx = batch * c * f + chan * f + frame;
                    let dst_idx = batch * f * c + frame * c + chan;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Ok(Tensor::from_data(result, vec![b, f, c]))
    }

    /// STRICT transpose: [B, F, C] → [B, C, F] 
    fn transpose_bfc_to_bcf_strict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3,
            "STRICT: transpose_bfc_to_bcf requires 3D tensor, got: {:?}", x.shape());
            
        let (b, f, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mut result = vec![0.0; b * c * f];
        
        for batch in 0..b {
            for frame in 0..f {
                for chan in 0..c {
                    let src_idx = batch * f * c + frame * c + chan;
                    let dst_idx = batch * c * f + chan * f + frame;
                    result[dst_idx] = x.data()[src_idx];
                }
            }
        }
        
        Ok(Tensor::from_data(result, vec![b, c, f]))
    }
    
    /// STRICT matrix multiplication: [B, C, T] @ [T, S] → [B, C, S]
    fn matmul_bct_ts_strict(&self, x: &Tensor<f32>, y: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        let (b, c, t1) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (t2, s) = (y.shape()[0], y.shape()[1]);
        
        // STRICT: Inner dimension validation
        assert_eq!(t1, t2, 
            "STRICT: Inner dimensions must match exactly: x has {}, y has {}", t1, t2);
        
        let mut result = vec![0.0; b * c * s];
        
        for batch in 0..b {
            for chan in 0..c {
                for i in 0..s {
                    let mut sum = 0.0;
                    for j in 0..t1 {
                        let x_idx = batch * c * t1 + chan * t1 + j;
                        let y_idx = j * s + i;
                        sum += x.data()[x_idx] * y.data()[y_idx];
                    }
                    let result_idx = batch * c * s + chan * s + i;
                    result[result_idx] = sum;
                }
            }
        }
        
        Ok(Tensor::from_data(result, vec![b, c, s]))
    }
    
    /// STRICT squeeze operation: [B, 1, F] → [B, F]
    fn squeeze_dim1_strict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, FerroError> {
        assert_eq!(x.shape().len(), 3,
            "STRICT: squeeze_dim1 requires 3D tensor, got: {:?}", x.shape());
        assert_eq!(x.shape()[1], 1, 
            "STRICT: Can only squeeze dimension 1 if size is 1, got size {}", x.shape()[1]);
        
        let b = x.shape()[0];
        let f = x.shape()[2];
        
        let mut result = vec![0.0; b * f];
        
        for batch in 0..b {
            for frame in 0..f {
                result[batch * f + frame] = x[&[batch, 0, frame]];
            }
        }
        
        Ok(Tensor::from_data(result, vec![b, f]))
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for ProsodyPredictor {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError> {
        // Load only specialized implementations
        self.txt_enc.load_weights_binary(loader, component, &format!("{}.text_encoder", prefix))?;
        self.dur_lstm.load_weights_binary(loader, component, &format!("{}.lstm", prefix))?;
        self.shared_lstm.load_weights_binary(loader, component, &format!("{}.shared", prefix))?;
        
        // Load specialized duration projection  
        self.dur_proj.load_weights_binary(loader, component, &format!("{}.duration_proj.linear_layer", prefix))?;
        
        // Load F0/noise blocks and specialized projections
        for i in 0..self.f0_blocks.len() {
            self.f0_blocks[i].load_weights_binary(loader, component, &format!("{}.F0.{}", prefix, i))?;
        }
        
        for i in 0..self.noise_blocks.len() {
            self.noise_blocks[i].load_weights_binary(loader, component, &format!("{}.N.{}", prefix, i))?;
        }
        
        self.f0_proj.load_weights_binary(loader, component, &format!("{}.F0_proj", prefix))?;
        self.noise_proj.load_weights_binary(loader, component, &format!("{}.N_proj", prefix))?;
        
        println!("✅ ProsodyPredictor: All specialized implementations loaded");
        Ok(())
    }
}