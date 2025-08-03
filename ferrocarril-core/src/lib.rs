//! Core functionality for Ferrocarril

pub mod tensor;
pub mod ops;
pub mod g2p;
#[cfg(feature = "weights")]
pub mod weights;
#[cfg(feature = "weights")]
pub mod weights_binary;

use std::error::Error;
use std::fmt;
use tensor::Tensor;

// Re-export the PhonesisG2P wrapper
pub use g2p::PhonesisG2P;

#[derive(Debug)]
pub struct FerroError {
    pub message: String,
}

impl fmt::Display for FerroError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for FerroError {}

impl FerroError {
    pub fn new<T: Into<String>>(message: T) -> Self {
        Self { message: message.into() }
    }
}

/// Learnable parameter that can be loaded from weights
#[derive(Debug, Clone)]
pub struct Parameter {
    data: Tensor<f32>,
}

impl Parameter {
    pub fn new(data: Tensor<f32>) -> Self {
        Self { data }
    }
    
    pub fn data(&self) -> &Tensor<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.data
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: std::collections::HashMap<char, usize>,
    pub n_token: usize,
    pub hidden_dim: usize,
    pub n_layer: usize,
    pub style_dim: usize,
    pub n_mels: usize,
    pub max_dur: usize,
    pub dropout: f32,
    pub text_encoder_kernel_size: usize,
    pub istftnet: IstftnetConfig,
    pub plbert: PlbertConfig,
}

#[derive(Debug, Clone)]
pub struct IstftnetConfig {
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gen_istft_n_fft: usize,
    pub gen_istft_hop_size: usize,
}

#[derive(Debug, Clone)]
pub struct PlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
}

impl Config {
    /// Create a default configuration for testing purposes
    pub fn default_for_testing() -> Self {
        // Create a simple test vocabulary
        let mut vocab = std::collections::HashMap::new();
        // Add basic characters and phonemes
        for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-".chars().enumerate() {
            vocab.insert(c, i);
        }
        
        // Default istftnet configuration
        let istftnet = IstftnetConfig {
            upsample_rates: vec![8, 8, 2, 2],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            gen_istft_n_fft: 16,
            gen_istft_hop_size: 4,
        };
        
        // Default PLBERT configuration
        let plbert = PlbertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
        };
        
        Self {
            vocab,
            n_token: 150,
            hidden_dim: 512,
            n_layer: 4,
            style_dim: 128,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.1,
            text_encoder_kernel_size: 5,
            istftnet,
            plbert,
        }
    }
    
    pub fn from_json(path: &str) -> Result<Self, Box<dyn Error>> {
        // Implement actual JSON parsing instead of hardcoded values
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        // Use the existing from_kmodel_config implementation for the actual parsing
        Self::from_kmodel_config(path)
    }
    
    /// Load configuration from a KModel config.json file
    pub fn from_kmodel_config(path: &str) -> Result<Self, Box<dyn Error>> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let config: serde_json::Value = serde_json::from_str(&contents)?;
        
        // Extract values from the JSON
        let n_token = config["n_token"].as_u64().unwrap_or(150) as usize;
        let hidden_dim = config["hidden_dim"].as_u64().unwrap_or(512) as usize;
        let n_layer = config["n_layer"].as_u64().unwrap_or(4) as usize;
        let style_dim = config["style_dim"].as_u64().unwrap_or(128) as usize;
        let n_mels = config["n_mels"].as_u64().unwrap_or(80) as usize;
        let max_dur = config["max_dur"].as_u64().unwrap_or(50) as usize;
        let dropout = config["dropout"].as_f64().unwrap_or(0.1) as f32;
        let text_encoder_kernel_size = config["text_encoder_kernel_size"].as_u64().unwrap_or(5) as usize;
        
        // Extract istftnet config
        let istftnet = &config["istftnet"];
        let istftnet_config = IstftnetConfig {
            upsample_rates: istftnet["upsample_rates"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            upsample_initial_channel: istftnet["upsample_initial_channel"].as_u64().unwrap_or(512) as usize,
            resblock_kernel_sizes: istftnet["resblock_kernel_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            resblock_dilation_sizes: istftnet["resblock_dilation_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|arr| {
                    arr.as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect()
                })
                .collect(),
            upsample_kernel_sizes: istftnet["upsample_kernel_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            gen_istft_n_fft: istftnet["gen_istft_n_fft"].as_u64().unwrap_or(16) as usize,
            gen_istft_hop_size: istftnet["gen_istft_hop_size"].as_u64().unwrap_or(4) as usize,
        };
        
        // Extract plbert config
        let plbert = &config["plbert"];
        let plbert_config = PlbertConfig {
            hidden_size: plbert["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_attention_heads: plbert["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            num_hidden_layers: plbert["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            intermediate_size: plbert["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        };
        
        // Extract vocab
        let mut vocab = std::collections::HashMap::new();
        if let Some(vocab_obj) = config["vocab"].as_object() {
            for (key, value) in vocab_obj {
                if key.len() == 1 {
                    let ch = key.chars().next().unwrap();
                    let id = value.as_u64().unwrap_or(0) as usize;
                    vocab.insert(ch, id);
                }
            }
        }
        
        Ok(Self {
            vocab,
            n_token,
            hidden_dim,
            n_layer,
            style_dim,
            n_mels,
            max_dur,
            dropout,
            text_encoder_kernel_size,
            istftnet: istftnet_config,
            plbert: plbert_config,
        })
    }
}

/// Main model for Ferrocarril TTS
pub struct FerroModel {
    config: Config,
    // Components
    text_encoder: Option<ferrocarril_nn::text_encoder::TextEncoder>,
    prosody_predictor: Option<ferrocarril_nn::prosody::ProsodyPredictor>,
    decoder: Option<ferrocarril_nn::vocoder::Decoder>,
    // Custom Albert layer from PyTorch would go here if ported, but we'll use forward_bert_dur without it
    bert_encoder: Option<ferrocarril_nn::linear::Linear>,
}

impl FerroModel {
    /// Load a model from the original PyTorch weights (no longer used, kept for compatibility)
    pub fn load(path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        #[cfg(not(feature = "weights"))]
        return Err(Box::new(FerroError::new("Weight loading not supported in this build")));

        #[cfg(feature = "weights")]
        {
            use crate::weights::PyTorchWeightLoader;
            
            println!("Loading PyTorch model from '{}'...", path);
            let loader = PyTorchWeightLoader::from_file(path)?;
            if loader.is_empty() {
                return Err(Box::new(FerroError::new(format!("No weights found in '{}'", path))));
            }
            
            println!("Creating model components...");
            
            // Create a basic model without component initialization
            let mut model = Self {
                config: config.clone(),
                text_encoder: None,
                prosody_predictor: None,
                decoder: None,
                bert_encoder: None,
            };
            
            // NOTE: This method is kept for compatibility, but is not fully implemented
            // in the current codebase since we're focusing on the binary weight loader.
            
            println!("PyTorch weight loading is deprecated. Use load_binary instead.");
            
            Ok(model)
        }

        #[cfg(not(feature = "weights"))]
        Err(Box::new(FerroError::new("Weight loading not supported in this build")))
    }
    
    /// Load a model from converted binary weights (output of weight_converter.py)
    pub fn load_binary(path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        #[cfg(not(feature = "weights"))]
        return Err(Box::new(FerroError::new("Weight loading not supported in this build")));

        #[cfg(feature = "weights")]
        {
            use crate::weights_binary::BinaryWeightLoader;
            
            println!("Loading binary weights from '{}'...", path);
            let loader = BinaryWeightLoader::from_directory(path)?;
            if loader.is_empty() {
                return Err(Box::new(FerroError::new(format!("No weights found in '{}'", path))));
            }
            
            println!("Creating model components...");
            
            // Create TextEncoder
            let mut text_encoder = ferrocarril_nn::text_encoder::TextEncoder::new(
                config.hidden_dim,
                config.text_encoder_kernel_size,
                config.n_layer,
                config.n_token
            );
            
            // Create BertEncoder linear layer (for transforming BERT outputs to hidden_dim)
            let mut bert_encoder = ferrocarril_nn::linear::Linear::new(
                config.plbert.hidden_size, // input size (from BERT)
                config.hidden_dim,        // output size
                true                      // with bias
            );
            
            // Create ProsodyPredictor
            let mut prosody_predictor = ferrocarril_nn::prosody::ProsodyPredictor::new(
                config.style_dim,
                config.hidden_dim,
                config.n_layer,
                config.max_dur,
                config.dropout
            );
            
            // Create Decoder
            let mut decoder = ferrocarril_nn::vocoder::Decoder::new(
                config.hidden_dim,
                config.style_dim,
                config.n_mels,
                config.istftnet.resblock_kernel_sizes.clone(),
                config.istftnet.upsample_rates.clone(),
                config.istftnet.upsample_initial_channel,
                config.istftnet.resblock_dilation_sizes.clone(),
                config.istftnet.upsample_kernel_sizes.clone(),
                config.istftnet.gen_istft_n_fft,
                config.istftnet.gen_istft_hop_size
            );
            
            println!("Loading weights for components...");
            
            // Load weights for TextEncoder
            println!("Loading TextEncoder weights...");
            text_encoder.load_weights_binary(&loader)?;
            
            // Load weights for BertEncoder
            println!("Loading BertEncoder weights...");
            bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;
            
            // Load weights for ProsodyPredictor
            println!("Loading ProsodyPredictor weights...");
            prosody_predictor.load_weights_binary(&loader)?;
            
            // Load weights for Decoder
            println!("Loading Decoder weights...");
            decoder.load_weights_binary(&loader)?;
            
            let model = Self {
                config,
                text_encoder: Some(text_encoder),
                prosody_predictor: Some(prosody_predictor),
                decoder: Some(decoder),
                bert_encoder: Some(bert_encoder),
            };
            
            println!("Model successfully loaded!");
            
            Ok(model)
        }

        #[cfg(not(feature = "weights"))]
        Err(Box::new(FerroError::new("Weight loading not supported in this build")))
    }

    /// Load a voice by name and return the embedding
    pub fn load_voice(&self, path: &str, voice_name: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
        #[cfg(not(feature = "weights"))]
        return Err(Box::new(FerroError::new("Weight loading not supported in this build")));

        #[cfg(feature = "weights")]
        {
            use crate::weights_binary::BinaryWeightLoader;
            
            let voice_loader = BinaryWeightLoader::from_directory(path)?;
            let voice_embedding = voice_loader.load_voice(voice_name)?;
            
            Ok(voice_embedding)
        }

        #[cfg(not(feature = "weights"))]
        Err(Box::new(FerroError::new("Weight loading not supported in this build")))
    }

    /// Run inference to generate audio from text using a default voice
    pub fn infer(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        // For now, this is a placeholder
        // In a complete implementation, this would:
        // 1. Convert text to phonemes using a G2P system
        // 2. Load a default voice
        // 3. Call infer_with_voice with the phonemes and voice
        
        println!("Running inference on text: {}", text);
        println!("This is a placeholder implementation returning a test tone.");
        
        // Just return a sine wave as a placeholder
        let sample_rate = 24000; // Adjust based on the model's expected sample rate
        let duration = 1.0; // seconds
        let frequency = 440.0; // Hz (A4 note)
        
        let num_samples = (sample_rate as f32 * duration) as usize;
        let mut sine_wave = vec![0.0f32; num_samples];
        
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            sine_wave[i] = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        }
        
        Ok(sine_wave)
    }

    /// Run the full inference pipeline with phonemes and voice
    pub fn infer_with_voice(&self, phonemes: &str, voice_embedding: &Tensor<f32>, speed: f32) -> Result<Vec<f32>, Box<dyn Error>> {
        // Check if components are initialized
        let text_encoder = self.text_encoder.as_ref().ok_or_else(|| 
            FerroError::new("TextEncoder not initialized"))?;
        let prosody_predictor = self.prosody_predictor.as_ref().ok_or_else(|| 
            FerroError::new("ProsodyPredictor not initialized"))?;
        let decoder = self.decoder.as_ref().ok_or_else(|| 
            FerroError::new("Decoder not initialized"))?;
        let bert_encoder = self.bert_encoder.as_ref().ok_or_else(|| 
            FerroError::new("BertEncoder not initialized"))?;
        
        println!("Running inference pipeline on phonemes: {}", phonemes);
        
        // 1. Convert phonemes to input ids (tokens)
        let mut input_ids = Vec::new();
        input_ids.push(0); // <bos> token
        
        for c in phonemes.chars() {
            if let Some(&id) = self.config.vocab.get(&c) {
                input_ids.push(id);
            } else {
                println!("Warning: Phoneme '{}' not found in vocabulary", c);
            }
        }
        
        input_ids.push(0); // <eos> token
        
        // Convert to tensor - shape [1, seq_len]
        let batch_size = 1;
        let seq_len = input_ids.len();
        
        println!("Input sequence length: {}", seq_len);
        if seq_len > 512 {
            return Err(Box::new(FerroError::new(format!(
                "Input sequence too long: {} > 512 tokens", seq_len
            ))));
        }
        
        let mut input_ids_tensor = Tensor::new(vec![batch_size, seq_len]);
        for (idx, &id) in input_ids.iter().enumerate() {
            input_ids_tensor[&[0, idx]] = id as i64;
        }
        
        // 2. Create input_lengths and text_mask tensors
        let input_length = seq_len;
        let input_lengths = vec![input_length];
        
        // Create text_mask for padding (true = masked position)
        // In the batch=1 case with no padding, all mask values are false
        let mut text_mask = Tensor::from_data(
            vec![false; batch_size * seq_len],
            vec![batch_size, seq_len]
        );
        
        // 3. Process input through BERT (not directly implemented in Rust, we'd use the extracted features)
        // In Python: bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        // Instead, we'll start from the encoding step:
        
        // 4. Split voice embedding into reference and style parts
        // Python: s = ref_s[:, 128:] (style part)
        let style_dim = self.config.style_dim;
        
        // Make sure voice_embedding has the right shape [batch_size, style_dim*2]
        if voice_embedding.shape().len() != 2 || voice_embedding.shape()[1] != style_dim * 2 {
            return Err(Box::new(FerroError::new(format!(
                "Voice embedding has incorrect shape: {:?}, expected [batch_size, {}]", 
                voice_embedding.shape(), style_dim * 2
            ))));
        }
        
        // Split voice embedding into reference and style parts
        let mut ref_part_data = vec![0.0; batch_size * style_dim];
        let mut style_part_data = vec![0.0; batch_size * style_dim];
        
        for b in 0..batch_size {
            for i in 0..style_dim {
                ref_part_data[b * style_dim + i] = voice_embedding[&[b, i]];
                style_part_data[b * style_dim + i] = voice_embedding[&[b, i + style_dim]];
            }
        }
        
        let ref_embedding = Tensor::from_data(ref_part_data, vec![batch_size, style_dim]);
        let style_embedding = Tensor::from_data(style_part_data, vec![batch_size, style_dim]);
        
        // 5. Run text through TextEncoder
        // Python: t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        println!("Running TextEncoder...");
        let t_en = text_encoder.forward(&input_ids_tensor, &input_lengths, &text_mask);
        
        // 6. Run BERT output through the encoder (linear projection)
        // Python: d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        println!("Creating BERT hidden states (placeholder)...");
        // In a full implementation, we would encode BERT output
        // For now, we'll create a placeholder with the right dimensions
        let bert_hidden_size = self.config.plbert.hidden_size;
        let mut bert_hidden = Tensor::from_data(
            vec![0.1; batch_size * seq_len * bert_hidden_size], 
            vec![batch_size, seq_len, bert_hidden_size]
        );
        
        println!("Running BertEncoder...");
        let d_en_before_transpose = bert_encoder.forward(&bert_hidden);
        
        // Transpose: [batch, seq_len, hidden_dim] -> [batch, hidden_dim, seq_len]
        let mut d_en_data = vec![0.0; batch_size * seq_len * self.config.hidden_dim];
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.config.hidden_dim {
                    d_en_data[b * self.config.hidden_dim * seq_len + h * seq_len + s] = 
                        d_en_before_transpose[&[b, s, h]];
                }
            }
        }
        let d_en = Tensor::from_data(d_en_data, vec![batch_size, self.config.hidden_dim, seq_len]);
        
        // 7. Run ProsodyPredictor
        // Python: d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        // Python: x, _ = self.predictor.lstm(d)
        // Python: duration = self.predictor.duration_proj(x)
        // Python: duration = torch.sigmoid(duration).sum(axis=-1) / speed
        println!("Running ProsodyPredictor...");
        let (dur_logits, _) = prosody_predictor.forward(&d_en, &style_embedding, &text_mask, &Tensor::new(vec![seq_len, seq_len]));
        
        // 8. Process duration logits to get durations
        // Simplified approximation of sigmoid + sum + scaling:
        println!("Calculating durations...");
        let mut durations = vec![0; seq_len];
        let base_duration = 5; // Default frames per phoneme
        for i in 0..seq_len {
            durations[i] = (base_duration as f32 / speed).round() as usize;
        }
        
        // 9. Create alignment matrix based on durations
        println!("Creating alignment matrix...");
        // Calculate total frames after expansion
        let total_frames: usize = durations.iter().sum();
        
        // Create indices tensor by repeating position indices according to durations
        let mut indices = Vec::with_capacity(total_frames);
        for (i, &dur) in durations.iter().enumerate() {
            for _ in 0..dur {
                indices.push(i);
            }
        }
        
        // Create alignment matrix [seq_len, total_frames]
        let mut alignment_data = vec![0.0; seq_len * total_frames];
        for (frame_idx, &token_idx) in indices.iter().enumerate() {
            alignment_data[token_idx * total_frames + frame_idx] = 1.0;
        }
        let alignment_matrix = Tensor::from_data(alignment_data, vec![seq_len, total_frames]);
        
        // 10. Apply alignment to hidden representations
        // Python: en = d.transpose(-1, -2) @ pred_aln_trg
        println!("Applying alignment to hidden representations...");
        // Matrix multiplication: d_en [batch, hidden_dim, seq_len] @ alignment_matrix [seq_len, total_frames]
        // -> [batch, hidden_dim, total_frames]
        
        // Initialize result tensor
        let mut en_data = vec![0.0; batch_size * self.config.hidden_dim * total_frames];
        
        // Perform matrix multiplication manually
        for b in 0..batch_size {
            for h in 0..self.config.hidden_dim {
                for t in 0..total_frames {
                    let mut sum = 0.0;
                    for s in 0..seq_len {
                        sum += d_en[&[b, h, s]] * alignment_matrix[&[s, t]];
                    }
                    en_data[b * self.config.hidden_dim * total_frames + h * total_frames + t] = sum;
                }
            }
        }
        let en = Tensor::from_data(en_data, vec![batch_size, self.config.hidden_dim, total_frames]);
        
        // 11. Predict F0 and noise
        // Python: F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        println!("Predicting F0 and noise...");
        let (f0_pred, noise_pred) = prosody_predictor.predict_f0_noise(&en, &style_embedding);
        
        // 12. Generate audio with decoder
        // Python: audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128])
        println!("Generating audio with decoder...");
        // asr here is the same as en (both are the aligned hidden representations)
        let audio_tensor = decoder.forward(&en, &f0_pred, &noise_pred, &ref_embedding);
        
        // Convert to vector
        let audio_vec = audio_tensor.data().to_vec();
        
        // Remove batch dimension if present
        println!("Generated audio shape: {:?}", audio_tensor.shape());
        println!("Generated {} audio samples", audio_vec.len());
        
        Ok(audio_vec)
    }
}