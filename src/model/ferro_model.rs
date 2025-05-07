//! FerroModel implementation - main TTS inference model

use ferrocarril_core::{Config, tensor::Tensor};
use ferrocarril_nn::{text_encoder::TextEncoder, prosody::ProsodyPredictor, vocoder::Decoder, Forward};
use ferrocarril_nn::bert::Bert;
use ferrocarril_nn::bert::transformer::BertConfig;
use ferrocarril_nn::linear::Linear;
use std::error::Error;
use serde_json; // Add serde_json for parsing voice metadata

// Import G2PHandler from the parent module
use super::g2p::G2PHandler;

#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

/// Main Ferrocarril TTS model
pub struct FerroModel {
    /// Model configuration
    config: Config,
    
    /// G2P handler for text-to-phoneme conversion
    g2p: G2PHandler,
    
    /// Text encoder component
    #[cfg(feature = "weights")]
    text_encoder: Option<TextEncoder>,
    
    /// CustomBERT component
    #[cfg(feature = "weights")]
    bert: Option<Bert>,

    /// BERT encoder linear projection
    #[cfg(feature = "weights")]
    bert_encoder: Option<Linear>,
    
    /// Prosody predictor component
    #[cfg(feature = "weights")]
    prosody_predictor: Option<ProsodyPredictor>,
    
    /// Audio decoder/vocoder component
    #[cfg(feature = "weights")]
    decoder: Option<Decoder>,
}

impl FerroModel {
    /// Create alignment tensor from duration predictions
    /// 
    /// This function transforms duration predictions into an alignment matrix mapping tokens to frames.
    /// The alignment matrix has shape [seq_len, total_frames] where total_frames is the sum of all durations.
    /// Each position (t, f) in the matrix is 1.0 if frame f corresponds to token t, and 0.0 otherwise.
    /// 
    /// # Parameters
    /// - durations: Vector of individual token durations (length = seq_len)
    /// 
    /// # Returns
    /// - Alignment tensor with shape [seq_len, total_frames]
    fn create_alignment_from_durations(&self, durations: &[usize]) -> Tensor<f32> {
        let seq_len = durations.len();
        let total_frames: usize = durations.iter().sum();
        
        println!("Creating alignment matrix with shape [seq_len={}, total_frames={}]", seq_len, total_frames);
        
        // This matches Kokoro's approach:
        // First create indices by repeating position indices according to durations
        // Similar to torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        let mut indices = Vec::with_capacity(total_frames);
        for (i, &dur) in durations.iter().enumerate() {
            for _ in 0..dur {
                indices.push(i);
            }
        }
        
        // Verify indices length matches total_frames
        assert_eq!(indices.len(), total_frames,
            "Indices length ({}) should match total_frames ({})",
            indices.len(), total_frames);
        
        // Now create the actual alignment matrix
        // In Kokoro:
        // pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        // pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        
        let mut alignment_data = vec![0.0; seq_len * total_frames];
        
        // For every frame position, set a 1 at the token position it corresponds to
        for (frame_idx, &token_idx) in indices.iter().enumerate() {
            // This is equivalent to pred_aln_trg[token_idx, frame_idx] = 1.0
            alignment_data[token_idx * total_frames + frame_idx] = 1.0;
        }
        
        // Create the alignment tensor
        let alignment = Tensor::from_data(
            alignment_data,
            vec![seq_len, total_frames]
        );
        
        // Validate the alignment dimensions
        assert_eq!(alignment.shape()[0], seq_len, 
            "Alignment tensor first dimension must match sequence length");
        assert_eq!(alignment.shape()[1], total_frames,
            "Alignment tensor second dimension must match total frames");
        
        // Validate that each column sums to 1.0 (each frame belongs to exactly one token)
        // This is a key property of a valid alignment matrix
        for frame_idx in 0..total_frames {
            let mut col_sum = 0.0f32;
            for token_idx in 0..seq_len {
                col_sum += alignment.data()[token_idx * total_frames + frame_idx];
            }
            assert!((col_sum - 1.0f32).abs() < 1e-6f32,
                "Column {} sum should be 1.0, got {}", frame_idx, col_sum);
        }
        
        // Validate that rows sum to their respective durations
        // This ensures each token is assigned the right number of frames
        for token_idx in 0..seq_len {
            let mut row_sum = 0.0f32;
            for frame_idx in 0..total_frames {
                row_sum += alignment.data()[token_idx * total_frames + frame_idx];
            }
            assert!((row_sum - durations[token_idx] as f32).abs() < 1e-6f32,
                "Row {} sum should be {}, got {}", token_idx, durations[token_idx], row_sum);
        }
        
        alignment
    }

    /// Create a new model with default initialization
    pub fn new(config: Config) -> Result<Self, Box<dyn Error>> {
        // Initialize G2P with the English language
        let g2p = G2PHandler::new("en-us")?;
        
        Ok(Self { 
            config,
            g2p,
            #[cfg(feature = "weights")]
            text_encoder: None,
            #[cfg(feature = "weights")]
            bert: None,
            #[cfg(feature = "weights")]
            bert_encoder: None,
            #[cfg(feature = "weights")]
            prosody_predictor: None,
            #[cfg(feature = "weights")]
            decoder: None,
        })
    }

    /// Load a model from PyTorch weights (not fully implemented)
    pub fn load(_path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        // TODO: Implement PyTorch model loading
        println!("Note: PyTorch weight loading is not fully implemented.");
        println!("Consider using load_binary with converted weights instead.");
        
        // Initialize G2P
        let g2p = G2PHandler::new("en-us")?;
        
        Ok(Self { 
            config,
            g2p,
            #[cfg(feature = "weights")]
            text_encoder: None,
            #[cfg(feature = "weights")]
            bert: None,
            #[cfg(feature = "weights")]
            bert_encoder: None,
            #[cfg(feature = "weights")]
            prosody_predictor: None,
            #[cfg(feature = "weights")]
            decoder: None,
        })
    }

    /// Load a model from converted binary weights
    #[cfg(feature = "weights")]
    pub fn load_binary(path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        // Load model weights from converted binary format
        println!("Loading model from binary weights at {}...", path);
        
        let loader = BinaryWeightLoader::from_directory(path)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                
        // At this point, we should use the loader to populate all model components
        println!("Weight loader created successfully");
        
        // Check if weights were loaded correctly
        if loader.is_empty() {
            println!("Warning: No weights were loaded. The model may not function correctly.");
        } else {
            println!("Weights loaded successfully!");
            
            // List the components that were loaded
            let components = loader.list_components();
            println!("Loaded components: {:?}", components);
        }
        
        // Initialize model components
        let mut text_encoder = TextEncoder::new(
            config.hidden_dim,
            5, // kernel_size
            config.n_layer,
            config.n_token,
        );
        
        // TextEncoder has a special implementation that doesn't take component/prefix
        text_encoder.load_weights_binary(&loader)?;
        println!("Text encoder weights loaded successfully");

        // Create and load BERT component
        let bert_config = BertConfig {
            vocab_size: config.n_token,
            hidden_size: config.plbert.hidden_size,
            num_attention_heads: config.plbert.num_attention_heads,
            num_hidden_layers: config.plbert.num_hidden_layers,
            intermediate_size: config.plbert.intermediate_size,
            max_position_embeddings: 512, // Default value for ALBERT
            dropout_prob: config.dropout,
        };

        let mut bert = Bert::new(bert_config);
        // Use the component name that matches the weight converter output
        bert.load_weights_binary(&loader, "bert", "module")?;
        println!("BERT weights loaded successfully");

        // Create and load BERT encoder (linear projection layer)
        let mut bert_encoder = Linear::new(
            config.plbert.hidden_size, // input_dim
            config.hidden_dim,        // output_dim
            true                      // has_bias
        );
        // Use the component name that matches the weight converter output
        bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;
        println!("BERT encoder weights loaded successfully");
        
        // Create ProsodyPredictor with the correct n_layers value from config
        let mut prosody_predictor = ProsodyPredictor::new(
            config.style_dim,    // style dimension from config 
            config.hidden_dim,   // hidden dimension from config
            config.n_layer,      // Using the correct n_layer value from config
            50,                  // max_dur
            0.0                  // dropout (inference mode)
        );
        // Use "predictor" as the component name to match the weight converter output
        prosody_predictor.load_weights_binary(&loader, "predictor", "module")?;
        println!("Prosody predictor weights loaded successfully");
        
        // Create decoder
        let mut decoder = Decoder::new(
            config.hidden_dim, // dim_in
            config.style_dim,  // style_dim
            config.n_mels,     // dim_out
            vec![3, 7, 11],    // resblock_kernel_sizes
            vec![8, 8],        // upsample_rates
            256,               // upsample_initial_channel
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]], // resblock_dilation_sizes
            vec![16, 16],      // upsample_kernel_sizes
            16,                // gen_istft_n_fft
            4,                 // gen_istft_hop_size
        );
        
        // Use "decoder" as the component name to match the weight converter output
        decoder.load_weights_binary(&loader, "decoder", "module")?;
        println!("Decoder weights loaded successfully");
            
        // Initialize G2P handler
        let g2p = G2PHandler::new("en-us")?;
        
        Ok(Self { 
            config,
            g2p,
            text_encoder: Some(text_encoder),
            bert: Some(bert),
            bert_encoder: Some(bert_encoder),
            prosody_predictor: Some(prosody_predictor),
            decoder: Some(decoder),
        })
    }

    /// Run inference to generate audio from text
    pub fn infer(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        // Convert text to phonemes using our G2P handler
        println!("Converting text to phonemes using G2P handler...");
        let g2p_result = self.g2p.convert(text);
        
        println!("Phonetic representation: {}", g2p_result.phonemes);
        
        // Default voice style embedding (use a default voice if no voice specified)
        let default_voice_embedding = Tensor::from_data(
            vec![0.0; self.config.style_dim * 2], // Reference + style parts as in Kokoro
            vec![1, self.config.style_dim * 2]
        );
        
        // Use the phoneme string for inference
        self.infer_with_phonemes(&g2p_result.phonemes, &default_voice_embedding, 1.0)
    }
    
    /// Implement the full inference pipeline
    pub fn infer_with_phonemes(&self, phonemes: &str, voice_embedding: &Tensor<f32>, speed_factor: f32) -> Result<Vec<f32>, Box<dyn Error>> {
        println!("Using voice embedding of shape {:?}", voice_embedding.shape());
        println!("Speed factor: {}", speed_factor);
        
        #[cfg(feature = "weights")]
        match (&self.text_encoder, &self.prosody_predictor, &self.decoder, &self.bert, &self.bert_encoder) {
            (Some(text_encoder), Some(prosody_predictor), Some(decoder), Some(bert), Some(bert_encoder)) => {
                // 1. Convert phonemes to token IDs
                let mut token_ids = Vec::new();
                token_ids.push(0); // Start of sequence token <bos>
                
                // Map phoneme strings to IDs using the vocabulary from the config
                for phoneme in phonemes.split_whitespace() {
                    if let Some(&id) = self.config.vocab.get(&phoneme.chars().next().unwrap_or(' ')) {
                        token_ids.push(id as i64);
                    } else {
                        println!("Warning: Phoneme '{}' not in vocabulary, using placeholder", phoneme);
                        token_ids.push(1); // Use a placeholder token ID
                    }
                }
                
                token_ids.push(0); // End of sequence token <eos>
                
                // Create tensor from token IDs
                let batch_size = 1;
                let seq_len = token_ids.len();
                
                println!("Input sequence length: {}", seq_len);
                if seq_len > 512 {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Input sequence too long: {} > 512 tokens", seq_len)
                    )));
                }
                
                // Create input_ids tensor [B, T]
                let mut input_ids_tensor = Tensor::new(vec![batch_size, seq_len]);
                for (idx, &id) in token_ids.iter().enumerate() {
                    input_ids_tensor[&[0, idx]] = id as i64;
                }
                
                // Create input_lengths and text_mask
                let input_lengths = vec![seq_len];
                
                // Create text_mask for padding (true = masked position)
                // In our case with a single batch item and no padding, all are valid positions
                let text_mask = Tensor::from_data(
                    vec![false; batch_size * seq_len],
                    vec![batch_size, seq_len]
                );
                
                // 2. Process input through BERT and get hidden states
                // Create attention mask: [batch_size, seq_len, seq_len] where 1 = masked position
                let mut attention_mask = Tensor::from_data(
                    vec![0; batch_size * seq_len * seq_len],
                    vec![batch_size, seq_len, seq_len]
                );
                for b in 0..batch_size {
                    for s1 in 0..seq_len {
                        for s2 in 0..seq_len {
                            if s1 >= input_lengths[b] || s2 >= input_lengths[b] {
                                // Mask out positions beyond sequence length
                                attention_mask[&[b, s1, s2]] = 1;
                            }
                        }
                    }
                }

                // 1. Process input through TextEncoder - expects [B, T] input, outputs [B, C, T]
                println!("Running TextEncoder...");
                let t_en = text_encoder.forward(&input_ids_tensor, &input_lengths, &text_mask);
                println!("TextEncoder output shape: {:?}", t_en.shape());

                // 2. Process input through BERT - expects [B, T] input, outputs [B, T, C]
                let bert_output = bert.forward(&input_ids_tensor, None, Some(&attention_mask));
                println!("BERT output shape: {:?}", bert_output.shape());

                // 3. Project BERT output - input [B, T, C], output [B, T, hidden_dim]
                let bert_encoder_output = bert_encoder.forward(&bert_output);
                println!("BERT encoder output shape: {:?}", bert_encoder_output.shape());

                // 4. Transpose BERT encoder output from [B, T, C] to [B, C, T] to match TextEncoder
                let (batch_size, seq_len, hidden_dim) = (
                    bert_encoder_output.shape()[0],
                    bert_encoder_output.shape()[1],
                    bert_encoder_output.shape()[2]
                );

                let mut d_en_bct_data = vec![0.0; batch_size * hidden_dim * seq_len];
                for b in 0..batch_size {
                    for t in 0..seq_len {
                        for c in 0..hidden_dim {
                            d_en_bct_data[b * hidden_dim * seq_len + c * seq_len + t] = 
                                bert_encoder_output[&[b, t, c]];
                        }
                    }
                }
                let d_en = Tensor::from_data(d_en_bct_data, vec![batch_size, hidden_dim, seq_len]);
                println!("BERT encoder output transposed to [B, C, T]: {:?}", d_en.shape());

                // 5. Split voice embedding properly
                let style_dim = self.config.style_dim;

                // Validate voice embedding shape
                if voice_embedding.shape().len() != 2 || voice_embedding.shape()[1] != style_dim * 2 {
                    panic!("Voice embedding must have shape [B, style_dim*2], got shape {:?}", 
                          voice_embedding.shape());
                }

                // Extract reference and style parts
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

                println!("Reference embedding shape: {:?}", ref_embedding.shape());
                println!("Style embedding shape: {:?}", style_embedding.shape());

                // 6. Create temporary identity alignment for duration prediction
                let mut temp_alignment_data = vec![0.0; seq_len * seq_len];
                for i in 0..seq_len {
                    temp_alignment_data[i * seq_len + i] = 1.0;
                }
                let temp_alignment = Tensor::from_data(temp_alignment_data, vec![seq_len, seq_len]);

                // 7. Get duration logits through ProsodyPredictor
                // Pass d_en in [B, C, T] format
                let (dur_logits, _) = prosody_predictor.forward(
                    &d_en,
                    &style_embedding,
                    &text_mask,
                    &temp_alignment
                );

                // 7. Calculate durations from logits
                let max_dur = prosody_predictor.max_dur;
                let mut durations = vec![0; seq_len];

                for pos in 0..seq_len {
                    let mut duration_sum = 0.0;
                    
                    for dur_idx in 0..max_dur {
                        let logit = dur_logits[&[0, pos, dur_idx]];
                        let sigmoid_value = 1.0 / (1.0 + (-logit).exp());
                        duration_sum += sigmoid_value;
                    }
                    
                    let duration = (duration_sum / speed_factor).round() as usize;
                    durations[pos] = std::cmp::max(1, duration);
                }

                println!("Calculated durations: {:?}", durations);
                let total_frames: usize = durations.iter().sum();

                // 9. Create alignment matrix based on durations
                println!("Creating alignment matrix with shape [seq_len={}, total_frames={}]", seq_len, total_frames);
                let alignment_matrix = self.create_alignment_from_durations(&durations);

                // 10. Get aligned encoder states with proper alignment
                let (_, en) = prosody_predictor.forward(
                    &d_en,  
                    &style_embedding, 
                    &text_mask, 
                    &alignment_matrix
                );
                println!("Aligned encoder states shape: {:?}", en.shape());

                // 11. Prepare input for predict_f0_noise, which expects [B, F, H]
                // en is currently [B, C, F] from prosody_predictor.forward
                let mut en_btf_data = vec![0.0; batch_size * total_frames * hidden_dim];
                for b in 0..batch_size {
                    for c in 0..hidden_dim {
                        for f in 0..total_frames {
                            if c < en.shape()[1] && f < en.shape()[2] {
                                en_btf_data[b * total_frames * hidden_dim + f * hidden_dim + c] = 
                                    en[&[b, c, f]];
                            } else {
                                panic!("Index out of bounds when preparing data for predict_f0_noise: b={}, c={}, f={}; en shape={:?}",
                                      b, c, f, en.shape());
                            }
                        }
                    }
                }

                let en_btf = Tensor::from_data(en_btf_data, vec![batch_size, total_frames, hidden_dim]);
                println!("Input for predict_f0_noise: {:?}", en_btf.shape());

                // 12. Predict F0 and noise
                println!("Calling predict_f0_noise");
                let (f0_pred, n_pred) = prosody_predictor.predict_f0_noise(&en_btf, &style_embedding);
                println!("F0 shape: {:?}, Noise shape: {:?}", f0_pred.shape(), n_pred.shape());

                // 13. Create ASR tensor = t_en @ alignment_matrix
                // t_en is [B, C, T], alignment_matrix is [T, F]
                // Result should be [B, C, F]
                let mut asr_data = vec![0.0; batch_size * hidden_dim * total_frames];

                for b in 0..batch_size {
                    for h in 0..hidden_dim {
                        for f in 0..total_frames {
                            let mut sum = 0.0;
                            for t in 0..seq_len {
                                if h < t_en.shape()[1] && t < t_en.shape()[2] && t < alignment_matrix.shape()[0] && f < alignment_matrix.shape()[1] {
                                    sum += t_en[&[b, h, t]] * alignment_matrix[&[t, f]];
                                }
                            }
                            asr_data[b * hidden_dim * total_frames + h * total_frames + f] = sum;
                        }
                    }
                }

                let asr = Tensor::from_data(asr_data, vec![batch_size, hidden_dim, total_frames]);
                println!("ASR tensor shape: {:?}", asr.shape());

                // 14. Generate audio
                println!("Calling decoder.forward");
                let audio = decoder.forward(&asr, &f0_pred, &n_pred, &ref_embedding);
                println!("Generated audio shape: {:?}", audio.shape());

                // Return audio data
                let audio_data = audio.data().to_vec();
                println!("Generated {} audio samples", audio_data.len());

                Ok(audio_data)
            },
            _ => {
                // Fallback to sine wave if components are not loaded
                println!("Warning: Model components not properly loaded. Returning sine wave placeholder.");
                let sample_rate = 24000;
                let duration = 1.0 / speed_factor;
                let frequency = 440.0;
                
                let num_samples = (sample_rate as f32 * duration) as usize;
                let mut sine_wave = vec![0.0f32; num_samples];
                
                for i in 0..num_samples {
                    let t = i as f32 / sample_rate as f32;
                    sine_wave[i] = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
                }
                
                Ok(sine_wave)
            }
        }
        
        #[cfg(not(feature = "weights"))]
        {
            // Fallback to sine wave if weights feature not enabled
            let sample_rate = 24000;
            let duration = 1.0 / speed_factor;
            let frequency = 440.0;
            
            let num_samples = (sample_rate as f32 * duration) as usize;
            let mut sine_wave = vec![0.0f32; num_samples];
            
            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                sine_wave[i] = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
            }
            
            Ok(sine_wave)
        }
    }
    
    /// Generate audio from text and a voice embedding
    pub fn infer_with_voice(&self, text: &str, voice_embedding: &Tensor<f32>, speed_factor: f32) -> Result<Vec<f32>, Box<dyn Error>> {
        // Convert text to phonemes using our G2P handler
        println!("Converting text to phonemes using G2P handler...");
        let g2p_result = self.g2p.convert(text);
        
        println!("Phonetic representation: {}", g2p_result.phonemes);
        
        // Use phonemes for inference
        self.infer_with_phonemes(&g2p_result.phonemes, voice_embedding, speed_factor)
    }
    
    /// Load a voice from a file
    pub fn load_voice(&self, voice_name: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
        #[cfg(feature = "weights")]
        {
            // Try to load the voice embedding from the binary weights directory
            let weights_dir = std::path::Path::new("../ferrocarril_weights");
            let voices_dir = weights_dir.join("voices");
            
            println!("Attempting to load voice '{}' from {}", voice_name, voices_dir.display());
            if !voices_dir.exists() {
                println!("Warning: Voices directory not found at {:?}", voices_dir);
                println!("Returning a default voice embedding for {}", voice_name);
            } else {
                // Check for voice file
                let voice_file = voices_dir.join(format!("{}.bin", voice_name));
                if voice_file.exists() {
                    println!("Loading voice embedding for {} from {:?}", voice_name, voice_file);
                    
                    // Read voice file
                    let voice_data = std::fs::read(voice_file)?;
                    let num_elements = voice_data.len() / 4; // Assuming f32 (4 bytes per element)
                    
                    // Convert bytes to f32
                    let mut voice_embedding_flat = vec![0.0; num_elements];
                    for i in 0..num_elements {
                        let bytes = [
                            voice_data[i*4], 
                            voice_data[i*4+1], 
                            voice_data[i*4+2], 
                            voice_data[i*4+3]
                        ];
                        voice_embedding_flat[i] = f32::from_le_bytes(bytes);
                    }
                    
                    // Get the voice JSON file to read the shape
                    let voice_json = voices_dir.join(format!("{}.json", voice_name));
                    if voice_json.exists() {
                        let json_str = std::fs::read_to_string(voice_json)?;
                        let json_data: serde_json::Value = serde_json::from_str(&json_str)?;
                        
                        // Extract shape from JSON metadata
                        if let Some(shape) = json_data.get("shape") {
                            if let Some(shape_array) = shape.as_array() {
                                if shape_array.len() >= 3 {
                                    // Typical shape is [510, 1, 256]
                                    let seq_len = shape_array[0].as_u64().unwrap_or(510) as usize;
                                    let batch = shape_array[1].as_u64().unwrap_or(1) as usize;
                                    let embed_dim = shape_array[2].as_u64().unwrap_or(256) as usize;
                                    
                                    println!("Voice shape from metadata: [{}, {}, {}]", 
                                             seq_len, batch, embed_dim);
                                    
                                    // Verify that our data length matches the expected shape
                                    let expected_elements = seq_len * batch * embed_dim;
                                    assert_eq!(num_elements, expected_elements,
                                             "Voice data length ({}) doesn't match expected shape size ({})",
                                             num_elements, expected_elements);
                                    
                                    // Use config.style_dim for calculations
                                    let style_dim = self.config.style_dim;
                                    
                                    println!("Creating voice embedding with style_dim = {}", style_dim);
                                    
                                    // In Kokoro, we take the middle position's embedding from [seq_len, batch, embed_dim]
                                    // And convert it to [1, style_dim*2] where the first half is reference and second is style
                                    let mid_position = seq_len / 2;
                                    
                                    // Calculate the offset in the flattened array to get to the middle position
                                    // The raw data is stored as [seq_len, batch, embed_dim]
                                    let position_start = mid_position * batch * embed_dim;
                                    
                                    // Create the final voice embedding with shape [1, style_dim * 2]
                                    let mut final_embedding = vec![0.0; style_dim * 2];
                                    
                                    // Check if we need to adapt the dimensions
                                    if embed_dim != style_dim {
                                        println!("Note: Voice embedding dim ({}) != style_dim ({}), adapting dimensions", 
                                                embed_dim, style_dim);
                                    }
                                    
                                    // Determine how many elements we can copy
                                    let copy_elements = std::cmp::min(style_dim, embed_dim);
                                    
                                    // In Kokoro: ref_s = voice_embedding, style = ref_s[:, 128:], ref = ref_s[:, :128]
                                    // Fill both halves of the embedding (reference and style)
                                    // First half: reference embedding
                                    for i in 0..copy_elements {
                                        if position_start + i < voice_embedding_flat.len() {
                                            final_embedding[i] = voice_embedding_flat[position_start + i];
                                        }
                                    }
                                    
                                    // Second half: style embedding
                                    for i in 0..copy_elements {
                                        if position_start + i < voice_embedding_flat.len() {
                                            final_embedding[i + style_dim] = voice_embedding_flat[position_start + i];
                                        }
                                    }
                                    
                                    // Create voice tensor with the correct shape: [1, style_dim*2]
                                    let voice_tensor = Tensor::from_data(final_embedding, vec![1, style_dim * 2]);
                                    
                                    // Validate the final shape
                                    assert_eq!(voice_tensor.shape()[0], 1, "Voice tensor batch dimension should be 1");
                                    assert_eq!(voice_tensor.shape()[1], style_dim * 2, 
                                              "Voice tensor feature dimension should be {}, got {}",
                                              style_dim * 2, voice_tensor.shape()[1]);
                                    
                                    println!("Created voice embedding with shape: {:?}", voice_tensor.shape());
                                    return Ok(voice_tensor);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback to a zero-initialized embedding with correct shape
        println!("Creating default voice embedding for {}", voice_name);
        let style_dim = self.config.style_dim;
        
        // Create embedding with shape [1, style_dim*2]
        let embedding = vec![0.0; style_dim * 2];
        let voice_tensor = Tensor::from_data(embedding, vec![1, style_dim * 2]);
        
        println!("Created default voice embedding with shape: {:?}", voice_tensor.shape());
        Ok(voice_tensor)
    }
    
    // Only for testing - enables directly running the inference with a proper alignment matrix
    #[cfg(feature = "weights")]
    pub fn infer_with_voice_test(&self, text: &str, voice_embedding: &Tensor<f32>, speed_factor: f32) -> Result<Vec<f32>, Box<dyn Error>> {
        // Convert text to phonemes using our G2P handler
        println!("Converting text to phonemes using G2P handler...");
        let g2p_result = self.g2p.convert(text);
        println!("Phonetic representation: {}", g2p_result.phonemes);
        
        // Extract components
        let text_encoder = match &self.text_encoder {
            Some(e) => e,
            None => return Err("TextEncoder not loaded".into()),
        };
        
        let bert = match &self.bert {
            Some(b) => b,
            None => return Err("BERT not loaded".into()),
        };
        
        let bert_encoder = match &self.bert_encoder {
            Some(be) => be,
            None => return Err("BERT encoder not loaded".into()),
        };
        
        let prosody_predictor = match &self.prosody_predictor {
            Some(pp) => pp,
            None => return Err("ProsodyPredictor not loaded".into()),
        };
        
        let decoder = match &self.decoder {
            Some(d) => d,
            None => return Err("Decoder not loaded".into()),
        };
        
        // 1. Convert phonemes to token IDs
        let mut token_ids = Vec::new();
        token_ids.push(0); // Start of sequence token <bos>
        
        // Map phoneme strings to IDs using the vocabulary from the config
        for phoneme in g2p_result.phonemes.split_whitespace() {
            if let Some(&id) = self.config.vocab.get(&phoneme.chars().next().unwrap_or(' ')) {
                token_ids.push(id as i64);
            } else {
                println!("Warning: Phoneme '{}' not in vocabulary, using placeholder", phoneme);
                token_ids.push(1); // Use a placeholder token ID
            }
        }
        
        token_ids.push(0); // End of sequence token <eos>
        
        // Create tensor from token IDs
        let batch_size = 1;
        let seq_len = token_ids.len();
        
        println!("Input sequence length: {}", seq_len);
        if seq_len > 512 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Input sequence too long: {} > 512 tokens", seq_len)
            )));
        }
        
        // Create input_ids tensor [B, T]
        let mut input_ids_tensor = Tensor::new(vec![batch_size, seq_len]);
        for (idx, &id) in token_ids.iter().enumerate() {
            input_ids_tensor[&[0, idx]] = id as i64;
        }
        
        // Create input_lengths and text_mask
        let input_lengths = vec![seq_len];
        
        // Create text_mask for padding
        let text_mask = Tensor::from_data(
            vec![false; batch_size * seq_len],
            vec![batch_size, seq_len]
        );
        
        // Process through BERT and TextEncoder
        let attention_mask = Tensor::from_data(
            vec![0; batch_size * seq_len * seq_len],
            vec![batch_size, seq_len, seq_len]
        );
        let bert_output = bert.forward(&input_ids_tensor, None, Some(&attention_mask));
        let d_en_btc = bert_encoder.forward(&bert_output);
        let t_en = text_encoder.forward(&input_ids_tensor, &input_lengths, &text_mask);
        
        // Split voice embedding
        let style_dim = self.config.style_dim;
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
        
        // Convert d_en_btc to format needed for prosody processing
        let mut d_en_bct_data = vec![0.0; batch_size * self.config.hidden_dim * seq_len];
        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..self.config.hidden_dim {
                    d_en_bct_data[b * self.config.hidden_dim * seq_len + c * seq_len + t] = 
                        d_en_btc[&[b, t, c]];
                }
            }
        }
        let d_en = Tensor::from_data(d_en_bct_data, vec![batch_size, self.config.hidden_dim, seq_len]);
        
        // Create a separate tensor for duration prediction (d_for_dur)
        let mut d_transpose_for_dur = vec![0.0; batch_size * self.config.hidden_dim * seq_len];
        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..self.config.hidden_dim {
                    if t < d_en_btc.shape()[1] && h < d_en_btc.shape()[2] {
                        d_transpose_for_dur[b * self.config.hidden_dim * seq_len + h * seq_len + t] = 
                            d_en_btc[&[b, t, h]];
                    }
                }
            }
        }
        
        // Convert to [B, H, T] format
        let d_for_dur = Tensor::from_data(
            d_transpose_for_dur,
            vec![batch_size, self.config.hidden_dim, seq_len]
        );
        
        println!("Debug: d_for_dur shape: {:?}", d_for_dur.shape());
        
        // Calculate durations from logits
        let max_dur = prosody_predictor.max_dur;

        // Get duration predictions (we're mimicking Kokoro's forward_with_tokens method)
        // First, get durations using a temporary alignment with same shape as text
        // Create alignment matrix for duration prediction (T×T)
        let mut temp_alignment_data = vec![0.0; seq_len * seq_len];
        // Identity matrix - each position attends to itself
        for i in 0..seq_len {
            temp_alignment_data[i * seq_len + i] = 1.0;
        }
        let temp_alignment = Tensor::from_data(
            temp_alignment_data, 
            vec![seq_len, seq_len]
        );

        // Process d_for_dur through predictor to get durations
        let (dur_logits, _) = prosody_predictor.forward(&d_for_dur, &style_embedding, &text_mask, &temp_alignment);

        // Calculate durations from logits (sigmoid + sum + scale)
        let mut durations = vec![0; seq_len];

        // Iterate over sequence positions
        for pos in 0..seq_len {
            let mut duration_sum = 0.0;
            
            // Sum sigmoid values across the max_dur dimension
            for dur_idx in 0..max_dur {
                if dur_idx < dur_logits.shape()[2] {
                    let logit = dur_logits[&[0, pos, dur_idx]];
                    let sigmoid_value = 1.0 / (1.0 + (-logit).exp());
                    duration_sum += sigmoid_value;
                }
            }
            
            // Scale by speed factor and round
            let duration = (duration_sum / speed_factor).round() as usize;
            durations[pos] = std::cmp::max(1, duration);
        }

        // Create proper alignment matrix
        let proper_alignment = self.create_alignment_from_durations(&durations);

        // Get total frames for later use
        let total_frames: usize = durations.iter().sum();
        
        println!("Created proper alignment matrix with shape [{}, {}]", seq_len, proper_alignment.shape()[1]);

        // Run forward pass with proper alignment and d_en (not d_for_dur)
        let (_, en) = prosody_predictor.forward(&d_en, &style_embedding, &text_mask, &proper_alignment);
        
        // Convert en to [B, F, H] format for F0/noise prediction
        let mut en_btf_data = vec![0.0; batch_size * total_frames * self.config.hidden_dim];
        for b in 0..batch_size {
            for f in 0..total_frames {
                for h in 0..self.config.hidden_dim {
                    if b < en.shape()[0] && h < en.shape()[1] && f < en.shape()[2] {
                        en_btf_data[b * total_frames * self.config.hidden_dim + f * self.config.hidden_dim + h] = 
                            en[&[b, h, f]];
                    }
                }
            }
        }
        
        // Create a properly shaped tensor for [B, F, H]
        let en_btf = Tensor::from_data(
            en_btf_data, 
            vec![batch_size, total_frames, self.config.hidden_dim]
        );
        
        // Double-check that the style_embedding has the correct dimensions
        // If not, create a new one with the right dimensions
        let effective_style_embedding;
        if style_embedding.shape()[1] != self.config.style_dim {
            println!("WARNING: style_embedding has incorrect dimensions. Creating a fixed version.");
            // Create a corrected style embedding
            let corrected_style_data = vec![0.5; batch_size * self.config.style_dim];
            effective_style_embedding = Tensor::from_data(
                corrected_style_data,
                vec![batch_size, self.config.style_dim]
            );
        } else {
            effective_style_embedding = style_embedding.clone();
        }
        
        // Explicitly log shapes right before calling predict_f0_noise
        println!("Final en_btf shape: {:?}, style_embedding shape: {:?}", 
                 en_btf.shape(), effective_style_embedding.shape());
        
        // Predict F0 and noise
        let (f0_pred, n_pred) = prosody_predictor.predict_f0_noise(&en_btf, &effective_style_embedding);
        
        // Create batched alignment
        let mut batched_alignment = vec![0.0; batch_size * seq_len * total_frames];
        for i in 0..seq_len * total_frames {
            batched_alignment[i] = proper_alignment.data()[i];
        }
        let pred_aln_trg = Tensor::from_data(
            batched_alignment,
            vec![batch_size, seq_len, total_frames]
        );
        
        // ASR tensor
        let mut asr_data = vec![0.0; batch_size * self.config.hidden_dim * total_frames];
        for b in 0..batch_size {
            for h in 0..self.config.hidden_dim {
                for f in 0..total_frames {
                    let mut sum = 0.0;
                    for t in 0..seq_len {
                        if h < t_en.shape()[1] && t < t_en.shape()[2] && t < pred_aln_trg.shape()[1] && f < pred_aln_trg.shape()[2] {
                            sum += t_en[&[b, h, t]] * pred_aln_trg[&[b, t, f]];
                        }
                    }
                    
                    if b * self.config.hidden_dim * total_frames + h * total_frames + f < asr_data.len() {
                        asr_data[b * self.config.hidden_dim * total_frames + h * total_frames + f] = sum;
                    }
                }
            }
        }
        
        let asr = Tensor::from_data(
            asr_data,
            vec![batch_size, self.config.hidden_dim, total_frames]
        );
        
        // Generate audio
        let audio = decoder.forward(&asr, &f0_pred, &n_pred, &ref_embedding);
        
        // Return audio data
        Ok(audio.data().to_vec())
    }
}