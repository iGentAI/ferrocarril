//! FerroModel implementation - main TTS inference model

use ferrocarril_core::{Config, tensor::Tensor};
use ferrocarril_nn::{text_encoder::TextEncoder, prosody::ProsodyPredictor, vocoder::Decoder, Forward};
use ferrocarril_nn::bert::{CustomAlbert, CustomAlbertConfig};
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

    /// CustomAlbert component
    #[cfg(feature = "weights")]
    bert: Option<CustomAlbert>,

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
        for frame_idx in 0..total_frames {
            let mut col_sum = 0.0f32;
            for token_idx in 0..seq_len {
                col_sum += alignment.data()[token_idx * total_frames + frame_idx];
            }
            assert!((col_sum - 1.0f32).abs() < 1e-6f32,
                "Column {} sum should be 1.0, got {}", frame_idx, col_sum);
        }

        // Validate that rows sum to their respective durations
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

    /// Load a model from PyTorch weights (not fully implemented)
    pub fn load(_path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        // TODO: Implement PyTorch model loading
        eprintln!(
            "ferrocarril: warning: PyTorch weight loading is not fully implemented; use load_binary with converted weights instead"
        );

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

    /// Load a model from converted binary weights on the filesystem.
    ///
    /// This is a thin wrapper around `load_from_loader`: it opens the
    /// directory-layout weights at `path` via
    /// `BinaryWeightLoader::from_directory`, then delegates the rest of
    /// the construction.
    #[cfg(feature = "weights")]
    pub fn load_binary(path: &str, config: Config) -> Result<Self, Box<dyn Error>> {
        let loader = BinaryWeightLoader::from_directory(path)
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;
        Self::load_from_loader(loader, config)
    }

    /// Construct a `FerroModel` from an already-built
    /// `BinaryWeightLoader`. This is the always-available entry point
    /// for environments that can't use `BinaryWeightLoader::from_directory`
    /// directly (notably WebAssembly, where the caller builds the loader
    /// via `BinaryWeightLoader::from_metadata_str` and an in-memory
    /// blob provider).
    #[cfg(feature = "weights")]
    pub fn load_from_loader(
        loader: BinaryWeightLoader,
        config: Config,
    ) -> Result<Self, Box<dyn Error>> {
        let profile = std::env::var("FERRO_PROFILE").is_ok();
        let t_start = std::time::Instant::now();
        let mut t_mark = t_start;
        macro_rules! stage {
            ($name:expr) => {
                if profile {
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[profile] load {:<32} {:>9.3} ms",
                        $name,
                        (now - t_mark).as_secs_f64() * 1000.0
                    );
                    t_mark = now;
                }
            };
        }

        if loader.is_empty() {
            eprintln!(
                "ferrocarril: warning: no weights were loaded into BinaryWeightLoader — model may not function correctly"
            );
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
        stage!("TextEncoder");

        // Create and load CustomAlbert component with fixed config type
        let albert_config = CustomAlbertConfig {
            vocab_size: config.n_token,
            embedding_size: 128,  // Albert factorized embedding size
            hidden_size: config.plbert.hidden_size,
            num_attention_heads: config.plbert.num_attention_heads,
            num_hidden_layers: config.plbert.num_hidden_layers,
            intermediate_size: config.plbert.intermediate_size,
            max_position_embeddings: 512,
            dropout_prob: 0.0,
        };

        let mut bert = CustomAlbert::new(albert_config);
        // Use the component name that matches the weight converter output
        bert.load_weights_binary(&loader, "bert", "module")?;
        stage!("CustomAlbert (BERT)");

        // Create and load BERT encoder (linear projection layer)
        let mut bert_encoder = Linear::new(
            config.plbert.hidden_size, // input_dim
            config.hidden_dim,        // output_dim
            true                      // has_bias
        );
        bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;
        stage!("BertEncoder Linear");

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
        stage!("ProsodyPredictor");

        let mut decoder = Decoder::new(
            config.hidden_dim,
            config.style_dim,
            config.n_mels,
            config.istftnet.resblock_kernel_sizes.clone(),
            config.istftnet.upsample_rates.clone(),
            config.istftnet.upsample_initial_channel,
            config.istftnet.resblock_dilation_sizes.clone(),
            config.istftnet.upsample_kernel_sizes.clone(),
            config.istftnet.gen_istft_n_fft,
            config.istftnet.gen_istft_hop_size,
        );

        // Use "decoder" as the component name to match the weight converter output
        decoder.load_weights_binary(&loader, "decoder", "module")?;
        stage!("Decoder (incl Generator)");

        // Initialize G2P handler. This still depends on the in-tree
        // Phonesis dictionary, which is fully vendored and works on all
        // supported targets (including wasm32).
        let g2p = G2PHandler::new("en-us")?;

        if profile {
            let total = (std::time::Instant::now() - t_start).as_secs_f64() * 1000.0;
            eprintln!("[profile] load {:-<32} {:->12}", "", "");
            eprintln!(
                "[profile] load {:<32} {:>9.3} ms",
                "TOTAL load_from_loader", total
            );
        }

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
        let g2p_result = self.g2p.convert(text);

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
        let profile = std::env::var("FERRO_PROFILE").is_ok();
        let t_start = std::time::Instant::now();
        let mut t_mark = t_start;
        macro_rules! stage {
            ($name:expr) => {
                if profile {
                    let now = std::time::Instant::now();
                    eprintln!(
                        "[profile] infer {:<31} {:>9.3} ms",
                        $name,
                        (now - t_mark).as_secs_f64() * 1000.0
                    );
                    t_mark = now;
                }
            };
        }

        if profile {
            ferrocarril_nn::conv::reset_conv1d_stats();
        }

        #[cfg(feature = "weights")]
        match (&self.text_encoder, &self.prosody_predictor, &self.decoder, &self.bert, &self.bert_encoder) {
            (Some(text_encoder), Some(prosody_predictor), Some(decoder), Some(bert), Some(bert_encoder)) => {
                // 1. Convert phonemes to token IDs.
                //
                // Kokoro tokenizes the phoneme stream character-by-character,
                // looking up each single IPA code point (diphthongs like `oʊ`
                // are TWO tokens: `o` and `ʊ`). Whitespace in the Phonesis
                // output is a separator, not a token.
                let mut token_ids: Vec<i64> = Vec::new();
                token_ids.push(0); // Start of sequence token <bos>

                for ch in phonemes.chars() {
                    if ch.is_whitespace() {
                        continue;
                    }
                    if let Some(&id) = self.config.vocab.get(&ch) {
                        token_ids.push(id as i64);
                    } else {
                        // Unknown phoneme: skip silently. The G2P layer
                        // owns vocabulary correctness.
                    }
                }

                token_ids.push(0); // End of sequence token <eos>

                // Create tensor from token IDs
                let batch_size = 1;
                let seq_len = token_ids.len();

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
                stage!("tokenize + input tensors");

                // 1. Process input through TextEncoder - expects [B, T] input, outputs [B, C, T]
                let t_en = text_encoder.forward(&input_ids_tensor, &input_lengths, &text_mask);
                stage!("TextEncoder forward");

                // 2. Process input through BERT - expects [B, T] input, outputs [B, T, C]
                let attention_mask = Tensor::from_data(vec![1i64; batch_size * seq_len], vec![batch_size, seq_len]);
                let bert_output = bert.forward(&input_ids_tensor, None, Some(&attention_mask));
                stage!("BERT forward");

                // 3. Project BERT output - input [B, T, C], output [B, T, hidden_dim]
                let bert_encoder_output = bert_encoder.forward(&bert_output);
                stage!("BertEncoder Linear");

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
                stage!("BTC->BCT transpose");

                // 5. Split voice embedding properly.
                //
                // Python Kokoro: `ref_s = pack[len(ps) - 1]` picks the row
                // from the `[510, 1, 256]` voice pack corresponding to the
                // current phoneme count, giving a `[1, 256]` tensor. We then
                // split `[:, :128]` as reference and `[:, 128:]` as style.
                //
                // We accept either the raw voice pack (`[510, 256]`) or an
                // already-indexed `[1, 256]` embedding.
                let style_dim = self.config.style_dim;
                let voice_shape = voice_embedding.shape();

                let (ref_embedding, style_embedding) = if voice_shape.len() == 2
                    && voice_shape[0] == 510
                    && voice_shape[1] == style_dim * 2
                {
                    // Raw voice pack [510, 256]. Python Kokoro indexes by
                    // `len(ps) - 1` where `ps` is the phoneme string BEFORE
                    // BOS/EOS insertion. Our `seq_len` here is
                    // `num_phonemes + 2` (BOS and EOS were already pushed),
                    // so the correct row is `seq_len - 3` clamped to the pack.
                    let raw_phoneme_count = seq_len.saturating_sub(2);
                    let row_idx =
                        std::cmp::min(raw_phoneme_count.saturating_sub(1), 509);
                    let mut ref_data = vec![0.0f32; batch_size * style_dim];
                    let mut style_data = vec![0.0f32; batch_size * style_dim];
                    for b in 0..batch_size {
                        for i in 0..style_dim {
                            ref_data[b * style_dim + i] =
                                voice_embedding[&[row_idx, i]];
                            style_data[b * style_dim + i] =
                                voice_embedding[&[row_idx, i + style_dim]];
                        }
                    }
                    (
                        Tensor::from_data(ref_data, vec![batch_size, style_dim]),
                        Tensor::from_data(style_data, vec![batch_size, style_dim]),
                    )
                } else if voice_shape.len() == 2
                    && voice_shape[0] == batch_size
                    && voice_shape[1] == style_dim * 2
                {
                    // Legacy pre-indexed embedding [B, 256].
                    let mut ref_data = vec![0.0f32; batch_size * style_dim];
                    let mut style_data = vec![0.0f32; batch_size * style_dim];
                    for b in 0..batch_size {
                        for i in 0..style_dim {
                            ref_data[b * style_dim + i] = voice_embedding[&[b, i]];
                            style_data[b * style_dim + i] =
                                voice_embedding[&[b, i + style_dim]];
                        }
                    }
                    (
                        Tensor::from_data(ref_data, vec![batch_size, style_dim]),
                        Tensor::from_data(style_data, vec![batch_size, style_dim]),
                    )
                } else {
                    panic!(
                        "Voice embedding must be either a raw pack of shape [510, {}] \
                         or a pre-indexed embedding of shape [{}, {}], got shape {:?}",
                        style_dim * 2,
                        batch_size,
                        style_dim * 2,
                        voice_shape
                    );
                };
                stage!("voice split");

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
                ).map_err(|e| Box::new(e) as Box<dyn Error>)?;
                stage!("ProsodyPredictor forward (dur)");

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

                let total_frames: usize = durations.iter().sum();
                let alignment_matrix = self.create_alignment_from_durations(&durations);
                stage!("decode durations + alignment");

                // 10. Get aligned encoder states with proper alignment
                let (_, en) = prosody_predictor.forward(
                    &d_en,
                    &style_embedding,
                    &text_mask,
                    &alignment_matrix
                ).map_err(|e| Box::new(e) as Box<dyn Error>)?;
                stage!("ProsodyPredictor forward (aligned)");

                // 11. Predict F0 and noise.
                //
                // ProsodyPredictor::predict_f0_noise expects `en` in
                // `[B, d_model + style_dim, F]` format and transposes to BFC internally.
                let (f0_pred, n_pred) = prosody_predictor.predict_f0_noise(&en, &style_embedding)
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                stage!("predict_f0_noise");

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
                stage!("ASR tensor (t_en @ align)");

                // 14. Generate audio
                let audio_result = decoder.forward(&asr, &f0_pred, &n_pred, &ref_embedding);
                let audio = match audio_result {
                    Ok(audio_tensor) => audio_tensor,
                    Err(e) => {
                        return Err(Box::new(e) as Box<dyn Error>);
                    }
                };
                stage!("Decoder forward (incl Generator)");

                // Return audio data
                let audio_data = audio.data().to_vec();

                if profile {
                    ferrocarril_nn::conv::dump_conv1d_stats();
                    let total = (std::time::Instant::now() - t_start).as_secs_f64() * 1000.0;
                    eprintln!("[profile] infer {:-<31} {:->12}", "", "");
                    eprintln!(
                        "[profile] infer {:<31} {:>9.3} ms",
                        "TOTAL infer_with_phonemes", total
                    );
                }

                Ok(audio_data)
            },
            _ => {
                // Fallback to sine wave if components are not loaded
                eprintln!(
                    "ferrocarril: warning: model components not properly loaded; returning sine wave placeholder"
                );
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
        let g2p_result = self.g2p.convert(text);

        // Use phonemes for inference
        self.infer_with_phonemes(&g2p_result.phonemes, voice_embedding, speed_factor)
    }

    /// Load a voice from a file
    pub fn load_voice(&self, voice_name: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
        #[cfg(feature = "weights")]
        {
            // Try to load the voice embedding from the binary weights directory.
            // The voice files ship as `[510, 1, 256]` — one `[1, 256]` style
            // embedding per possible phoneme count. Python Kokoro indexes the
            // pack by `len(phonemes) - 1` at inference time; we preserve the
            // full pack here as `[seq_len, embed_dim]` and let
            // `infer_with_phonemes` do the indexing.
            let weights_dir = std::path::Path::new("../ferrocarril_weights");
            let voices_dir = weights_dir.join("voices");

            if !voices_dir.exists() {
                eprintln!(
                    "ferrocarril: warning: voices directory not found at {:?}",
                    voices_dir
                );
            } else {
                let voice_file = voices_dir.join(format!("{}.bin", voice_name));
                if voice_file.exists() {
                    let voice_data = std::fs::read(voice_file)?;
                    let num_elements = voice_data.len() / 4;
                    let mut voice_flat = vec![0.0f32; num_elements];
                    for i in 0..num_elements {
                        voice_flat[i] = f32::from_le_bytes([
                            voice_data[i * 4],
                            voice_data[i * 4 + 1],
                            voice_data[i * 4 + 2],
                            voice_data[i * 4 + 3],
                        ]);
                    }

                    let voice_json = voices_dir.join(format!("{}.json", voice_name));
                    if voice_json.exists() {
                        let json_str = std::fs::read_to_string(voice_json)?;
                        let json_data: serde_json::Value = serde_json::from_str(&json_str)?;
                        if let Some(shape) = json_data.get("shape") {
                            if let Some(shape_array) = shape.as_array() {
                                if shape_array.len() >= 3 {
                                    let seq_len = shape_array[0].as_u64().unwrap_or(510) as usize;
                                    let batch = shape_array[1].as_u64().unwrap_or(1) as usize;
                                    let embed_dim =
                                        shape_array[2].as_u64().unwrap_or(256) as usize;

                                    let expected = seq_len * batch * embed_dim;
                                    if num_elements != expected {
                                        return Err(format!(
                                            "Voice data length {} != expected {} ({}*{}*{})",
                                            num_elements, expected, seq_len, batch, embed_dim
                                        ).into());
                                    }

                                    if batch != 1 {
                                        return Err(format!(
                                            "Voice pack batch dim must be 1, got {} (shape [{}, {}, {}])",
                                            batch, seq_len, batch, embed_dim
                                        ).into());
                                    }
                                    let voice_tensor = Tensor::from_data(
                                        voice_flat,
                                        vec![seq_len, embed_dim],
                                    );
                                    return Ok(voice_tensor);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback to a zero-initialized embedding with correct shape.
        // Keep the legacy `[1, style_dim*2]` shape so any tests that expect
        // the pre-indexed form still work.
        eprintln!(
            "ferrocarril: warning: falling back to zero voice embedding for '{}'",
            voice_name
        );
        let style_dim = self.config.style_dim;
        let embedding = vec![0.0; style_dim * 2];
        let voice_tensor = Tensor::from_data(embedding, vec![1, style_dim * 2]);
        Ok(voice_tensor)
    }
}