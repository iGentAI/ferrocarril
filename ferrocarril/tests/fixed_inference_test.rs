//! Fixed inference pipeline test for Ferrocarril

#[cfg(test)]
mod tests {
    use ferrocarril_core::{Config, tensor::Tensor, PhonesisG2P};
    use ferrocarril::model::FerroModel;
    use ferrocarril_dsp;
    use std::error::Error;
    use std::path::Path;
    use std::fs;

    // Skip this test by default since it requires the weights to be downloaded first
    // Run with: cargo test --test fixed_inference_test -- --ignored
    #[test]
    #[ignore = "Requires downloaded weights"]
    fn test_full_inference_pipeline() -> Result<(), Box<dyn Error>> {
        println!("=== Ferrocarril Full Inference Test ===");
        
        // Step 1: Check if weights exist
        let weights_dir = Path::new("./ferrocarril_weights");
        
        if !weights_dir.exists() {
            println!("Weight directory not found at {}", weights_dir.display());
            println!("Please run the download_and_convert_weights.sh script first.");
            return Ok(());  // Skip test rather than fail
        }
        
        println!("Found weights directory: {}", weights_dir.display());
        
        // Step 2: Load configuration
        let config_path = weights_dir.join("config.json");
        println!("Loading configuration from {}", config_path.display());
        let config = Config::from_json(config_path.to_str().unwrap())?;
        println!("Configuration loaded successfully");
        
        // Step 3: Initialize the model with binary weights
        println!("Initializing model with binary weights...");
        
        // Use feature detection for conditional compilation
        #[cfg(not(feature = "weights"))]
        {
            println!("Test skipped: This test requires the 'weights' feature to be enabled.");
            println!("Please rebuild with: cargo test --features weights");
            return Ok(());
        }
        
        #[cfg(feature = "weights")]
        let model = match FerroModel::load_binary(weights_dir.to_str().unwrap(), config) {
            Ok(m) => m,
            Err(e) => {
                println!("Failed to load model: {}", e);
                return Err(e);
            }
        };
        
        println!("Model initialized successfully");
        
        // Step 4: Set up test utterances
        let test_text = "Hello, world!";
        
        // Create output directory if it doesn't exist
        let output_dir = Path::new("test_output");
        fs::create_dir_all(output_dir)?;
        
        // Step 5: Run inference
        println!("\nTest text: \"{}\"", test_text);
        
        // Run inference - this will test the entire pipeline
        println!("Running inference...");
        #[cfg(feature = "weights")]
        let audio_data = model.infer(test_text)?;
        
        #[cfg(feature = "weights")]
        {
            // Create audio tensor and save to WAV file
            let audio_tensor = Tensor::from_data(audio_data.clone(), vec![audio_data.len()]);
            let output_path = output_dir.join("test_output.wav");
            ferrocarril_dsp::save_wav(&audio_tensor, output_path.to_str().unwrap())?;
            
            println!("Generated audio saved to: {}", output_path.display());
            println!("Audio length: {} samples", audio_data.len());
            
            // Verify audio data was generated
            assert!(audio_data.len() > 0, "No audio data was generated");
            
            // Check for realistic audio values (should be between -1.0 and 1.0)
            let has_valid_values = audio_data.iter().all(|&sample| sample >= -1.0 && sample <= 1.0);
            assert!(has_valid_values, "Audio contains values outside the valid range");
            
            // Check for non-zero values (zero tensor would indicate potential issues)
            let has_non_zero = audio_data.iter().any(|&sample| sample != 0.0);
            assert!(has_non_zero, "Audio contains only zeros, indicating potential issues");
        }
        
        println!("\n=== Full Inference Test Completed Successfully ===");
        Ok(())
    }

    #[test]
    fn test_basic_inference() -> Result<(), Box<dyn Error>> {
        println!("=== Ferrocarril Basic Inference Test ===");
        
        // Create output directory if it doesn't exist
        let output_dir = Path::new("test_output");
        fs::create_dir_all(output_dir)?;
        
        // Create a basic Config
        // Instead of using default_for_testing, load from config.json if available
        // or create a basic configuration manually if needed
        let config = if Path::new("ferrocarril_weights/config.json").exists() {
            // Load from existing config file if available
            match Config::from_json("ferrocarril_weights/config.json") {
                Ok(config) => config,
                Err(e) => {
                    println!("Warning: Failed to load config from file: {}", e);
                    // Create a basic config manually
                    let mut vocab = std::collections::HashMap::new();
                    for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-".chars().enumerate() {
                        vocab.insert(c, i);
                    }
                    
                    Config {
                        vocab,
                        n_token: 150,
                        hidden_dim: 512,
                        n_layer: 4,
                        style_dim: 128,
                        n_mels: 80,
                        max_dur: 50,
                        dropout: 0.1,
                        text_encoder_kernel_size: 5,
                        istftnet: ferrocarril_core::IstftnetConfig {
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
                        },
                        plbert: ferrocarril_core::PlbertConfig {
                            hidden_size: 768,
                            num_attention_heads: 12,
                            num_hidden_layers: 12,
                            intermediate_size: 3072,
                        },
                    }
                }
            }
        } else {
            // Create a basic config manually
            let mut vocab = std::collections::HashMap::new();
            for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-".chars().enumerate() {
                vocab.insert(c, i);
            }
            
            Config {
                vocab,
                n_token: 150,
                hidden_dim: 512,
                n_layer: 4,
                style_dim: 128,
                n_mels: 80,
                max_dur: 50,
                dropout: 0.1,
                text_encoder_kernel_size: 5,
                istftnet: ferrocarril_core::IstftnetConfig {
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
                },
                plbert: ferrocarril_core::PlbertConfig {
                    hidden_size: 768,
                    num_attention_heads: 12,
                    num_hidden_layers: 12,
                    intermediate_size: 3072,
                },
            }
        };
        
        // Create a new model
        let model = FerroModel::new(config)?;
        
        // Test text
        let test_text = "Hello, this is a test of the Ferrocarril TTS system with random weights.";
        println!("\nTest text: \"{}\"", test_text);
        
        // Test G2P component
        let g2p = PhonesisG2P::new("en-us")?;
        let phonemes = match g2p.convert(test_text) {
            Ok(p) => p,
            Err(e) => {
                println!("Warning: G2P conversion failed: {}. Using raw text.", e);
                test_text.to_string()
            }
        };
        println!("G2P output: {}", phonemes);
         
        // Create a dummy voice embedding tensor
        let voice_embedding = Tensor::from_data(
            vec![0.5; 128 * 2], // style_dim * 2 (128 for reference, 128 for style)
            vec![1, 128 * 2]
        );
        
        // Run inference with the voice
        println!("Running inference with randomly initialized weights...");
        let audio_data = model.infer_with_voice(&phonemes, &voice_embedding, 1.0)?;
        
        // Create audio tensor and save to WAV file
        let audio_tensor = Tensor::from_data(audio_data.clone(), vec![audio_data.len()]);
        let output_path = output_dir.join("random_weights_output.wav");
        ferrocarril_dsp::save_wav(&audio_tensor, output_path.to_str().unwrap())?;
        
        println!("Generated audio saved to: {}", output_path.display());
        println!("Audio length: {} samples", audio_data.len());
        
        println!("\n=== Basic Inference Test Completed Successfully ===");
        Ok(())
    }
}