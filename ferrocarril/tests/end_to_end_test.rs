#[cfg(feature = "weights")]
mod end_to_end_test {
    use ferrocarril::model::FerroModel;
    use ferrocarril_core::{Config, tensor::Tensor};
    use std::error::Error;
    
    // Helper function to load test config that matches the actual model dimensions
    fn load_test_config() -> Config {
        // Create a minimal configuration for testing
        let mut vocab = std::collections::HashMap::new();
        // Add minimal entries needed for test
        vocab.insert('A', 65); 
        vocab.insert('B', 66);
        vocab.insert('C', 67);
        vocab.insert('D', 68);
        vocab.insert('E', 69);
        
        // Create IstftnetConfig
        let istftnet = ferrocarril_core::IstftnetConfig {
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
        
        // Create PlbertConfig - With the CORRECT dimensions matching the actual weights
        let plbert = ferrocarril_core::PlbertConfig {
            hidden_size: 768,          // Matches model - was 384
            num_attention_heads: 12,   // Matches model - was 6
            num_hidden_layers: 6,      
            intermediate_size: 2048,   // Matches model - was 1536
        };
        
        Config {
            vocab,
            n_token: 178,              // Matches model - was 128
            hidden_dim: 512,           // Matches model - was 384
            n_layer: 3,
            style_dim: 128,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.0,              // Changed to 0.0 for inference
            text_encoder_kernel_size: 5,
            istftnet,
            plbert,
        }
    }

    #[test]
    fn test_end_to_end_tts_synthesis() -> Result<(), Box<dyn Error>> {
        // Skip test if weight files are not available
        let weight_path = std::path::Path::new("../ferrocarril_weights");
        if !weight_path.exists() {
            println!("Skipping test_end_to_end_tts_synthesis - weight files not available");
            return Ok(());
        }

        // Load the model
        println!("Loading model...");
        let config = load_test_config();
        
        // Print key config dimensions for debugging
        println!("Config dimensions: hidden_dim={}, style_dim={}", config.hidden_dim, config.style_dim);
        
        let model = FerroModel::load_binary("../ferrocarril_weights", config)?;

        // Create a test voice embedding
        // The style_dim should match config value (128), and the embedding has shape [1, style_dim*2]
        let style_dim = 128; 
        let voice_embedding = Tensor::from_data(
            vec![0.1; 2 * style_dim],
            vec![1, 2 * style_dim]
        );
        
        println!("Created voice embedding with shape: {:?}", voice_embedding.shape());

        // Simple phoneme input for testing - using minimal representation
        // Avoid using complex text that would require extensive G2P processing
        let test_phonemes = "AA";

        // Test audio generation
        println!("Generating audio from phonemes: {}", test_phonemes);
        let audio = model.infer_with_phonemes(
            test_phonemes, 
            &voice_embedding,
            1.0 // speed factor
        )?;

        // Verify audio output
        println!("Generated audio with {} samples", audio.len());
        
        // Basic validation that we got audio output
        assert!(!audio.is_empty(), "Audio output should not be empty");
        
        println!("End-to-end test completed successfully!");
        
        Ok(())
    }

    #[test]
    fn test_voice_embedding_handling() -> Result<(), Box<dyn Error>> {
        // Skip test if weight files are not available
        let weight_path = std::path::Path::new("../ferrocarril_weights");
        if !weight_path.exists() {
            println!("Skipping test_voice_embedding_handling - weight files not available");
            return Ok(());
        }

        // Load the model
        println!("Loading model...");
        let config = load_test_config();
        let model = FerroModel::load_binary("../ferrocarril_weights", config)?;
        
        // Get the style dimension from the config
        let style_dim = 128;

        // Test default voice loading
        println!("Testing default voice loading");
        let default_voice = model.load_voice("default")?;
        
        // Check the voice embedding shape
        println!("Voice embedding shape: {:?}", default_voice.shape());
        assert_eq!(default_voice.shape().len(), 2, "Voice embedding should be 2-dimensional");
        assert_eq!(default_voice.shape()[0], 1, "Batch dimension should be 1");
        assert_eq!(default_voice.shape()[1], 2 * style_dim, 
                  "Feature dimension should be 2 * style_dim for reference + style parts");
        
        println!("Voice embedding test completed successfully!");
        
        Ok(())
    }
}