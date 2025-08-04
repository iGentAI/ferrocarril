#[cfg(test)]
#[cfg(feature = "weights")]
mod tests {
    use ferrocarril_core::{Config, tensor::Tensor, PhonesisG2P, LoadWeightsBinary};
    use ferrocarril_core::weights_binary::BinaryWeightLoader;
    use ferrocarril_nn::{Forward, text_encoder::TextEncoder};
    use std::error::Error;

    #[test]
    fn test_text_encoder_real_weights() -> Result<(), Box<dyn Error>> {
        println!("Testing TextEncoder with real weights...");
        
        // Load config and weights
        let config = Config::from_json("/home/sandbox/ferrocarril_weights/config.json")?;
        let loader = BinaryWeightLoader::from_directory("/home/sandbox/ferrocarril_weights")?;
        
        // Test G2P conversion
        let g2p = PhonesisG2P::new("en-us")?;
        let phonemes = g2p.convert("Hello world")?;
        let input_ids = vec![0i64, 1, 2, 3, 0]; // Simplified token sequence
        println!("Phonemes: {}", phonemes);
        
        // Create input tensors
        let input_tensor = Tensor::<i64>::from_data(input_ids.clone(), vec![1, 5]);
        let text_mask = Tensor::<bool>::from_data(vec![false; 5], vec![1, 5]);
        let input_lengths = vec![5];
        
        // Initialize and load TextEncoder
        let mut text_encoder = TextEncoder::new(config.hidden_dim, config.text_encoder_kernel_size, config.n_layer, config.n_token);
        text_encoder.load_weights_binary(&loader, "text_encoder", "module")?;
        println!("TextEncoder weights loaded");
        
        // Run text encoding
        let output = text_encoder.forward(&input_tensor, &input_lengths, &text_mask);
        println!("Output shape: {:?}", output.shape());
        
        // Validate output
        let mut has_nonzero = false;
        for i in 0..std::cmp::min(100, output.data().len()) {
            if output.data()[i].abs() > 1e-6 {
                has_nonzero = true;
                break;
            }
        }
        
        assert!(has_nonzero, "TextEncoder should produce non-zero values");
        assert_eq!(output.shape(), &[1, config.hidden_dim, 5]);
        
        println!("✅ TextEncoder test passed!");
        Ok(())
    }
}
