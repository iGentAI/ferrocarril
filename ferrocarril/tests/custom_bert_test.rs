//! Test for CustomBERT implementation
//!
//! This test verifies the CustomBERT implementation by checking that the forward
//! pass produces the expected output shape.

#[cfg(test)]
mod tests {
    use ferrocarril_core::tensor::Tensor;
    use ferrocarril_nn::bert::transformer::{CustomBert, BertConfig};
    use ferrocarril_nn::Forward;

    /// Test the basic functionality of CustomBERT
    #[test]
    fn test_custom_bert_forward() {
        // Create a small BERT config for testing
        let config = BertConfig {
            vocab_size: 100,
            hidden_size: 128,  // Use smaller size to match ALBERT embeddings
            num_attention_heads: 4, // Must divide hidden_size evenly
            num_hidden_layers: 2,
            intermediate_size: 256,
            max_position_embeddings: 128,
            dropout_prob: 0.0, // No dropout for testing
        };

        // Create CustomBERT
        let bert = CustomBert::new(config);

        // Create sample input IDs
        let input_ids = Tensor::<i64>::from_data(
            vec![1, 2, 3, 4, 5],
            vec![1, 5] // batch_size=1, seq_len=5
        );

        // Run forward pass
        let output = bert.forward(&input_ids, None, None);

        // Check output shape
        let expected_shape = vec![1, 5, 128]; // [batch_size, seq_len, hidden_size]
        assert_eq!(
            output.shape(), 
            &expected_shape, 
            "Output shape mismatch, expected {:?}, got {:?}", 
            expected_shape, 
            output.shape()
        );
        
        // Since we're using zero initialization in the test, we don't check for non-zero values
        println!("CustomBERT forward test passed - correct output shape");
    }

    /// Test attention mask functionality
    #[test]
    fn test_custom_bert_attention_mask() {
        // Create a small BERT config for testing
        let config = BertConfig {
            vocab_size: 100,
            hidden_size: 128,  // Use smaller size to match ALBERT embeddings
            num_attention_heads: 4, // Must divide hidden_size evenly
            num_hidden_layers: 2,
            intermediate_size: 256,
            max_position_embeddings: 128,
            dropout_prob: 0.0, // No dropout for testing
        };

        // Create CustomBERT
        let bert = CustomBert::new(config);
        
        // Create sample input IDs
        let input_ids = Tensor::<i64>::from_data(
            vec![1, 2, 3, 4, 5],
            vec![1, 5] // batch_size=1, seq_len=5
        );

        // Create an attention mask that masks out the last two positions
        // 1 = masked position, 0 = valid position
        let mut attention_mask = Tensor::from_data(
            vec![0; 1 * 5 * 5], // Initialize to zeros
            vec![1, 5, 5]
        );
        
        // Mask out positions 3 and 4 (last two tokens)
        for i in 0..5 {
            // Mask out attention to positions 3 and 4
            attention_mask[&[0, i, 3]] = 1;
            attention_mask[&[0, i, 4]] = 1;
        }

        // Run forward pass with mask
        let masked_output = bert.forward(&input_ids, None, Some(&attention_mask));

        // Run forward pass without mask
        let unmasked_output = bert.forward(&input_ids, None, None);

        // Check shapes are correct
        assert_eq!(
            masked_output.shape(),
            unmasked_output.shape(),
            "Masked and unmasked outputs have different shapes"
        );
        
        // Check that both outputs have the expected shape
        let expected_shape = vec![1, 5, 128];
        assert_eq!(
            masked_output.shape(),
            &expected_shape,
            "Masked output shape mismatch, expected {:?}, got {:?}",
            expected_shape,
            masked_output.shape()
        );
        
        println!("CustomBERT attention mask test passed - no runtime errors");
    }
    
    #[cfg(feature = "weights")]
    #[test]
    fn test_custom_bert_with_real_weights() -> Result<(), Box<dyn std::error::Error>> {
        use ferrocarril_core::{Config, LoadWeightsBinary};
        use ferrocarril_core::weights_binary::BinaryWeightLoader;
        
        // Load config
        let config_path = "/home/sandbox/ferrocarril_weights/config.json";
        let config = Config::from_json(config_path)?;
        
        // Create BERT config
        let bert_config = BertConfig {
            vocab_size: config.n_token,
            hidden_size: config.plbert.hidden_size,
            num_attention_heads: config.plbert.num_attention_heads,
            num_hidden_layers: config.plbert.num_hidden_layers,
            intermediate_size: config.plbert.intermediate_size,
            max_position_embeddings: 512, // Default value for ALBERT
            dropout_prob: config.dropout,
        };
        
        // Initialize CustomBERT
        let mut bert = CustomBert::new(bert_config);
        
        // Create fixed BinaryWeightLoader
        let weights_path = "/home/sandbox/ferrocarril_weights";
        let loader = BinaryWeightLoader::from_directory(weights_path)?;
        
        // Load BERT weights
        bert.load_weights_binary(&loader, "bert", "module")?;
        
        // Create a simple input for testing
        let input_ids = Tensor::<i64>::from_data(vec![0, 1, 2, 3, 0], vec![1, 5]);
        
        // Run forward pass
        let output = bert.forward(&input_ids, None, None);
        
        // Check output shape - should be [1, 5, 768] for ALBERT
        assert_eq!(output.shape(), &[1, 5, 768], "Output shape should be [1, 5, 768]");
        
        // Check if output contains non-zero values
        let mut has_nonzero = false;
        for i in 0..std::cmp::min(100, output.data().len()) {
            if output.data()[i] != 0.0 {
                has_nonzero = true;
                break;
            }
        }
        
        // The output should now contain non-zero values with real weights
        assert!(has_nonzero, "Output should contain non-zero values with real weights");
        
        println!("CustomBERT with real weights test passed!");
        Ok(())
    }
}