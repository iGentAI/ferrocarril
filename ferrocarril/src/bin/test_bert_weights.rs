//! Test binary for loading real BERT weights
//!
//! This is a simple binary to test loading the real BERT weights
//! and running inference with them.

use std::error::Error;
use std::path::Path;
use ferrocarril_core::{Config, tensor::Tensor};
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_nn::bert::{CustomAlbert, CustomAlbertConfig}; // Fixed imports
use ferrocarril_core::LoadWeightsBinary;
use ferrocarril_nn::Forward; // Added missing Forward trait

fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing CustomAlbert with real weights...");
    
    // Get the absolute path to the working directory
    let cwd = std::env::current_dir()?;
    println!("Current working directory: {:?}", cwd);
    
    // Load the model config
    let config_path = "/home/sandbox/ferrocarril_weights/config.json";
    println!("Loading config from: {}", config_path);
    let config = Config::from_json(config_path)?;
    
    // Create CustomAlbert config from model config - FIXED TYPE
    let bert_config = CustomAlbertConfig {
        vocab_size: config.n_token,
        embedding_size: 128, // Albert factorized embedding size
        hidden_size: config.plbert.hidden_size,
        num_attention_heads: config.plbert.num_attention_heads,
        num_hidden_layers: config.plbert.num_hidden_layers,
        intermediate_size: config.plbert.intermediate_size,
        max_position_embeddings: 512, // Default value for ALBERT
    };
    println!("Created CustomAlbert config: vocab_size={}, hidden_size={}, num_heads={}, num_layers={}",
            bert_config.vocab_size, 
            bert_config.hidden_size, 
            bert_config.num_attention_heads,
            bert_config.num_hidden_layers);
    
    // Initialize CustomAlbert
    let mut bert = CustomAlbert::new(bert_config);
    println!("Initialized CustomAlbert model");
    
    // Load weights
    let weights_path = "/home/sandbox/ferrocarril_weights/model";
    println!("Loading weights from: {}", weights_path);
    let loader = BinaryWeightLoader::from_directory(weights_path)?;
    println!("Created weight loader from: {}", weights_path);
    
    // List the components to verify the loader is working
    let components = loader.list_components();
    println!("Available components: {:?}", components);
    
    // Load BERT weights
    bert.load_weights_binary(&loader, "bert", "module")?;
    println!("Successfully loaded BERT weights");
    
    // Create a simple input for testing
    let input_text = "Hello world";
    println!("Testing with input text: \"{}\"", input_text);
    
    // Create token IDs for the input text
    let input_ids = Tensor::from_data(vec![0, 1, 2, 3, 0], vec![1, 5]);
    
    // Create attention mask in correct [B, T] format for CustomAlbert
    let attention_mask = Tensor::from_data(vec![1i64, 1, 1, 1, 0], vec![1, 5]); // 1=attend, 0=mask
    
    // Run forward pass - FIXED: 2 arguments only (input_ids, attention_mask)
    let output = bert.forward(&input_ids, Some(&attention_mask));
    
    // Check output shape 
    println!("Output shape: {:?}", output.shape());
    
    // Check if output contains non-zero values
    let mut has_nonzero = false;
    let mut sum = 0.0;
    let mut count = 0;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    
    for i in 0..output.data().len() {
        let val = output.data()[i];
        if val != 0.0 {
            has_nonzero = true;
        }
        sum += val;
        count += 1;
        min = min.min(val);
        max = max.max(val);
    }
    
    let avg = sum / count as f32;
    
    println!("Output statistics:");
    println!("  Contains non-zero: {}", has_nonzero);
    println!("  Average value: {}", avg);
    println!("  Min value: {}", min);
    println!("  Max value: {}", max);
    if count > 0 {
        let slice_end = std::cmp::min(count, 10);
        println!("  First few values: {:?}", &output.data()[0..slice_end]);
    }
    
    if has_nonzero {
        println!("✅ Test passed: CustomAlbert produced non-zero outputs with real weights");
    } else {
        println!("❌ Test failed: CustomAlbert produced all zeros");
    }
    
    Ok(())
}