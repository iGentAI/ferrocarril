//! Direct weight test utility
//!
//! This binary tests loading BERT weights directly by reading the files manually
//! and dumps their contents to verify they are accessible.

use std::error::Error;
use std::path::Path;
use std::fs;
use ferrocarril_nn::bert::transformer::BertConfig;
use ferrocarril_nn::bert::Bert;
use ferrocarril_nn::Forward;
use ferrocarril_core::tensor::Tensor;
use serde_json::Value;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing CustomBERT with direct file access...");
    
    // Get the absolute path to the working directory
    let cwd = std::env::current_dir()?;
    println!("Current working directory: {:?}", cwd);
    
    // Load config directly
    let config_path = "/home/sandbox/ferrocarril_weights/config.json";
    println!("Loading config from: {}", config_path);
    let config_str = fs::read_to_string(config_path)?;
    let config: Value = serde_json::from_str(&config_str)?;
    
    println!("Config loaded successfully!");
    
    // Extract config values
    let vocab_size = config["n_token"].as_u64().unwrap_or(178) as usize;
    let hidden_size = config["plbert"]["hidden_size"].as_u64().unwrap_or(768) as usize;
    let num_attention_heads = config["plbert"]["num_attention_heads"].as_u64().unwrap_or(12) as usize;
    let num_hidden_layers = config["plbert"]["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
    let intermediate_size = config["plbert"]["intermediate_size"].as_u64().unwrap_or(3072) as usize;
    let dropout = config["dropout"].as_f64().unwrap_or(0.1) as f32;
    
    // Create BERT model directly
    let bert_config = BertConfig {
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        intermediate_size,
        max_position_embeddings: 512, // Default value
        dropout_prob: dropout,
    };
    println!("Created BERT config: vocab_size={}, hidden_size={}, num_heads={}, num_layers={}",
            bert_config.vocab_size, 
            bert_config.hidden_size, 
            bert_config.num_attention_heads,
            bert_config.num_hidden_layers);
    
    // Now check if the weight files exist and can be read directly
    let bert_dir = Path::new("/home/sandbox/ferrocarril_weights/model/bert");
    
    if !bert_dir.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("BERT weights directory not found at {:?}", bert_dir)
        )));
    }
    
    println!("\nChecking weight files in {:?}:", bert_dir);
    if let Ok(entries) = fs::read_dir(bert_dir) {
        let mut count = 0;
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() && path.extension().map_or(false, |e| e == "bin") {
                    count += 1;
                    
                    if count <= 5 { // Just show the first 5 for brevity
                        println!("Weight file: {}", path.file_name().unwrap().to_string_lossy());
                        
                        // Get file size
                        if let Ok(metadata) = fs::metadata(&path) {
                            println!("  Size: {} bytes", metadata.len());
                            
                            // Read the file to confirm it's accessible
                            match fs::read(&path) {
                                Ok(data) => {
                                    println!("  Successfully read {} bytes", data.len());
                                    
                                    // Try to interpret first few bytes as f32 values
                                    if data.len() >= 16 { // At least 4 floats
                                        let mut floats = Vec::new();
                                        for i in 0..4 {
                                            let bytes = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
                                            floats.push(f32::from_le_bytes(bytes));
                                        }
                                        println!("  First few values: {:?}", floats);
                                    }
                                }
                                Err(e) => {
                                    println!("  Error reading file: {}", e);
                                }
                            }
                        } else {
                            println!("  Error getting file metadata");
                        }
                    }
                }
            }
        }
        println!("Found {} weight files", count);
    } else {
        println!("Error reading weight directory");
    }
    
    // Initialize a BERT model and run a forward pass with dummy weights
    let bert = Bert::new(bert_config);
    
    // Create a dummy input and run a forward pass
    let input_ids = Tensor::from_data(vec![0, 1, 2, 3, 0], vec![1, 5]);
    
    // Run forward pass
    let output = bert.forward(&input_ids, None, None);
    println!("\nForward pass with uninitialized weights:");
    println!("  Output shape: {:?}", output.shape());
    
    println!("\nDirect file access test completed successfully!");
    
    Ok(())
}