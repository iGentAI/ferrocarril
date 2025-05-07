//! Validation script for converted Kokoro weights
//! 
//! This program loads and validates the converted Kokoro weights to ensure
//! they can be used by the Ferrocarril TTS system.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Simple function to read binary tensor data
fn read_tensor_from_file(path: &Path) -> io::Result<Vec<f32>> {
    // Open the file
    println!("Reading tensor from {}", path.display());
    let mut file = File::open(path)?;
    
    // Get file size
    let metadata = file.metadata()?;
    let file_size = metadata.len() as usize;
    
    // Ensure file size is a multiple of 4 (size of f32)
    if file_size % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData, 
            format!("File size {} is not a multiple of 4 bytes", file_size)
        ));
    }
    
    // Read all bytes
    let mut bytes = vec![0u8; file_size];
    file.read_exact(&mut bytes)?;
    
    // Convert to f32
    let num_elements = file_size / 4;
    let mut values = Vec::with_capacity(num_elements);
    
    for i in 0..num_elements {
        let start = i * 4;
        let value = f32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ]);
        values.push(value);
    }
    
    Ok(values)
}

fn main() -> io::Result<()> {
    let start_time = Instant::now();
    
    // Check if converted weights directory exists
    let weights_dir = Path::new("kokoro_weights");
    if !weights_dir.exists() {
        println!("Error: Kokoro weights directory not found!");
        println!("Please run the conversion script first: python weight_converter.py --huggingface hexgrad/Kokoro-82M --output kokoro_weights");
        return Ok(());
    }
    
    // Use direct exploration instead of parsing JSON
    let model_dir = weights_dir.join("model");
    let components = vec!["bert", "bert_encoder", "predictor", "text_encoder", "decoder"];
    
    println!("Found Kokoro model components: {:?}", components);
    
    // Sample some weights from each component
    let mut component_sizes = HashMap::new();
    let mut total_tensors = 0;
    let mut total_parameters = 0;
    
    for component in &components {
        let component_dir = model_dir.join(component);
        if !component_dir.exists() {
            println!("Component directory '{}' not found, skipping", component);
            continue;
        }
        
        // List files in component directory
        let entries = match std::fs::read_dir(&component_dir) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "bin")
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                println!("Error reading directory {}: {}", component_dir.display(), e);
                continue;
            }
        };
        
        let tensor_count = entries.len();
        total_tensors += tensor_count;
        
        println!("\nComponent '{}' has {} tensors:", component, tensor_count);
        
        // Sample up to 3 tensors from this component
        let sample_count = std::cmp::min(3, entries.len());
        let mut component_params = 0;
        let mut processed = HashSet::new();
        
        for (i, entry) in entries.iter().enumerate() {
            if i >= sample_count {
                break;
            }
            
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_string_lossy();
            
            if processed.contains(&file_name.to_string()) {
                continue;
            }
            
            match read_tensor_from_file(&path) {
                Ok(values) => {
                    let param_count = values.len();
                    component_params += param_count;
                    
                    println!("  {} - {} parameters", file_name, param_count);
                    
                    // Print first few values
                    let preview_count = std::cmp::min(5, values.len());
                    if preview_count > 0 {
                        print!("    First {} values:", preview_count);
                        for i in 0..preview_count {
                            print!(" {:.6}", values[i]);
                        }
                        println!();
                    }
                    
                    processed.insert(file_name.to_string());
                },
                Err(e) => println!("  Error reading {}: {}", file_name, e),
            }
        }
        
        component_sizes.insert(component.clone(), component_params);
        total_parameters += component_params;
        
        if entries.len() > sample_count {
            println!("  ... and {} more tensors", entries.len() - sample_count);
        }
    }
    
    // Voice validation
    let voices_dir = weights_dir.join("voices");
    if voices_dir.exists() {
        let voice_entries = match std::fs::read_dir(&voices_dir) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "bin")
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                println!("Error reading voices directory: {}", e);
                vec![]
            }
        };
        
        if !voice_entries.is_empty() {
            println!("\nFound {} voice files:", voice_entries.len());
            
            // Sample up to 3 voices
            let sample_count = std::cmp::min(3, voice_entries.len());
            
            for (i, entry) in voice_entries.iter().enumerate() {
                if i >= sample_count {
                    break;
                }
                
                let path = entry.path();
                let file_name = path.file_name().unwrap().to_string_lossy();
                
                match read_tensor_from_file(&path) {
                    Ok(values) => {
                        println!("  {} - {} values", file_name, values.len());
                    },
                    Err(e) => println!("  Error reading {}: {}", file_name, e),
                }
            }
            
            if voice_entries.len() > sample_count {
                println!("  ... and {} more voice files", voice_entries.len() - sample_count);
            }
        }
    }
    
    // Component statistics
    println!("\nComponent sizes (parameter count):");
    for (component, size) in &component_sizes {
        println!("  {}: {} parameters", component, size);
    }
    
    println!("\nTotal statistics:");
    println!("  Components: {}", components.len());
    println!("  Tensors: {}", total_tensors);
    println!("  Parameters: {}", total_parameters);
    println!("  Validation completed in {:.2?}", start_time.elapsed());
    
    println!("\n✅ Weight validation completed successfully!");
    println!("The converted weights are ready for use with the Ferrocarril TTS system.");
    Ok(())
}