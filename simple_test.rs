//! Simple binary data loading test
//! 
//! This program demonstrates loading binary weight data without needing
//! the full Ferrocarril library.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

// Simple function to read f32 values from a binary file
fn read_f32_from_file(path: &Path) -> io::Result<Vec<f32>> {
    // Open the file
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
    // Check if converted files exist
    let weights_dir = Path::new("test_data/converted");
    let weight_file = weights_dir.join("linear/linear_weight.bin");
    let bias_file = weights_dir.join("linear/linear_bias.bin");
    
    if !weights_dir.exists() || !weight_file.exists() || !bias_file.exists() {
        println!("Converted weight files not found.");
        println!("Please run conversion_test.py first to create test data.");
        return Ok(());
    }
    
    println!("Reading weights from {}...", weight_file.display());
    match read_f32_from_file(&weight_file) {
        Ok(values) => {
            println!("Successfully read {} f32 values", values.len());
            println!("First 5 values: {:?}", &values[..std::cmp::min(5, values.len())]);
        },
        Err(e) => println!("Error reading weights: {}", e),
    }
    
    println!("\nReading bias from {}...", bias_file.display());
    match read_f32_from_file(&bias_file) {
        Ok(values) => {
            println!("Successfully read {} f32 values", values.len());
            println!("All values: {:?}", values);
        },
        Err(e) => println!("Error reading bias: {}", e),
    }
    
    println!("\nSimple test completed successfully!");
    Ok(())
}