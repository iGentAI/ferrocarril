//! Test the end-to-end inference pipeline for Ferrocarril TTS
//! 
//! This program loads the Kokoro model from converted binary weights
//! and attempts to synthesize text into speech.

use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Ferrocarril TTS Inference Pipeline Test");
    println!("---------------------------------------");
    
    // Check if the weights directory exists
    let weights_dir = "kokoro_weights";
    if !Path::new(weights_dir).exists() {
        println!("Error: Weights directory '{}' not found!", weights_dir);
        println!("Please run the conversion script first: python weight_converter.py --huggingface hexgrad/Kokoro-82M --output kokoro_weights");
        return Ok(());
    }
    
    // Check if the ferrocarril binary exists
    if !Path::new("ferrocarril/target/debug/ferrocarril").exists() {
        println!("Building ferrocarril binary...");
        
        // Change to ferrocarril directory and build
        std::process::Command::new("cargo")
            .current_dir("ferrocarril")
            .args(&["build"])
            .status()?;
    }
    
    // Run the ferrocarril binary with the weights directory
    println!("Running TTS inference pipeline...");
    let status = std::process::Command::new("./ferrocarril/target/debug/ferrocarril")
        .args(&[weights_dir, "Hello world, this is a test of the Ferrocarril TTS system."])
        .status()?;
    
    if status.success() {
        println!("✅ Inference pipeline test completed successfully!");
    } else {
        println!("❌ Inference pipeline test failed! Exit code: {:?}", status.code());
    }
    
    Ok(())
}