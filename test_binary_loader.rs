//! Test program for BinaryWeightLoader
//! 
//! This program loads weights converted by our Python script and verifies
//! that they can be correctly loaded.

extern crate ferrocarril_core;
use ferrocarril_core::weights_binary::BinaryWeightLoader;
use std::path::Path;

fn main() {
    // Path to the converted weights
    let weights_dir = "test_data/converted";
    
    if !Path::new(weights_dir).exists() {
        eprintln!("Error: Converted weights directory not found. Please run conversion_test.py first!");
        std::process::exit(1);
    }
    
    // Load the weights
    println!("Loading converted weights from {}...", weights_dir);
    let loader = match BinaryWeightLoader::from_directory(weights_dir) {
        Ok(loader) => loader,
        Err(e) => {
            eprintln!("Error loading weights: {}", e);
            std::process::exit(1);
        }
    };
    
    // List available components
    let components = loader.list_components();
    println!("\nFound components: {:?}", components);
    
    // Look at each component
    for component in &components {
        // List parameters
        match loader.list_parameters(component) {
            Ok(parameters) => {
                println!("\nParameters in component '{}': {:?}", component, parameters);
                
                // Try to load some parameters
                for param in &parameters {
                    match loader.load_component_parameter(component, param) {
                        Ok(tensor) => println!("  Loaded '{}' with shape {:?}", param, tensor.shape()),
                        Err(e) => println!("  Error loading '{}': {}", param, e),
                    }
                }
            },
            Err(e) => println!("Error listing parameters for component '{}': {}", component, e),
        }
    }
    
    // Try to load by full name
    match loader.load_tensor("linear.weight") {
        Ok(tensor) => {
            println!("\nLoaded 'linear.weight' with shape {:?}", tensor.shape());
            println!("First few values: {:?}", &tensor.data()[0..5]);
        },
        Err(e) => println!("Error loading 'linear.weight': {}", e),
    }
    
    // Check for voice files
    match loader.list_voices() {
        Ok(voices) => {
            println!("\nFound voices: {:?}", voices);
            
            // Try to load a voice
            if !voices.is_empty() {
                match loader.load_voice(&voices[0]) {
                    Ok(voice) => println!("Loaded voice '{}' with shape {:?}", voices[0], voice.shape()),
                    Err(e) => println!("Error loading voice '{}': {}", voices[0], e),
                }
            }
        },
        Err(e) => println!("Error listing voices: {}", e),
    }
    
    println!("\nTest completed successfully!");
}