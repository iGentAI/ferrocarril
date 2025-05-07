//! Simple utility to check if weights can be loaded
//! 
//! This binary directly attempts to load weights from the ferrocarril_weights directory
//! and reports any issues it finds.

use std::error::Error;
use std::path::Path;
use ferrocarril_core::weights_binary::BinaryWeightLoader;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Checking if weights can be loaded...");
    
    // Get the absolute path to the working directory
    let cwd = std::env::current_dir()?;
    println!("Current working directory: {:?}", cwd);
    
    // Try both relative and absolute paths
    let relative_path = "ferrocarril_weights/model";
    let absolute_path = "/home/sandbox/ferrocarril_weights/model";
    
    println!("\nAttempting to load weights from relative path: {}", relative_path);
    match BinaryWeightLoader::from_directory(relative_path) {
        Ok(loader) => {
            let components = loader.list_components();
            println!("✅ Successfully loaded weights!");
            println!("Found components: {:?}", components);
            
            // Try to load a specific tensor to verify full functionality
            if let Some(component) = components.first() {
                match loader.list_parameters(component) {
                    Ok(params) => {
                        println!("Parameters in component {}: {}", component, params.len());
                        if let Some(param) = params.first() {
                            match loader.load_component_parameter(component, param) {
                                Ok(tensor) => {
                                    println!("Successfully loaded tensor: {} with shape {:?}", param, tensor.shape());
                                }
                                Err(e) => {
                                    println!("❌ Failed to load tensor: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to list parameters: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to load weights from relative path: {}", e);
        }
    }
    
    println!("\nAttempting to load weights from absolute path: {}", absolute_path);
    match BinaryWeightLoader::from_directory(absolute_path) {
        Ok(loader) => {
            let components = loader.list_components();
            println!("✅ Successfully loaded weights!");
            println!("Found components: {:?}", components);
            
            // Try to load a specific tensor to verify full functionality
            if let Some(component) = components.first() {
                match loader.list_parameters(component) {
                    Ok(params) => {
                        println!("Parameters in component {}: {}", component, params.len());
                        if let Some(param) = params.first() {
                            match loader.load_component_parameter(component, param) {
                                Ok(tensor) => {
                                    println!("Successfully loaded tensor: {} with shape {:?}", param, tensor.shape());
                                }
                                Err(e) => {
                                    println!("❌ Failed to load tensor: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("❌ Failed to list parameters: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to load weights from absolute path: {}", e);
        }
    }
    
    println!("\nChecking if metadata.json exists and is readable:");
    let relative_metadata_path = Path::new("ferrocarril_weights/model/metadata.json");
    let absolute_metadata_path = Path::new("/home/sandbox/ferrocarril_weights/model/metadata.json");
    
    println!("Relative path exists: {}", relative_metadata_path.exists());
    println!("Absolute path exists: {}", absolute_metadata_path.exists());
    
    // Try to read the metadata file directly
    if absolute_metadata_path.exists() {
        match std::fs::read_to_string(absolute_metadata_path) {
            Ok(content) => {
                println!("✅ Successfully read metadata file!");
                println!("Metadata file size: {} bytes", content.len());
                println!("First 100 characters: {}", &content[0..100.min(content.len())]);
            }
            Err(e) => {
                println!("❌ Failed to read metadata file: {}", e);
            }
        }
    }
    
    Ok(())
}