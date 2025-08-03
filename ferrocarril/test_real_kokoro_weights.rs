#[cfg(test)]
mod tests {
    use ferrocarril_core::weights_binary::BinaryWeightLoader;
    
    #[test]
    fn test_real_kokoro_weights() {
        println!("🔍 Testing BinaryWeightLoader with Real Kokoro Weights");
        
        let weights_path = "../real_kokoro_weights";
        
        match BinaryWeightLoader::from_directory(weights_path) {
            Ok(loader) => {
                println!("✅ BinaryWeightLoader created successfully");
                
                let components = loader.list_components();
                println!("📦 Found {} components: {:?}", components.len(), components);
                
                // Test loading a parameter from each component
                for component in &components {
                    match loader.list_parameters(component) {
                        Ok(params) => {
                            println!("  ✓ {}: {} parameters", component, params.len());
                            
                            if let Some(first_param) = params.first() {
                                match loader.load_component_parameter(component, first_param) {
                                    Ok(tensor) => {
                                        println!("    ✅ Loaded {}: shape {:?}, {} values", 
                                                first_param, tensor.shape(), tensor.data().len());
                                    }
                                    Err(e) => {
                                        println!("    ❌ Failed to load {}: {}", first_param, e);
                                        panic!("Parameter loading failed");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!("  ❌ Failed to list parameters for {}: {}", component, e);
                            panic!("Parameter listing failed");
                        }
                    }
                }
                
                println!("\n🎯 REAL KOKORO WEIGHTS VALIDATION: SUCCESS");
            }
            Err(e) => {
                println!("❌ Failed to create BinaryWeightLoader: {}", e);
                panic!("BinaryWeightLoader creation failed: {}", e);
            }
        }
    }
}
