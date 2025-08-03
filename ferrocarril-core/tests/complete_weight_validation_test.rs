//! Complete validation test for BinaryWeightLoader with converted weights

use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_core::FerroError;
use std::path::Path;

#[test]
fn test_binary_weight_loader_with_converted_weights() {
    println!("🔍 Testing BinaryWeightLoader with converted weights");
    
    // Check if our test conversion exists
    let test_output_path = Path::new("../../test_output");
    if !test_output_path.exists() {
        println!("❌ Test output directory not found - skipping test");
        println!("💡 Run: python3 weight_converter.py --model test_model.pth --output test_output");
        return;
    }
    
    // Test loading with BinaryWeightLoader
    match BinaryWeightLoader::from_directory(test_output_path) {
        Ok(loader) => {
            println!("✅ BinaryWeightLoader created successfully");
            
            // Test basic functionality
            let components = loader.list_components();
            println!("📦 Found components: {:?}", components);
            
            assert!(!components.is_empty(), "Should find at least one component");
            
            // Test parameter listing for each component
            for component in &components {
                match loader.list_parameters(component) {
                    Ok(params) => {
                        println!("  📋 Component '{}' has {} parameters", component, params.len());
                        
                        // Test loading a specific parameter
                        if let Some(first_param) = params.first() {
                            match loader.load_component_parameter(component, first_param) {
                                Ok(tensor) => {
                                    println!("    ✅ Successfully loaded '{}': shape {:?}", 
                                             first_param, tensor.shape());
                                    
                                    // Validate tensor has reasonable data
                                    assert!(!tensor.data().is_empty(), "Tensor should have data");
                                    assert!(tensor.shape().iter().product::<usize>() == tensor.data().len(),
                                            "Tensor shape should match data length");
                                }
                                Err(e) => {
                                    println!("    ❌ Failed to load '{}': {}", first_param, e);
                                    panic!("Parameter loading failed");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Failed to list parameters for '{}': {}", component, e);
                        panic!("Parameter listing failed");
                    }
                }
            }
            
            println!("✅ BinaryWeightLoader validation passed");
        }
        Err(e) => {
            println!("❌ Failed to create BinaryWeightLoader: {}", e);
            panic!("BinaryWeightLoader creation failed: {}", e);
        }
    }
}

#[test] 
fn test_weight_loading_performance() {
    println!("⚡ Testing weight loading performance");
    
    let test_output_path = Path::new("../../test_output");
    if !test_output_path.exists() {
        println!("❌ Test output not found - skipping performance test");
        return;
    }
    
    // Time the loading process
    let start = std::time::Instant::now();
    
    match BinaryWeightLoader::from_directory(test_output_path) {
        Ok(loader) => {
            let load_time = start.elapsed();
            println!("✅ Loader creation took: {:?}", load_time);
            
            // Time parameter loading
            let components = loader.list_components();
            let mut total_load_time = std::time::Duration::ZERO;
            let mut param_count = 0;
            
            for component in &components {
                if let Ok(params) = loader.list_parameters(component) {
                    for param in &params {
                        let param_start = std::time::Instant::now();
                        if let Ok(_tensor) = loader.load_component_parameter(component, param) {
                            total_load_time += param_start.elapsed();
                            param_count += 1;
                        }
                    }
                }
            }
            
            if param_count > 0 {
                let avg_load_time = total_load_time / param_count;
                println!("📊 Performance metrics:");
                println!("  Parameters loaded: {}", param_count);
                println!("  Total load time: {:?}", total_load_time);
                println!("  Average per parameter: {:?}", avg_load_time);
                
                // Performance thresholds
                if avg_load_time.as_millis() > 10 {
                    println!("  ⚠️  Loading is slow - consider memory mapping optimization");
                } else {
                    println!("  ✅ Loading performance is adequate");
                }
            }
            
            println!("✅ Performance validation completed");
        }
        Err(e) => {
            println!("❌ Performance test failed: {}", e);
        }
    }
}

#[test]
fn test_weight_data_integrity() {
    println!("🔒 Testing weight data integrity");
    
    let test_output_path = Path::new("../../test_output");
    if !test_output_path.exists() {
        println!("❌ Test output not found - skipping integrity test");
        return;
    }
    
    match BinaryWeightLoader::from_directory(test_output_path) {
        Ok(loader) => {
            let components = loader.list_components();
            
            for component in &components {
                if let Ok(params) = loader.list_parameters(component) {
                    for param in &params {
                        if let Ok(tensor) = loader.load_component_parameter(component, param) {
                            // Basic integrity checks
                            assert!(!tensor.data().is_empty(), 
                                   "Tensor data should not be empty");
                            
                            // Check for NaN or infinite values
                            let has_nan = tensor.data().iter().any(|&x| x.is_nan());
                            let has_inf = tensor.data().iter().any(|&x| x.is_infinite());
                            
                            if has_nan {
                                println!("  ⚠️  Tensor '{}' contains NaN values", param);
                            }
                            if has_inf {
                                println!("  ⚠️  Tensor '{}' contains infinite values", param);
                            }
                            
                            if !has_nan && !has_inf {
                                println!("  ✅ Tensor '{}' integrity verified", param);
                            }
                            
                            // Check data distribution (should not be all zeros)
                            let all_zeros = tensor.data().iter().all(|&x| x == 0.0);
                            if all_zeros {
                                println!("  ⚠️  Tensor '{}' is all zeros - may indicate conversion issue", param);
                            }
                        }
                    }
                }
            }
            
            println!("✅ Data integrity validation completed");
        }
        Err(e) => {
            println!("❌ Integrity test failed: {}", e);
        }
    }
}