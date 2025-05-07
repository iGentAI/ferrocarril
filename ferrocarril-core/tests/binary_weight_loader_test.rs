//! Test for binary weight loading functionality
//! 
//! This test verifies that the BinaryWeightLoader can load converted model weights
//! Note: This test requires converted model files to be present

#[cfg(feature = "weights")]
mod tests {
    use ferrocarril_core::weights_binary::BinaryWeightLoader;
    use ferrocarril_core::tensor::Tensor;
    use std::path::Path;
    use std::fs;
    use std::io::Write;

    const TEST_DIR: &str = "test_weights";
    const CONFIG_JSON: &str = r#"{
        "format_version": "1.0",
        "original_file": "test_model.pth",
        "components": {
            "test_component": {
                "parameters": {
                    "weight": {
                        "file": "test_component/weight.bin",
                        "shape": [2, 3],
                        "dtype": "float32",
                        "byte_size": 24
                    },
                    "bias": {
                        "file": "test_component/bias.bin",
                        "shape": [3],
                        "dtype": "float32",
                        "byte_size": 12
                    }
                }
            }
        }
    }"#;

    fn setup_test_dir() {
        // Create test directory
        let test_path = Path::new(TEST_DIR);
        let component_path = test_path.join("test_component");
        
        if !test_path.exists() {
            fs::create_dir_all(&component_path).expect("Failed to create test directory");
        }
        
        // Write metadata.json
        let metadata_file = test_path.join("metadata.json");
        fs::write(&metadata_file, CONFIG_JSON).expect("Failed to write metadata");
        
        // Create weight tensor data
        let weight_data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight_file = component_path.join("weight.bin");
        let mut file = fs::File::create(&weight_file).expect("Failed to create weight file");
        
        // Convert to bytes and write
        let weight_bytes = unsafe { 
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const u8,
                weight_data.len() * std::mem::size_of::<f32>()
            )
        };
        file.write_all(weight_bytes).expect("Failed to write weight data");
        
        // Create bias tensor data
        let bias_data: &[f32] = &[7.0, 8.0, 9.0];
        let bias_file = component_path.join("bias.bin");
        let mut file = fs::File::create(&bias_file).expect("Failed to create bias file");
        
        // Convert to bytes and write
        let bias_bytes = unsafe { 
            std::slice::from_raw_parts(
                bias_data.as_ptr() as *const u8,
                bias_data.len() * std::mem::size_of::<f32>()
            )
        };
        file.write_all(bias_bytes).expect("Failed to write bias data");
    }
    
    fn cleanup_test_dir() {
        let test_path = Path::new(TEST_DIR);
        if test_path.exists() {
            fs::remove_dir_all(test_path).expect("Failed to clean up test directory");
        }
    }

    #[test]
    fn test_binary_weight_loader() {
        // Set up test directory with mock data
        setup_test_dir();
        
        // Create the loader
        let mut loader = BinaryWeightLoader::from_directory(TEST_DIR)
            .expect("Failed to create BinaryWeightLoader");
        
        // Check loader is not empty
        assert!(!loader.is_empty(), "Loader should not be empty");
        
        // List components
        let components = loader.list_components();
        assert_eq!(components.len(), 1, "Should have 1 component");
        assert_eq!(components[0], "test_component", "Component should be 'test_component'");
        
        // List parameters
        let parameters = loader.list_parameters("test_component")
            .expect("Failed to list parameters");
        assert_eq!(parameters.len(), 2, "Should have 2 parameters");
        assert!(parameters.contains(&"weight".to_string()), "Should have 'weight' parameter");
        assert!(parameters.contains(&"bias".to_string()), "Should have 'bias' parameter");
        
        // Load weight tensor
        let weight = loader.load_tensor("test_component", "weight")
            .expect("Failed to load weight tensor");
        
        // Verify shape
        assert_eq!(weight.shape(), &[2, 3], "Weight shape should be [2, 3]");
        
        // Verify values
        assert_eq!(weight.data()[0], 1.0);
        assert_eq!(weight.data()[1], 2.0);
        assert_eq!(weight.data()[2], 3.0);
        assert_eq!(weight.data()[3], 4.0);
        assert_eq!(weight.data()[4], 5.0);
        assert_eq!(weight.data()[5], 6.0);
        
        // Load bias tensor
        let bias = loader.load_tensor("test_component", "bias")
            .expect("Failed to load bias tensor");
        
        // Verify shape
        assert_eq!(bias.shape(), &[3], "Bias shape should be [3]");
        
        // Verify values
        assert_eq!(bias.data()[0], 7.0);
        assert_eq!(bias.data()[1], 8.0);
        assert_eq!(bias.data()[2], 9.0);
        
        // Clean up
        cleanup_test_dir();
    }
    
    // Uncomment and modify to test with real converted weights
    /*
    #[test]
    #[ignore] // Ignore by default to avoid requiring external files
    fn test_with_real_weights() {
        // Test with real converted weights
        // Modify the path to your converted weights directory
        let weights_dir = "path/to/converted/weights";
        
        if !Path::new(weights_dir).exists() {
            println!("Skipping test - converted weights not found at {}", weights_dir);
            return;
        }
        
        let mut loader = BinaryWeightLoader::from_directory(weights_dir)
            .expect("Failed to create BinaryWeightLoader");
            
        // List available components
        let components = loader.list_components();
        println!("Found components: {:?}", components);
        
        // Try to load a voice
        if loader.list_voices().is_ok() {
            let voices = loader.list_voices().unwrap();
            println!("Found voices: {:?}", voices);
            
            if !voices.is_empty() {
                let voice = loader.load_voice(&voices[0]).expect("Failed to load voice");
                println!("Loaded voice with shape: {:?}", voice.shape());
            }
        }
        
        // Load some tensors from the first component
        if !components.is_empty() {
            let component = &components[0];
            let parameters = loader.list_parameters(component).expect("Failed to list parameters");
            println!("Component {} has parameters: {:?}", component, parameters);
            
            if !parameters.is_empty() {
                let param = &parameters[0];
                let tensor = loader.load_tensor(component, param).expect("Failed to load tensor");
                println!("Parameter {} has shape: {:?}", param, tensor.shape());
            }
        }
    }
    */
}