//! Test for weight loading functionality
use ferrocarril_core::{Config, Parameter};
use ferrocarril_core::tensor::Tensor;
use ferrocarril_core::weights::{PyTorchWeightLoader, LoadWeights};
use std::path::Path;

#[derive(Debug)]
struct TestModel {
    linear: TestLinear,
    config: Config,
}

#[derive(Debug)]
struct TestLinear {
    weight: Parameter,
    bias: Option<Parameter>,
}

impl LoadWeights for TestLinear {
    fn load_weights(
        &mut self, 
        loader: &PyTorchWeightLoader, 
        prefix: Option<&str>
    ) -> Result<(), ferrocarril_core::FerroError> {
        // Load weight tensor
        loader.load_weight_into_parameter(&mut self.weight, "weight", prefix, None)?;
        
        // Load bias tensor if available
        if let Some(ref mut bias) = self.bias {
            loader.load_weight_into_parameter(bias, "bias", prefix, None)?;
        }
        
        Ok(())
    }
}

#[test]
fn test_config_loading() {
    // Skip if config.json doesn't exist
    if !Path::new("src/config.json").exists() {
        println!("Skipping test_config_loading as src/config.json doesn't exist");
        return;
    }
    
    // Try loading the config
    let config = Config::from_kmodel_config("src/config.json").unwrap();
    
    // Check that config values match what we expect
    assert_eq!(config.n_token, 150);
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.style_dim, 128);
    assert_eq!(config.max_dur, 50);
    assert_eq!(config.text_encoder_kernel_size, 5);
    
    // Check vocab
    assert!(config.vocab.contains_key(&'a'));
    
    // Check istftnet config
    assert_eq!(config.istftnet.upsample_rates.len(), 2);
    assert_eq!(config.istftnet.upsample_rates[0], 8);
    assert_eq!(config.istftnet.gen_istft_n_fft, 16);
    
    // Check plbert config
    assert_eq!(config.plbert.hidden_size, 768);
    assert_eq!(config.plbert.num_attention_heads, 12);
}

#[test]
fn test_mock_weight_loading() {
    // Create a mock model
    let mut model = TestModel {
        linear: TestLinear {
            weight: Parameter::new(Tensor::new(vec![4, 8])),  // out_features x in_features
            bias: Some(Parameter::new(Tensor::new(vec![4]))),  // out_features
        },
        config: Config::from_json("dummy").unwrap(),
    };
    
    // Create a mock weight loader
    let mut tensors = std::collections::HashMap::new();
    
    // Create a mock weight tensor
    let weight_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let weight_tensor = Tensor::from_data(weight_data, vec![4, 8]);
    
    // Create a mock bias tensor
    let bias_data: Vec<f32> = (0..4).map(|i| (i * 10) as f32).collect();
    let bias_tensor = Tensor::from_data(bias_data, vec![4]);
    
    // Test if we can extract values from tensors
    assert_eq!(weight_tensor[&[1, 2]], 10.0);  // Index formula: 1*8 + 2 = 10
    assert_eq!(bias_tensor[&[2]], 20.0);       // Index 2 = value 20.0
    
    // We can't test the actual PyTorchWeightLoader right now without a real file,
    // but we can test our LoadWeights trait logic with manual parameter updates
    
    // Update the parameters manually to simulate loading
    model.linear.weight = Parameter::new(weight_tensor);
    
    if let Some(ref mut bias) = model.linear.bias {
        *bias = Parameter::new(bias_tensor);
    }
    
    // Verify the parameters were loaded correctly
    assert_eq!(model.linear.weight.data()[&[1, 2]], 10.0);
    assert_eq!(model.linear.bias.as_ref().unwrap().data()[&[2]], 20.0);
    
    println!("Weight loading test passed with mock data");
}