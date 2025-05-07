#[cfg(feature = "weights")]
mod adain_test {
    use ferrocarril_core::tensor::Tensor;
    use ferrocarril_core::weights_binary::BinaryWeightLoader;
    use ferrocarril_core::LoadWeightsBinary;
    use ferrocarril_nn::adain::{AdaIN1d, InstanceNorm1d};
    use ferrocarril_nn::linear::Linear;
    use std::error::Error;

    #[test]
    fn test_adain_with_real_weights() -> Result<(), Box<dyn Error>> {
        // Create an AdaIN1d instance with parameters from vocoder
        let style_dim = 128;
        let num_features = 256;  // Common channel dimension in vocoder
        
        let mut adain = AdaIN1d::new(style_dim, num_features);
        
        // Load real weights from model files
        println!("Loading real weights from ferrocarril_weights...");
        let loader = BinaryWeightLoader::from_directory("../ferrocarril_weights")?;
        
        // Try to load weights from one of the vocoder AdaIN modules
        // The exact component path might vary depending on the model structure
        adain.load_weights_binary(&loader, "decoder", "module.generator.resblocks.0.adain1.0")?;
        println!("Weights loaded successfully");
        
        // Create realistic input tensors
        let batch_size = 1;
        let time_dimension = 32;
        
        // Input tensor: [B, C, T]
        let x = Tensor::from_data(
            vec![0.1; batch_size * num_features * time_dimension], 
            vec![batch_size, num_features, time_dimension]
        );
        
        // Style tensor: [B, style_dim]
        let style = Tensor::from_data(
            vec![0.1; batch_size * style_dim],
            vec![batch_size, style_dim]
        );
        
        // Run the forward pass
        println!("Running AdaIN forward pass...");
        let output = adain.forward(&x, &style);
        
        // Verify output shape
        println!("Output shape: {:?}", output.shape());
        assert_eq!(output.shape(), x.shape(), "Output shape should match input shape");
        
        // Verify non-zero output (functional correctness check)
        assert!(!output.data().iter().all(|&v| v.abs() < 1e-6),
                "AdaIN output is all close to zero - component may be functionally dead!");
        
        // Normalize input for comparison
        let input_mean = x.data().iter().sum::<f32>() / x.data().len() as f32;
        let input_variance = x.data().iter()
            .map(|&val| (val - input_mean).powi(2))
            .sum::<f32>() / x.data().len() as f32;
        
        let output_mean = output.data().iter().sum::<f32>() / output.data().len() as f32;
        let output_variance = output.data().iter()
            .map(|&val| (val - output_mean).powi(2))
            .sum::<f32>() / output.data().len() as f32;
        
        println!("Input - Mean: {}, Variance: {}", input_mean, input_variance);
        println!("Output - Mean: {}, Variance: {}", output_mean, output_variance);
        
        // With non-trivial inputs and style, output should have different statistics than input
        assert!(
            (output_mean - input_mean).abs() > 1e-2 || (output_variance - input_variance).abs() > 1e-2,
            "AdaIN should transform the input statistics!"
        );
        
        // Additional test to verify style conditioning is working
        // Create a different style tensor
        let style2 = Tensor::from_data(
            vec![0.5; batch_size * style_dim],
            vec![batch_size, style_dim]
        );
        
        let output2 = adain.forward(&x, &style2);
        
        // Calculate mean and variance of second output
        let output2_mean = output2.data().iter().sum::<f32>() / output2.data().len() as f32;
        let output2_variance = output2.data().iter()
            .map(|&val| (val - output2_mean).powi(2))
            .sum::<f32>() / output2.data().len() as f32;
        
        println!("Output with style2 - Mean: {}, Variance: {}", output2_mean, output2_variance);
        
        // Different style should give different output statistics
        assert!(
            (output_mean - output2_mean).abs() > 1e-2 || (output_variance - output2_variance).abs() > 1e-2, 
            "AdaIN with different style should produce different results!"
        );
        
        println!("Test completed successfully!");
        
        Ok(())
    }
    
    // Add a test specifically to verify the impact of affine=false
    #[test]
    fn test_adain_with_affine_false() -> Result<(), Box<dyn Error>> {
        // Create a simple AdaIN1d implementation with affine=false in the InstanceNorm1d layer
        let style_dim = 64;
        let num_features = 128;
        
        // Create an AdaIN1d instance which will have affine=false after our fix
        let mut adain = AdaIN1d::new(style_dim, num_features);

        // Manually create a style conditioning network with easily predictable values
        // This will make debugging easier by making the gamma/beta values more controllable
        let mut style_fc_weight_data = vec![0.0; style_dim * num_features * 2];
        let mut style_fc_bias_data = vec![0.0; num_features * 2];
        
        // Set all gamma weights to 1.0 and all beta weights to 0.1
        // This will make the style vector directly influence gamma and beta values
        for i in 0..style_dim {
            // First half is gamma weights
            for j in 0..num_features {
                style_fc_weight_data[i * num_features * 2 + j] = 0.2; // Gamma weights
            }
            
            // Second half is beta weights
            for j in 0..num_features {
                style_fc_weight_data[i * num_features * 2 + num_features + j] = 0.2; // Beta weights
            }
        }
        
        // Replace the fc layer's weights and bias
        let style_fc_weight = Tensor::from_data(style_fc_weight_data, vec![num_features * 2, style_dim]);
        let style_fc_bias = Tensor::from_data(style_fc_bias_data, vec![num_features * 2]);
        
        // Update the fc layer
        adain.fc.load_weight_bias(&style_fc_weight, Some(&style_fc_bias))
            .expect("Failed to update style fc layer");
        
        // Create test inputs
        let batch_size = 1;
        let time_dimension = 16;
        
        // Create a patterned input for testing
        let mut input_data = vec![0.0; batch_size * num_features * time_dimension];
        for i in 0..input_data.len() {
            input_data[i] = (i % 10) as f32 * 0.1;
        }
        
        let x = Tensor::from_data(
            input_data.clone(), 
            vec![batch_size, num_features, time_dimension]
        );
        
        // Create two different style tensors for comparison with extremely large values
        // Using extremely large values should force differences in the output
        let mut style_data1 = vec![0.0; batch_size * style_dim];
        let mut style_data2 = vec![0.0; batch_size * style_dim];
        
        for i in 0..style_dim {
            style_data1[i] = 10.0; // Large positive value
            style_data2[i] = -10.0; // Large negative value
        }
        
        let style1 = Tensor::from_data(
            style_data1,
            vec![batch_size, style_dim]
        );
        
        let style2 = Tensor::from_data(
            style_data2,
            vec![batch_size, style_dim]
        );
        
        // Generate outputs with both styles
        println!("Processing with style1 (all large positive)...");
        let output1 = adain.forward(&x, &style1);
        
        println!("Processing with style2 (all large negative)...");
        let output2 = adain.forward(&x, &style2);
        
        // Print the first few values from both outputs
        println!("Output1 first few values: {:?}", &output1.data()[0..10]);
        println!("Output2 first few values: {:?}", &output2.data()[0..10]);
        
        // Compare the outputs between the two styles more extensively
        // The two outputs should be different for at least some values
        let mut styles_differ = false;
        let mut max_diff = 0.0;
        let mut diff_position = 0;
        
        for i in 0..output1.data().len() {
            let diff = (output1.data()[i] - output2.data()[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                diff_position = i;
            }
            if diff > 0.1 { // Increased threshold for detecting differences
                styles_differ = true;
                println!("Found difference at position {}: {} vs {}, diff={}", 
                         i, output1.data()[i], output2.data()[i], diff);
                break;
            }
        }
        
        println!("Maximum difference between style1 and style2 outputs: {} at position {}", 
                 max_diff, diff_position);
        
        // If the adain implementation works correctly, changing the style should affect the output
        assert!(styles_differ, 
                "Different style tensors should produce different outputs (max diff: {})", max_diff);
        
        println!("Test completed successfully!");
        
        Ok(())
    }
}