//! Tests for AdainResBlk1d upsampling implementation

#[cfg(test)]
mod tests {
    use ferrocarril_core::tensor::Tensor;
    use ferrocarril_nn::vocoder::{AdaINResBlock1, UpsampleType};
    
    #[test]
    fn test_adain_resblk_upsampling() {
        // Create a block with upsampling enabled
        let block = AdaINResBlock1::with_upsample(
            64,                 // channels
            3,                  // kernel_size
            vec![1, 3, 5],      // dilation
            128,                // style_dim
            Some(UpsampleType::Nearest) // Enable upsampling
        );
        
        // Create input tensors with extreme values to ensure non-zero output
        let batch_size = 1;
        let channels = 64;
        let time_dim = 16;
        
        // Create alternating positive and negative values to better trigger non-zero outputs
        let mut x_data = Vec::with_capacity(batch_size * channels * time_dim);
        for i in 0..(batch_size * channels * time_dim) {
            x_data.push(if i % 2 == 0 { 100.0 } else { -100.0 });
        }
        
        // Input features (shape [B, C, T])
        let x = Tensor::from_data(x_data, vec![batch_size, channels, time_dim]);
        
        // Style vector (shape [B, style_dim]) with alternating values
        let mut s_data = Vec::with_capacity(batch_size * 128);
        for i in 0..(batch_size * 128) {
            s_data.push(if i % 2 == 0 { 50.0 } else { -50.0 });
        }
        let s = Tensor::from_data(s_data, vec![batch_size, 128]);
        
        // Forward pass
        let y = block.forward(&x, &s);
        
        // The output time dimension should be doubled due to upsampling
        assert_eq!(y.shape().len(), 3, "Output should be 3D");
        assert_eq!(y.shape()[0], batch_size, "Batch size should be preserved");
        assert_eq!(y.shape()[1], channels, "Channel size should be preserved");
        assert_eq!(y.shape()[2], time_dim * 2, "Time dimension should be doubled from upsampling");
        
        // Print first few values for debugging
        println!("First output values: {:?}", &y.data()[0..5.min(y.data().len())]);
        
        // Check for NaN or infinity values
        for &value in y.data() {
            assert!(!value.is_nan() && !value.is_infinite(), "Output contains NaN or infinite values");
        }
        
        println!("AdaINResBlock1 upsampling test passed with output shape: {:?}", y.shape());
    }
    
    /// Note: This test doesn't actually change channel dimensions because the current AdaINResBlock1 
    /// implementation doesn't support changing the channel dimensions in the constructor.
    /// However, it still tests the learned shortcut path because upsampling is enabled,
    /// which triggers the learned shortcut functionality.
    #[test]
    fn test_adain_resblk_learned_shortcut() {
        // Create a block with upsampling enabled which requires a learned shortcut
        let channels = 64;
        
        let block = AdaINResBlock1::with_upsample(
            channels,           // channels
            3,                  // kernel_size
            vec![1, 3, 5],      // dilation
            128,                // style_dim
            Some(UpsampleType::Nearest) // Enable upsampling, triggering learned shortcut
        );
        
        // Create input tensors with extreme values
        let batch_size = 1;
        let time_dim = 16;
        
        // Create alternating positive and negative values
        let mut x_data = Vec::with_capacity(batch_size * channels * time_dim);
        for i in 0..(batch_size * channels * time_dim) {
            x_data.push(if i % 2 == 0 { 100.0 } else { -100.0 });
        }
        
        // Input features (shape [B, C, T])
        let x = Tensor::from_data(x_data, vec![batch_size, channels, time_dim]);
        
        // Style vector (shape [B, style_dim])
        let mut s_data = Vec::with_capacity(batch_size * 128);
        for i in 0..(batch_size * 128) {
            s_data.push(if i % 2 == 0 { 50.0 } else { -50.0 });
        }
        let s = Tensor::from_data(s_data, vec![batch_size, 128]);
        
        // Forward pass
        let y = block.forward(&x, &s);
        
        // The output should have doubled time dimension
        assert_eq!(y.shape().len(), 3, "Output should be 3D");
        assert_eq!(y.shape()[0], batch_size, "Batch size should be preserved");
        assert_eq!(y.shape()[1], channels, "Channel size should be preserved");
        assert_eq!(y.shape()[2], time_dim * 2, "Time dimension should be doubled from upsampling");
        
        println!("AdaINResBlock1 with learned shortcut test passed with output shape: {:?}", y.shape());
        
        // If we were to enhance AdaINResBlock1 to support changing channel dimensions, 
        // the test could be updated as follows:
        /*
        let in_channels = 64;
        let out_channels = 32;
        
        let block = AdaINResBlock1::with_upsample_and_channel_change(
            in_channels,        // in_channels
            out_channels,       // out_channels
            3,                  // kernel_size
            vec![1, 3, 5],      // dilation
            128,                // style_dim
            Some(UpsampleType::Nearest) // Enable upsampling
        );
        
        // Then verify output shape has out_channels:
        assert_eq!(y.shape()[1], out_channels, "Output channels should match out_channels");
        */
    }
    
    #[test]
    fn test_adain_resblk_no_upsampling() {
        // Create a block without upsampling for comparison
        let block = AdaINResBlock1::new(
            64,                 // channels
            3,                  // kernel_size
            vec![1, 3, 5],      // dilation
            128                 // style_dim
        );
        
        // Create input tensors
        let batch_size = 1;
        let channels = 64;
        let time_dim = 16;
        
        // Input features (shape [B, C, T])
        let x = Tensor::from_data(vec![0.1; batch_size * channels * time_dim], 
                               vec![batch_size, channels, time_dim]);
        
        // Style vector (shape [B, style_dim])
        let s = Tensor::from_data(vec![0.1; batch_size * 128], 
                               vec![batch_size, 128]);
        
        // Forward pass
        let y = block.forward(&x, &s);
        
        // The output should have the same shape as the input (no upsampling)
        assert_eq!(y.shape(), x.shape(), "Output shape should match input shape without upsampling");
        
        println!("AdaINResBlock1 no upsampling test passed with output shape: {:?}", y.shape());
    }
}