//! Simple test for Ferrocarril_nn LSTM layer

#[cfg(test)]
mod tests {
    use ferrocarril_core::{tensor::Tensor};
    use ferrocarril_nn::{Forward, lstm::LSTM};

    #[test]
    fn test_lstm_bidirectional_with_style() {
        // This test explicitly verifies that we can properly process 
        // concatenated inputs (hidden_dim + style_dim) through the LSTM
        
        // Create LSTM with proper dimensions
        let input_size = 640; // hidden_dim 512 + style_dim 128
        let hidden_size = 512 / 2; // half of hidden_dim because it's bidirectional
        let lstm = LSTM::new(
            input_size,
            hidden_size, // hidden_size will be doubled for output because bidirectional=true
            1, // num_layers
            true, // batch_first
            true // bidirectional
        );
        
        // Create input with proper dimension [batch=1, seq_len=29, input_size=640]
        let batch_size = 1;
        let seq_len = 29;
        let mut input_data = vec![0.1; batch_size * seq_len * input_size];
        let input = Tensor::from_data(input_data, vec![batch_size, seq_len, input_size]);
        
        // Run forward pass
        let (output, _) = lstm.forward_batch_first(&input, None, None);
        
        // Check output dimensions 
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size * 2]);
        
        println!("LSTM bidirectional test passed");
        println!("Input shape: {:?}, output shape: {:?}", input.shape(), output.shape());
    }
    
    #[test]
    fn test_custom_prosody_predictor() {
        // Here we would test a minimal version of the ProsodyPredictor
        // that just tests the specific interactions with style dimensions
        // This is left as an exercise for future work
    }
}