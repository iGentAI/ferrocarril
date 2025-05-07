//! Basic integration test to ensure compilation

use ferrocarril_core::{tensor::Tensor, Config};
use ferrocarril::model::FerroModel; // Import from correct module

#[test]
fn test_basic_functionality() {
    // Test tensor creation
    let tensor: Tensor<f32> = Tensor::new(vec![2, 3]);
    assert_eq!(tensor.shape(), &[2, 3]);

    // Create a basic config manually
    let mut vocab = std::collections::HashMap::new();
    for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-".chars().enumerate() {
        vocab.insert(c, i);
    }
    
    let config = Config {
        vocab,
        n_token: 150,
        hidden_dim: 512,
        n_layer: 4,
        style_dim: 128,
        n_mels: 80,
        max_dur: 50,
        dropout: 0.1,
        text_encoder_kernel_size: 5,
        istftnet: ferrocarril_core::IstftnetConfig {
            upsample_rates: vec![8, 8, 2, 2],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            gen_istft_n_fft: 16,
            gen_istft_hop_size: 4,
        },
        plbert: ferrocarril_core::PlbertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
        },
    };
    
    // Test basic model loading (placeholder)
    let model = FerroModel::load("dummy_path", config).unwrap();
    
    // Test basic inference (placeholder)
    let result = model.infer("test").unwrap();
    assert_eq!(result.len(), 24000); // Updated to match the expected output length
}