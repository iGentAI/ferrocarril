//! Structural smoke test for the Generator-side `AdaINResBlock1`.
//!
//! This block is a port of `kokoro/istftnet.py::AdaINResBlock1`. It has
//! no upsample and no learned shortcut — for upsample behaviour see the
//! Decoder's `AdainResBlk1d` instead. The test just verifies the
//! forward pass produces a 3-D output with the same shape as the input
//! and that no NaN/Inf sneaks through the Snake1D activations.

#[cfg(test)]
mod tests {
    use ferrocarril_core::tensor::Tensor;
    use ferrocarril_nn::vocoder::AdaINResBlock1;

    #[test]
    fn test_adain_resblk1_forward_shape() {
        let block = AdaINResBlock1::new(
            64,             // channels
            3,              // kernel_size
            vec![1, 3, 5],  // dilation
            128,            // style_dim
        );

        let batch_size = 1;
        let channels = 64;
        let time_dim = 16;

        // Input features [B, C, T]
        let x = Tensor::from_data(
            vec![0.1f32; batch_size * channels * time_dim],
            vec![batch_size, channels, time_dim],
        );

        // Style vector [B, style_dim]
        let s = Tensor::from_data(
            vec![0.1f32; batch_size * 128],
            vec![batch_size, 128],
        );

        let y = block.forward(&x, &s);

        // Output shape must match input shape (no upsample).
        assert_eq!(
            y.shape(),
            x.shape(),
            "AdaINResBlock1 output shape should match input shape",
        );

        // All values must be finite.
        for &value in y.data() {
            assert!(
                value.is_finite(),
                "AdaINResBlock1 output contains NaN or infinite values"
            );
        }

        println!(
            "AdaINResBlock1 forward test passed with output shape: {:?}",
            y.shape()
        );
    }
}