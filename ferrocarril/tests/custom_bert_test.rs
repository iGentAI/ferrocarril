//! Structural unit tests for `CustomBert`.
//!
//! These tests do not load any real Kokoro weights — they exercise the
//! BERT module's public API (construct, forward, attention mask) with
//! small random configs. The full real-weights numerical validation
//! lives in `bert_golden_test.rs`, which compares against the Python
//! `bert.npy` fixture to ~2e-6 precision.

#[cfg(test)]
mod tests {
    use ferrocarril_core::tensor::Tensor;
    use ferrocarril_nn::bert::{BertConfig, CustomBert};

    /// Smoke test: build a small BERT and verify forward pass produces
    /// the expected `[batch, seq_len, hidden_size]` shape.
    #[test]
    fn test_custom_bert_forward() {
        let config = BertConfig {
            vocab_size: 100,
            embedding_size: 128,
            hidden_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            max_position_embeddings: 128,
            dropout_prob: 0.0,
        };

        let bert = CustomBert::new(config);

        let input_ids = Tensor::<i64>::from_data(vec![1, 2, 3, 4, 5], vec![1, 5]);
        let output = bert.forward(&input_ids, None, None);

        assert_eq!(
            output.shape(),
            &[1, 5, 128],
            "CustomBert forward output shape mismatch"
        );
    }

    /// Smoke test: build a small BERT and verify the 2-D HF-convention
    /// attention mask is accepted without runtime errors and produces
    /// the same shape as the unmasked forward pass.
    #[test]
    fn test_custom_bert_attention_mask() {
        let config = BertConfig {
            vocab_size: 100,
            embedding_size: 128,
            hidden_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            max_position_embeddings: 128,
            dropout_prob: 0.0,
        };

        let bert = CustomBert::new(config);
        let input_ids = Tensor::<i64>::from_data(vec![1, 2, 3, 4, 5], vec![1, 5]);

        // 2-D HF-convention mask: 1 = visible, 0 = masked. Hide the last
        // two tokens.
        let attention_mask = Tensor::from_data(vec![1i64, 1, 1, 0, 0], vec![1, 5]);

        let masked_output = bert.forward(&input_ids, None, Some(&attention_mask));
        let unmasked_output = bert.forward(&input_ids, None, None);

        assert_eq!(
            masked_output.shape(),
            unmasked_output.shape(),
            "Masked and unmasked CustomBert outputs have different shapes"
        );
        assert_eq!(masked_output.shape(), &[1, 5, 128]);
    }
}