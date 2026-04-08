//! BertEncoder golden-reference test for Phase 3 numerical validation.
//!
//! The `bert_encoder` component in Kokoro is a single `nn.Linear(768, 512)`
//! that projects the BERT output into the prosody predictor's input space.
//! It is trivial but still deserves a regression gate, both as a check on
//! the binary weight loader and as a safety net before the downstream
//! prosody / decoder golden tests.
//!
//! Runs Rust CustomBert → Linear on the canonical 7-token input
//! `[0, 50, 86, 54, 59, 135, 0]` (same input as every other Phase 3 golden
//! test), then diffs the output against
//! `tests/fixtures/kmodel/bert_encoder.npy` dumped by
//! `scripts/validate_kmodel.py`.

#![cfg(feature = "weights")]

mod common;

use std::error::Error;

use ferrocarril_core::{tensor::Tensor, weights_binary::BinaryWeightLoader, LoadWeightsBinary};
use ferrocarril_nn::bert::{BertConfig, CustomBert};
use ferrocarril_nn::linear::Linear;
use ferrocarril_nn::Forward;

const DRIFT_TOLERANCE: f32 = 1e-4;

#[test]
fn test_bert_encoder_golden_vs_python_reference() -> Result<(), Box<dyn Error>> {
    let weights_path = common::find_weights_path().ok_or_else(|| -> Box<dyn Error> {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
            .into()
    })?;
    let fixtures_path = common::find_kmodel_fixtures_path().ok_or_else(|| -> Box<dyn Error> {
        "tests/fixtures/kmodel not found; run `python3 scripts/validate_kmodel.py` \
         to regenerate the golden fixtures"
            .into()
    })?;

    println!("Loading converted weights from {}", weights_path);
    let loader = BinaryWeightLoader::from_directory(&weights_path)?;

    // ---- BERT ------------------------------------------------------------
    let bert_config = BertConfig {
        vocab_size: 178,
        embedding_size: 128,
        hidden_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 12,
        intermediate_size: 2048,
        max_position_embeddings: 512,
        dropout_prob: 0.0,
    };
    let mut bert = CustomBert::new(bert_config);
    bert.load_weights_binary(&loader, "bert", "module")?;

    // Same canonical input as every other Phase 3 golden test.
    let input_ids = Tensor::<i64>::from_data(vec![0, 50, 86, 54, 59, 135, 0], vec![1, 7]);
    let attention_mask = Tensor::<i64>::from_data(vec![1, 1, 1, 1, 1, 1, 1], vec![1, 7]);

    let bert_out = bert.forward(&input_ids, None, Some(&attention_mask));
    assert_eq!(bert_out.shape(), &[1, 7, 768], "BERT output shape mismatch");

    // ---- bert_encoder: Linear(768 → 512) ---------------------------------
    let mut bert_encoder = Linear::new(768, 512, true);
    bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;

    let encoded = bert_encoder.forward(&bert_out);
    assert_eq!(
        encoded.shape(),
        &[1, 7, 512],
        "bert_encoder output shape mismatch"
    );

    // ---- Load the Python fixture ----------------------------------------
    let fixture = common::npy::load(format!("{}/bert_encoder.npy", fixtures_path));
    assert_eq!(
        fixture.shape,
        vec![1, 7, 512],
        "bert_encoder.npy fixture shape mismatch"
    );
    let reference = fixture.as_f32();

    // ---- Diff ------------------------------------------------------------
    let rust_data = encoded.data();
    assert_eq!(
        rust_data.len(),
        reference.len(),
        "element count mismatch: Rust={}, Python={}",
        rust_data.len(),
        reference.len()
    );

    let max_diff = common::max_abs_diff(rust_data, reference);
    let rust_mean_abs = common::mean_abs(rust_data);
    let py_mean_abs = common::mean_abs(reference);

    println!("\n=== BertEncoder golden comparison ===");
    println!("Python reference mean|x|: {:.6}", py_mean_abs);
    println!("Rust output      mean|x|: {:.6}", rust_mean_abs);
    println!(
        "Max elementwise |Δ|    : {:.6e} (tolerance {:.0e})",
        max_diff, DRIFT_TOLERANCE
    );

    // Spot-check the first position across several channels for readability.
    print!("Rust first 6 channels @ pos 0: ");
    for c in 0..6 {
        print!("{:.6} ", encoded[&[0, 0, c]]);
    }
    println!();
    print!("Py   first 6 channels @ pos 0: ");
    for c in 0..6 {
        print!("{:.6} ", reference[c]);
    }
    println!();

    assert!(
        max_diff < DRIFT_TOLERANCE,
        "BertEncoder drift {:.6e} exceeds tolerance {:.0e}",
        max_diff,
        DRIFT_TOLERANCE
    );
    assert!(
        (rust_mean_abs - py_mean_abs).abs() < DRIFT_TOLERANCE,
        "BertEncoder mean|x| drift {:.6e} exceeds tolerance {:.0e}",
        (rust_mean_abs - py_mean_abs).abs(),
        DRIFT_TOLERANCE
    );

    Ok(())
}