//! BERT golden-reference test for Phase 3 numerical validation.
//!
//! Runs the Rust `CustomBert::forward` on the exact same input that the
//! Python harness `scripts/validate_bert.py` uses, then compares the
//! output against reference values extracted from
//! `transformers.AlbertModel` loaded with the real Kokoro-82M weights.
//!
//! **Status: VALIDATED.** As of commit 33, the Rust `CustomBert` output
//! matches the Python reference to within ~2e-6 on the test input. This
//! test is no longer `#[ignore]`d and runs as part of the normal
//! `cargo test` suite to catch regressions.
//!
//! When this test fails, the printed diff tells us at which dim the
//! Rust output diverges from the Python reference, which is where a
//! newly-introduced numerical bug lives.

#![cfg(feature = "weights")]

use std::error::Error;
use std::path::Path;

use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_core::{tensor::Tensor, LoadWeightsBinary};
use ferrocarril_nn::bert::{BertConfig, CustomBert};

/// Python reference values from `transformers.AlbertModel` with Kokoro
/// weights, run on `[0, 50, 86, 54, 59, 135, 0]`. See
/// `scripts/validate_bert.py`. 7-token input, hidden_size 768. First and
/// last token, first 8 dimensions of the last hidden state.
const PY_FIRST_TOKEN_FIRST_8: [f32; 8] = [
    0.00324002, 0.09995311, 0.11737258, 0.18591806, 0.19735263, 0.10913242,
    -0.06349043, 0.07820973,
];

const PY_LAST_TOKEN_FIRST_8: [f32; 8] = [
    0.00618068, 0.00297593, -0.04712997, 0.05773854, 0.13661838, 0.35195938,
    -0.17414665, 0.07154756,
];

const PY_MEAN_ABS: f32 = 0.260139;

/// Drift tolerance the Rust `CustomBert` must hit for this test to pass.
/// The actual observed drift on this test input is ~2e-6, so 1e-4 is
/// very generous headroom while still catching any real regression.
const DRIFT_TOLERANCE: f32 = 1e-4;

/// Find the converted weight directory. Matches the decoder real-weights
/// test's search path convention.
fn find_weights_path() -> Option<String> {
    for candidate in [
        "../ferrocarril_weights",
        "ferrocarril_weights",
        "../../ferrocarril_weights",
    ] {
        if Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

#[test]
fn test_bert_golden_vs_python_reference() -> Result<(), Box<dyn Error>> {
    let weights_path = find_weights_path().ok_or_else(|| {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
    })?;

    println!("Loading converted weights from {}", weights_path);
    let loader = BinaryWeightLoader::from_directory(&weights_path)?;

    // Build a BertConfig matching Kokoro's plbert sub-config.
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

    // Same input as scripts/validate_bert.py.
    let input_ids =
        Tensor::<i64>::from_data(vec![0, 50, 86, 54, 59, 135, 0], vec![1, 7]);
    let attention_mask =
        Tensor::<i64>::from_data(vec![1, 1, 1, 1, 1, 1, 1], vec![1, 7]);

    let output = bert.forward(&input_ids, None, Some(&attention_mask));

    // Expected shape [1, 7, 768].
    assert_eq!(
        output.shape(),
        &[1, 7, 768],
        "CustomBert output shape mismatch"
    );

    // Extract first-token and last-token first-8 dims.
    let mut rust_first = [0.0f32; 8];
    let mut rust_last = [0.0f32; 8];
    for d in 0..8 {
        rust_first[d] = output[&[0, 0, d]];
        rust_last[d] = output[&[0, 6, d]];
    }

    // Global statistics for a cross-check.
    let rust_mean_abs: f32 =
        output.data().iter().map(|x| x.abs()).sum::<f32>() / output.data().len() as f32;

    println!("\n=== BERT golden comparison (input: [0, 50, 86, 54, 59, 135, 0]) ===");
    println!("Python reference (transformers.AlbertModel with Kokoro weights):");
    println!("  first token first 8 dims: {:?}", PY_FIRST_TOKEN_FIRST_8);
    println!("  last  token first 8 dims: {:?}", PY_LAST_TOKEN_FIRST_8);
    println!("  mean |x|                : {:.6}", PY_MEAN_ABS);
    println!("Rust CustomAlbert output:");
    println!("  first token first 8 dims: {:?}", rust_first);
    println!("  last  token first 8 dims: {:?}", rust_last);
    println!("  mean |x|                : {:.6}", rust_mean_abs);

    let first_max_abs_err: f32 = PY_FIRST_TOKEN_FIRST_8
        .iter()
        .zip(rust_first.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let last_max_abs_err: f32 = PY_LAST_TOKEN_FIRST_8
        .iter()
        .zip(rust_last.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let mean_abs_err = (rust_mean_abs - PY_MEAN_ABS).abs();

    println!("\nDiff vs Python reference (tolerance {}):", DRIFT_TOLERANCE);
    println!("  first token max |Δ|: {:.6}", first_max_abs_err);
    println!("  last  token max |Δ|: {:.6}", last_max_abs_err);
    println!("  mean|x| Δ          : {:.6}", mean_abs_err);

    // Hard assertions on the tolerance. The test is `#[ignore]`d so normal
    // `cargo test` does not run it until Phase 3 numerical fixes land; an
    // explicit `cargo test -- --ignored` run will fail loudly and the
    // printed diff above will point at the drifting quantity.
    assert!(
        first_max_abs_err < DRIFT_TOLERANCE,
        "CustomAlbert first-token drift {:.6} exceeds tolerance {:.6}",
        first_max_abs_err,
        DRIFT_TOLERANCE
    );
    assert!(
        last_max_abs_err < DRIFT_TOLERANCE,
        "CustomAlbert last-token drift {:.6} exceeds tolerance {:.6}",
        last_max_abs_err,
        DRIFT_TOLERANCE
    );
    assert!(
        mean_abs_err < DRIFT_TOLERANCE,
        "CustomAlbert mean|x| drift {:.6} exceeds tolerance {:.6}",
        mean_abs_err,
        DRIFT_TOLERANCE
    );

    Ok(())
}