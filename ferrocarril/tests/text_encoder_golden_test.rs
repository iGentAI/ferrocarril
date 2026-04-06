//! TextEncoder golden-reference test for Phase 3 numerical validation.
//!
//! Runs Rust `TextEncoder::forward` on the same fixed input that the
//! Python harness `scripts/validate_text_encoder.py` uses, then compares
//! the output against reference values extracted from a faithful
//! PyTorch reimplementation of Kokoro's TextEncoder loaded with the real
//! Kokoro-82M `text_encoder` sub-state-dict.
//!
//! The test is `#[ignore]`d by default because the Phase 3 numerical
//! correctness pass is in flight and the Rust TextEncoder is not yet
//! validated. Run explicitly:
//!
//!     cargo test --release --test text_encoder_golden_test -- --ignored --nocapture
//!
//! Expected Python reference output (from Kokoro v1.0 weights, 7-token
//! input `[0, 50, 86, 54, 59, 135, 0]`, BCT layout `[1, 512, 7]`):
//!
//!   First channel, first 8 time steps (actually 7 steps here):
//!     [0.28264, -0.01137, -0.01913, -0.25654, -0.76695, -0.05909, -0.10944]
//!   Last channel, first 8 time steps:
//!     [0.04528, 0.01993, 0.15778, 0.00147, -0.40214, -0.69545, -0.62567]
//!   Mean |x|: 0.219714

#![cfg(feature = "weights")]

use std::error::Error;
use std::path::Path;

use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_core::{tensor::Tensor, LoadWeightsBinary};
use ferrocarril_nn::text_encoder::TextEncoder;

/// Python reference values from the faithful PyTorch reimplementation of
/// Kokoro's TextEncoder in `scripts/validate_text_encoder.py`.
const PY_FIRST_CHANNEL: [f32; 7] = [
    0.28263554, -0.01137452, -0.01912925, -0.25653628, -0.76695275, -0.05908985,
    -0.10944211,
];

const PY_LAST_CHANNEL: [f32; 7] = [
    0.04528069, 0.01992723, 0.15777664, 0.00147066, -0.4021421, -0.69544894,
    -0.6256699,
];

const PY_MEAN_ABS: f32 = 0.219714;

const DRIFT_TOLERANCE: f32 = 1e-4;

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
#[ignore = "Phase 3 numerical validation: enable with --ignored until Rust TextEncoder matches Python"]
fn test_text_encoder_golden_vs_python_reference() -> Result<(), Box<dyn Error>> {
    let weights_path = find_weights_path().ok_or_else(|| {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
    })?;

    println!("Loading converted weights from {}", weights_path);
    let loader = BinaryWeightLoader::from_directory(&weights_path)?;

    // Kokoro text encoder config: channels=512, kernel_size=5, depth=3,
    // n_symbols=178.
    let mut text_encoder = TextEncoder::new(512, 5, 3, 178);
    text_encoder.load_weights_binary(&loader)?;

    // Same input as scripts/validate_text_encoder.py.
    let input_ids =
        Tensor::<i64>::from_data(vec![0, 50, 86, 54, 59, 135, 0], vec![1, 7]);
    let input_lengths = vec![7usize];
    let text_mask = Tensor::from_data(vec![false; 7], vec![1, 7]);

    let output = text_encoder.forward(&input_ids, &input_lengths, &text_mask);

    // Rust TextEncoder returns BCT [1, 512, 7].
    assert_eq!(
        output.shape(),
        &[1, 512, 7],
        "TextEncoder output shape mismatch"
    );

    // Extract first-channel and last-channel time-step values.
    let mut rust_first_channel = [0.0f32; 7];
    let mut rust_last_channel = [0.0f32; 7];
    for t in 0..7 {
        rust_first_channel[t] = output[&[0, 0, t]];
        rust_last_channel[t] = output[&[0, 511, t]];
    }

    let rust_mean_abs: f32 =
        output.data().iter().map(|x| x.abs()).sum::<f32>() / output.data().len() as f32;

    println!(
        "\n=== TextEncoder golden comparison (input: [0, 50, 86, 54, 59, 135, 0]) ==="
    );
    println!("Python reference:");
    println!("  first channel: {:?}", PY_FIRST_CHANNEL);
    println!("  last  channel: {:?}", PY_LAST_CHANNEL);
    println!("  mean |x|     : {:.6}", PY_MEAN_ABS);
    println!("Rust TextEncoder output:");
    println!("  first channel: {:?}", rust_first_channel);
    println!("  last  channel: {:?}", rust_last_channel);
    println!("  mean |x|     : {:.6}", rust_mean_abs);

    let first_max_abs_err: f32 = PY_FIRST_CHANNEL
        .iter()
        .zip(rust_first_channel.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let last_max_abs_err: f32 = PY_LAST_CHANNEL
        .iter()
        .zip(rust_last_channel.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let mean_abs_err = (rust_mean_abs - PY_MEAN_ABS).abs();

    println!("\nDiff vs Python reference (tolerance {}):", DRIFT_TOLERANCE);
    println!("  first channel max |Δ|: {:.6}", first_max_abs_err);
    println!("  last  channel max |Δ|: {:.6}", last_max_abs_err);
    println!("  mean|x| Δ            : {:.6}", mean_abs_err);

    assert!(
        first_max_abs_err < DRIFT_TOLERANCE,
        "TextEncoder first-channel drift {:.6} exceeds tolerance {:.6}",
        first_max_abs_err,
        DRIFT_TOLERANCE
    );
    assert!(
        last_max_abs_err < DRIFT_TOLERANCE,
        "TextEncoder last-channel drift {:.6} exceeds tolerance {:.6}",
        last_max_abs_err,
        DRIFT_TOLERANCE
    );
    assert!(
        mean_abs_err < DRIFT_TOLERANCE,
        "TextEncoder mean|x| drift {:.6} exceeds tolerance {:.6}",
        mean_abs_err,
        DRIFT_TOLERANCE
    );

    Ok(())
}