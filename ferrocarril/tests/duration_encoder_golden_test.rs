//! DurationEncoder golden-reference test for Phase 3.
//!
//! Runs the Rust `DurationEncoder::forward` on the exact same input that
//! the Python `KModel.forward_with_tokens` passes to
//! `predictor.text_encoder`, then compares the output against the
//! `predictor_text_encoder.npy` fixture produced by
//! `scripts/validate_kmodel.py`.
//!
//! Input construction mirrors Python line-for-line:
//!   bert_dur = bert(input_ids, attention_mask=~text_mask)
//!   d_en     = bert_encoder(bert_dur).transpose(-1, -2)   # [1, 512, 7]
//!   s        = ref_s[:, 128:]                              # [1, 128]
//!   d        = predictor.text_encoder(d_en, s, input_lengths, text_mask)
//!
//! Reference values were extracted from the Python fixture:
//!   shape (1, 7, 640)
//!   first time, first 8 dims: [-0.32289, 0.19647, -0.03532, -0.02273,
//!                              0.37823, -0.34230, 1.77888, -0.21553]
//!   last  time, first 8 dims: [-0.27839, 0.25613, -0.45376, -0.05143,
//!                              0.46138, -0.25172, 2.27205, -1.16052]
//!   mean |x|: 0.440264
//!
//! The test is `#[ignore]`d until the Rust DurationEncoder matches Python
//! within tolerance. Run explicitly with
//! `cargo test --release --test duration_encoder_golden_test -- --ignored --nocapture`.

#![cfg(feature = "weights")]

use std::error::Error;
use std::path::Path;

use ferrocarril_core::weights_binary::BinaryWeightLoader;
use ferrocarril_core::{tensor::Tensor, LoadWeightsBinary};
use ferrocarril_nn::bert::{BertConfig, CustomBert};
use ferrocarril_nn::linear::Linear;
use ferrocarril_nn::prosody::DurationEncoder;
use ferrocarril_nn::Forward;

/// Python reference values for `predictor.text_encoder` output with input
/// `[0, 50, 86, 54, 59, 135, 0]` and voice `af_heart` row 4.
const PY_FIRST_TIME_FIRST_8: [f32; 8] = [
    -0.32289004, 0.19647229, -0.03532180, -0.02273023, 0.37822828, -0.34229538,
    1.77887976, -0.21553150,
];

const PY_LAST_TIME_FIRST_8: [f32; 8] = [
    -0.27838776, 0.25612694, -0.45376116, -0.05143222, 0.46138108, -0.25171515,
    2.27204823, -1.16051972,
];

const PY_MEAN_ABS: f32 = 0.440264;

const DRIFT_TOLERANCE: f32 = 1e-3;

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

/// Load the raw af_heart voice pack as a flat `[510, 256]` tensor.
fn load_voice_pack(weights_path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let voice_file = format!("{}/voices/af_heart.bin", weights_path);
    let bytes = std::fs::read(&voice_file)?;
    let num_f32 = bytes.len() / 4;
    // Voice pack shape: [510, 1, 256] = 130560 floats.
    assert_eq!(
        num_f32,
        510 * 256,
        "af_heart voice file has {} f32 values, expected {}",
        num_f32,
        510 * 256
    );
    let mut data = Vec::with_capacity(num_f32);
    for i in 0..num_f32 {
        let b = &bytes[i * 4..i * 4 + 4];
        data.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(Tensor::from_data(data, vec![510, 256]))
}

#[test]
#[ignore = "Phase 3 numerical validation: enable with --ignored until DurationEncoder matches Python"]
fn test_duration_encoder_golden_vs_python_reference() -> Result<(), Box<dyn Error>> {
    let weights_path = find_weights_path().ok_or_else(|| {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
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

    // Input matches the BERT/TextEncoder/KModel harnesses.
    let input_ids =
        Tensor::<i64>::from_data(vec![0, 50, 86, 54, 59, 135, 0], vec![1, 7]);
    let attention_mask =
        Tensor::<i64>::from_data(vec![1, 1, 1, 1, 1, 1, 1], vec![1, 7]);

    let bert_out = bert.forward(&input_ids, None, Some(&attention_mask));
    assert_eq!(bert_out.shape(), &[1, 7, 768], "BERT output shape mismatch");

    // ---- bert_encoder: Linear(768 → 512) ---------------------------------
    let mut bert_encoder = Linear::new(768, 512, true);
    bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;
    let bert_enc_out = bert_encoder.forward(&bert_out);
    assert_eq!(
        bert_enc_out.shape(),
        &[1, 7, 512],
        "bert_encoder output shape mismatch"
    );

    // ---- transpose BTC → BCT to build d_en --------------------------------
    let (b, t, c) = (1, 7, 512);
    let mut d_en_data = vec![0.0f32; b * c * t];
    for bb in 0..b {
        for tt in 0..t {
            for cc in 0..c {
                d_en_data[bb * c * t + cc * t + tt] = bert_enc_out[&[bb, tt, cc]];
            }
        }
    }
    let d_en = Tensor::from_data(d_en_data, vec![b, c, t]);
    assert_eq!(d_en.shape(), &[1, 512, 7]);

    // ---- style from voice pack -------------------------------------------
    // Python: ref_s = pack[len(ps) - 1]; s = ref_s[:, 128:]
    // num_phonemes = input_ids.shape[-1] - 2 (BOS/EOS) = 5
    // voice_row = 5 - 1 = 4
    let voice_pack = load_voice_pack(&weights_path)?;
    let row_idx = 4usize;
    let mut style_data = vec![0.0f32; 128];
    for i in 0..128 {
        // ref_s is pack[row_idx], shape [256]; style = [128:256]
        style_data[i] = voice_pack[&[row_idx, 128 + i]];
    }
    let style = Tensor::from_data(style_data, vec![1, 128]);

    // ---- text mask: all zeros for a 7-token, length-7 input --------------
    let text_mask = Tensor::from_data(vec![false; 7], vec![1, 7]);

    // ---- DurationEncoder -------------------------------------------------
    // DurationEncoder(style_dim=128, d_model=512, n_layers=3, dropout=0.0).
    let mut duration_encoder = DurationEncoder::new(128, 512, 3, 0.0);
    duration_encoder
        .load_weights_binary(&loader, "predictor", "module.text_encoder")?;

    let output = duration_encoder.forward(&d_en, &style, &text_mask);
    assert_eq!(
        output.shape(),
        &[1, 7, 640],
        "DurationEncoder output shape mismatch (expected [1, 7, 640])"
    );

    // ---- Extract values for comparison ------------------------------------
    let mut rust_first = [0.0f32; 8];
    let mut rust_last = [0.0f32; 8];
    for d in 0..8 {
        rust_first[d] = output[&[0, 0, d]];
        rust_last[d] = output[&[0, 6, d]];
    }
    let rust_mean_abs: f32 =
        output.data().iter().map(|x| x.abs()).sum::<f32>() / output.data().len() as f32;

    println!("\n=== DurationEncoder golden comparison ===");
    println!("Python reference:");
    println!("  first time first 8 dims: {:?}", PY_FIRST_TIME_FIRST_8);
    println!("  last  time first 8 dims: {:?}", PY_LAST_TIME_FIRST_8);
    println!("  mean |x|               : {:.6}", PY_MEAN_ABS);
    println!("Rust DurationEncoder output:");
    println!("  first time first 8 dims: {:?}", rust_first);
    println!("  last  time first 8 dims: {:?}", rust_last);
    println!("  mean |x|               : {:.6}", rust_mean_abs);

    let first_err: f32 = PY_FIRST_TIME_FIRST_8
        .iter()
        .zip(rust_first.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let last_err: f32 = PY_LAST_TIME_FIRST_8
        .iter()
        .zip(rust_last.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0, f32::max);
    let mean_err = (rust_mean_abs - PY_MEAN_ABS).abs();

    println!("\nDiff vs Python reference (tolerance {}):", DRIFT_TOLERANCE);
    println!("  first time max |Δ|: {:.6}", first_err);
    println!("  last  time max |Δ|: {:.6}", last_err);
    println!("  mean|x| Δ         : {:.6}", mean_err);

    assert!(
        first_err < DRIFT_TOLERANCE,
        "DurationEncoder first-time drift {:.6} exceeds tolerance {:.6}",
        first_err,
        DRIFT_TOLERANCE
    );
    assert!(
        last_err < DRIFT_TOLERANCE,
        "DurationEncoder last-time drift {:.6} exceeds tolerance {:.6}",
        last_err,
        DRIFT_TOLERANCE
    );
    assert!(
        mean_err < DRIFT_TOLERANCE,
        "DurationEncoder mean|x| drift {:.6} exceeds tolerance {:.6}",
        mean_err,
        DRIFT_TOLERANCE
    );

    Ok(())
}