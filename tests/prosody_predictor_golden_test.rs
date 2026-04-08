//! ProsodyPredictor golden-reference test for Phase 3 numerical validation.
//!
//! Runs the full Rust prosody inference path on the canonical 7-token
//! Kokoro input and diffs every stage against Python fixtures dumped by
//! `scripts/validate_kmodel.py`.
//!
//! Pipeline mirrors Python `KModel.forward_with_tokens`:
//!
//! ```text
//! input_ids -> bert -> bert_encoder.transpose -> d_en [1, 512, 7]
//!           -> predictor.text_encoder  -> d      [1, 7, 640]
//!           -> predictor.lstm          -> x      [1, 7, 512]
//!           -> predictor.duration_proj -> dur_logits [1, 7, 50]
//!              |
//!              v (sigmoid + sum(-1) + round + clamp)
//!           -> pred_dur [7]
//!              |
//!              v (build pred_aln_trg [7, 56])
//!           -> en = d.transpose(-1, -2) @ pred_aln_trg [1, 640, 56]
//!           -> F0Ntrain(en, s):
//!                shared(en.transpose)  -> shared_out [1, 56, 512]
//!                for block in F0:      -> F0 passes through 3 blocks
//!                F0_proj               -> F0 [1, 1, 112]
//!                (same for N)
//! ```
//!
//! Because the Rust `ProsodyPredictor::forward` takes an explicit
//! `alignment` argument, we build the Python pred_aln_trg directly from
//! the golden `pred_dur.npy` fixture and feed it in. The test then
//! separately verifies that Rust's own duration prediction round-trips to
//! the same `pred_dur`. Finally, to isolate where numerical drift enters
//! the F0/N pipeline, the test also walks the `F0Ntrain` path manually
//! through the (public) predictor sub-modules and compares every
//! intermediate to the corresponding fixture.

#![cfg(feature = "weights")]

mod common;

use std::error::Error;

use ferrocarril_core::{tensor::Tensor, weights_binary::BinaryWeightLoader, LoadWeightsBinary};
use ferrocarril_nn::bert::{BertConfig, CustomBert};
use ferrocarril_nn::linear::Linear;
use ferrocarril_nn::prosody::ProsodyPredictor;
use ferrocarril_nn::Forward;

const ABS_TOLERANCE: f32 = 1e-4;
const REL_TOLERANCE: f32 = 2e-5;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Layer is considered within tolerance if either the absolute drift is
/// below `ABS_TOLERANCE` OR the drift relative to the Python magnitude is
/// below `REL_TOLERANCE`. This handles both small-magnitude layers (where
/// absolute error is meaningful) and large-magnitude layers (where f32
/// accumulation noise scales with value magnitude).
fn layer_ok(max_diff: f32, py_mean_abs: f32) -> bool {
    let rel_cap = (REL_TOLERANCE * py_mean_abs).max(ABS_TOLERANCE);
    max_diff < rel_cap
}

/// Build the `pred_aln_trg [seq_len, total_frames]` one-hot alignment
/// matrix from a per-position duration vector. This is exactly what
/// `KModel.forward_with_tokens` does via `torch.repeat_interleave`.
fn build_alignment(pred_dur: &[i64], seq_len: usize) -> Tensor<f32> {
    assert_eq!(pred_dur.len(), seq_len, "pred_dur length mismatch");
    // Durations must be non-negative integers; guard the usize cast.
    for (i, &d) in pred_dur.iter().enumerate() {
        assert!(
            d >= 0,
            "pred_dur[{}] = {} is negative; expected non-negative frame counts",
            i,
            d
        );
    }
    let total_frames: usize = pred_dur.iter().map(|&d| d as usize).sum();
    let mut data = vec![0.0f32; seq_len * total_frames];
    let mut col = 0usize;
    for (pos, &dur) in pred_dur.iter().enumerate() {
        for _ in 0..dur as usize {
            data[pos * total_frames + col] = 1.0;
            col += 1;
        }
    }
    Tensor::from_data(data, vec![seq_len, total_frames])
}

/// Transpose [B, C, T] → [B, T, C].
fn transpose_bct_to_btc(x: &Tensor<f32>) -> Tensor<f32> {
    let (b, c, t) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let mut out = vec![0.0f32; b * t * c];
    for bb in 0..b {
        for cc in 0..c {
            for tt in 0..t {
                out[bb * t * c + tt * c + cc] = x[&[bb, cc, tt]];
            }
        }
    }
    Tensor::from_data(out, vec![b, t, c])
}

/// Transpose [B, T, C] → [B, C, T].
fn transpose_btc_to_bct(x: &Tensor<f32>) -> Tensor<f32> {
    let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let mut out = vec![0.0f32; b * c * t];
    for bb in 0..b {
        for tt in 0..t {
            for cc in 0..c {
                out[bb * c * t + cc * t + tt] = x[&[bb, tt, cc]];
            }
        }
    }
    Tensor::from_data(out, vec![b, c, t])
}

/// Load the raw af_heart voice pack as a `[510, 256]` flat tensor.
fn load_voice_pack(weights_path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let voice_file = format!("{}/voices/af_heart.bin", weights_path);
    let bytes = std::fs::read(&voice_file)?;
    let num_f32 = bytes.len() / 4;
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

/// Compare a Rust tensor against a fixture, printing shape/mean and
/// asserting on the max absolute difference. Returns the max diff so the
/// caller can report a cumulative summary.
fn check_layer(
    label: &str,
    rust: &Tensor<f32>,
    fixture: &common::npy::NpyArray,
) -> (f32, f32) {
    let rust_shape: Vec<usize> = rust.shape().to_vec();
    assert_eq!(
        rust_shape, fixture.shape,
        "{}: shape mismatch — Rust {:?} vs Python {:?}",
        label, rust_shape, fixture.shape
    );
    let py = fixture.as_f32();
    let rust_data = rust.data();
    assert_eq!(rust_data.len(), py.len(), "{}: element count mismatch", label);

    let max_diff = common::max_abs_diff(rust_data, py);
    let py_mean_abs = common::mean_abs(py);
    let rel_diff = if py_mean_abs > 0.0 {
        max_diff / py_mean_abs
    } else {
        0.0
    };
    println!(
        "{:<30} shape={:?} Rust mean|x|={:.6} Py mean|x|={:.6} max|Δ|={:.6e} rel={:.6e}",
        label,
        rust.shape(),
        common::mean_abs(rust_data),
        py_mean_abs,
        max_diff,
        rel_diff
    );
    (max_diff, py_mean_abs)
}

#[test]
fn test_prosody_predictor_golden_vs_python_reference() -> Result<(), Box<dyn Error>> {
    let weights_path = common::find_weights_path().ok_or_else(|| -> Box<dyn Error> {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
            .into()
    })?;
    let fixtures_path = common::find_kmodel_fixtures_path().ok_or_else(|| -> Box<dyn Error> {
        "tests/fixtures/kmodel not found; run `python3 scripts/validate_kmodel.py` first"
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

    let input_ids = Tensor::<i64>::from_data(vec![0, 50, 86, 54, 59, 135, 0], vec![1, 7]);
    let attention_mask = Tensor::<i64>::from_data(vec![1, 1, 1, 1, 1, 1, 1], vec![1, 7]);

    let bert_out = bert.forward(&input_ids, None, Some(&attention_mask));
    assert_eq!(bert_out.shape(), &[1, 7, 768]);

    // ---- bert_encoder: Linear(768 → 512) ---------------------------------
    let mut bert_encoder = Linear::new(768, 512, true);
    bert_encoder.load_weights_binary(&loader, "bert_encoder", "module")?;
    let bert_enc_out = bert_encoder.forward(&bert_out);
    assert_eq!(bert_enc_out.shape(), &[1, 7, 512]);

    // Build d_en = bert_enc.transpose(-1, -2): [1, 7, 512] -> [1, 512, 7]
    let d_en = transpose_btc_to_bct(&bert_enc_out);
    assert_eq!(d_en.shape(), &[1, 512, 7]);

    // ---- Style from voice pack -------------------------------------------
    let voice_pack = load_voice_pack(&weights_path)?;
    let row_idx = 4usize;
    let mut style_data = vec![0.0f32; 128];
    for i in 0..128 {
        style_data[i] = voice_pack[&[row_idx, 128 + i]];
    }
    let style = Tensor::from_data(style_data, vec![1, 128]);

    // ---- Load golden pred_dur fixture ------------------------------------
    let pred_dur_fixture = common::npy::load(format!("{}/pred_dur.npy", fixtures_path));
    assert_eq!(pred_dur_fixture.shape, vec![7]);
    let pred_dur_golden: Vec<i64> = pred_dur_fixture.as_i64().to_vec();
    let total_frames: usize = pred_dur_golden.iter().map(|&d| d as usize).sum();
    println!(
        "Golden pred_dur = {:?} (total frames = {})",
        pred_dur_golden, total_frames
    );
    assert_eq!(total_frames, 56, "expected 56 frames for this input");

    let alignment = build_alignment(&pred_dur_golden, 7);
    assert_eq!(alignment.shape(), &[7, 56]);

    // ---- ProsodyPredictor ------------------------------------------------
    let mut prosody = ProsodyPredictor::new(
        128, // style_dim
        512, // d_hid
        3,   // n_layers
        50,  // max_dur
        0.0, // dropout (inference)
    );
    prosody.load_weights_binary(&loader, "predictor", "module")?;

    let text_mask = Tensor::from_data(vec![false; 7], vec![1, 7]);

    let (dur_logits, en) = prosody.forward(&d_en, &style, &text_mask, &alignment)?;
    assert_eq!(dur_logits.shape(), &[1, 7, 50]);
    assert_eq!(en.shape(), &[1, 640, 56]);

    println!("\n======== Prosody layer-by-layer bisection ========");

    // ---- 1) dur_logits vs predictor_duration_proj.npy --------------------
    let dur_proj_fixture =
        common::npy::load(format!("{}/predictor_duration_proj.npy", fixtures_path));
    let (dur_max_diff, dur_py_mean) = check_layer("dur_logits", &dur_logits, &dur_proj_fixture);
    assert!(
        layer_ok(dur_max_diff, dur_py_mean),
        "dur_logits drift {:.6e} exceeds tolerance (Py mean|x|={:.6})",
        dur_max_diff,
        dur_py_mean
    );

    // ---- 2) Rust pred_dur vs golden pred_dur -----------------------------
    let mut rust_pred_dur = vec![0i64; 7];
    for t_idx in 0..7 {
        let mut sum = 0.0f32;
        for d in 0..50 {
            sum += sigmoid(dur_logits[&[0, t_idx, d]]);
        }
        let rounded = sum.round() as i64;
        rust_pred_dur[t_idx] = rounded.max(1);
    }
    println!("pred_dur Rust={:?} Python={:?}", rust_pred_dur, pred_dur_golden);
    assert_eq!(rust_pred_dur, pred_dur_golden, "pred_dur mismatch");

    // ---- 3) Bisect F0Ntrain through the public sub-modules ---------------
    // Python: x_btc = en.transpose(-1, -2)  -> [1, 56, 640]
    //         x_btc, _ = shared_lstm(x_btc) -> [1, 56, 512]
    //         F0 = x_btc.transpose(-1, -2)  -> [1, 512, 56]
    //         for block in F0: F0 = block(F0, s)
    //         F0 = F0_proj(F0)              -> [1, 1, 112]
    let en_btc = transpose_bct_to_btc(&en); // [1, 56, 640]
    assert_eq!(en_btc.shape(), &[1, 56, 640]);

    let (shared_out_btc, _) = prosody
        .shared_lstm
        .forward_batch_first(&en_btc, None, None);
    assert_eq!(shared_out_btc.shape(), &[1, 56, 512]);

    // Compare against predictor_shared fixture: Python hook captures
    // the LSTM output directly (BTC format).
    let shared_fixture =
        common::npy::load(format!("{}/predictor_shared.npy", fixtures_path));
    let (shared_diff, shared_py_mean) = check_layer("shared_lstm", &shared_out_btc, &shared_fixture);

    // F0 path ------------------------------------------------------------
    let mut f0 = transpose_btc_to_bct(&shared_out_btc); // [1, 512, 56]
    let f0_0_fixture = common::npy::load(format!("{}/predictor_F0_0.npy", fixtures_path));
    let f0_1_fixture = common::npy::load(format!("{}/predictor_F0_1.npy", fixtures_path));
    let f0_2_fixture = common::npy::load(format!("{}/predictor_F0_2.npy", fixtures_path));
    let f0_proj_fixture = common::npy::load(format!("{}/predictor_F0_proj.npy", fixtures_path));

    f0 = prosody.f0_blocks[0].forward(&f0, &style);
    let (f0_0_diff, f0_0_py_mean) = check_layer("F0.0", &f0, &f0_0_fixture);

    f0 = prosody.f0_blocks[1].forward(&f0, &style);
    let (f0_1_diff, f0_1_py_mean) = check_layer("F0.1 (upsample)", &f0, &f0_1_fixture);

    f0 = prosody.f0_blocks[2].forward(&f0, &style);
    let (f0_2_diff, f0_2_py_mean) = check_layer("F0.2", &f0, &f0_2_fixture);

    let f0_proj_out = prosody.f0_proj.forward(&f0);
    let (f0_proj_diff, f0_proj_py_mean) = check_layer("F0_proj", &f0_proj_out, &f0_proj_fixture);

    // N path -------------------------------------------------------------
    let mut n = transpose_btc_to_bct(&shared_out_btc);
    let n_0_fixture = common::npy::load(format!("{}/predictor_N_0.npy", fixtures_path));
    let n_1_fixture = common::npy::load(format!("{}/predictor_N_1.npy", fixtures_path));
    let n_2_fixture = common::npy::load(format!("{}/predictor_N_2.npy", fixtures_path));
    let n_proj_fixture = common::npy::load(format!("{}/predictor_N_proj.npy", fixtures_path));

    n = prosody.noise_blocks[0].forward(&n, &style);
    let (n_0_diff, n_0_py_mean) = check_layer("N.0", &n, &n_0_fixture);

    n = prosody.noise_blocks[1].forward(&n, &style);
    let (n_1_diff, n_1_py_mean) = check_layer("N.1 (upsample)", &n, &n_1_fixture);

    n = prosody.noise_blocks[2].forward(&n, &style);
    let (n_2_diff, n_2_py_mean) = check_layer("N.2", &n, &n_2_fixture);

    let n_proj_out = prosody.noise_proj.forward(&n);
    let (n_proj_diff, n_proj_py_mean) = check_layer("N_proj", &n_proj_out, &n_proj_fixture);

    println!("\n======== Summary of per-layer drifts ========");
    let all_layers: [(&str, f32, f32); 10] = [
        ("dur_logits", dur_max_diff, dur_py_mean),
        ("shared_lstm", shared_diff, shared_py_mean),
        ("F0.0", f0_0_diff, f0_0_py_mean),
        ("F0.1 (upsample)", f0_1_diff, f0_1_py_mean),
        ("F0.2", f0_2_diff, f0_2_py_mean),
        ("F0_proj", f0_proj_diff, f0_proj_py_mean),
        ("N.0", n_0_diff, n_0_py_mean),
        ("N.1 (upsample)", n_1_diff, n_1_py_mean),
        ("N.2", n_2_diff, n_2_py_mean),
        ("N_proj", n_proj_diff, n_proj_py_mean),
    ];
    for (name, diff, py_mean) in all_layers.iter() {
        let rel = if *py_mean > 0.0 { diff / py_mean } else { 0.0 };
        println!(
            "{:<18}: max|Δ|={:.3e}  rel={:.3e}  py_mean|x|={:.3}",
            name, diff, rel, py_mean
        );
    }

    // Identify the first failing layer.
    if let Some((first_bad, drift, py_mean)) = all_layers
        .iter()
        .find(|(_, d, pm)| !layer_ok(*d, *pm))
        .map(|(n, d, pm)| (*n, *d, *pm))
    {
        println!(
            "\n>>> FIRST FAILING LAYER: {} (drift {:.6e}, py_mean|x|={:.6})",
            first_bad, drift, py_mean
        );
    } else {
        println!(
            "\n>>> ALL LAYERS PASS (abs tol {:.0e} / rel tol {:.0e})",
            ABS_TOLERANCE, REL_TOLERANCE
        );
    }

    // Assert the whole chain is within tolerance.
    for (label, diff, py_mean) in all_layers.iter() {
        assert!(
            layer_ok(*diff, *py_mean),
            "{} drift {:.6e} exceeds tolerance (py_mean|x|={:.6})",
            label,
            diff,
            py_mean
        );
    }

    println!("\n=== ProsodyPredictor golden: ALL LAYERS PASS ===");
    Ok(())
}