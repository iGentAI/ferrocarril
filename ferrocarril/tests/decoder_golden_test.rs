//! Decoder golden-reference test for Phase 3 numerical validation.
//!
//! Walks the Rust `Decoder` (StyleTTS2 decode stack + iSTFTNet generator)
//! on inputs derived directly from the Python kmodel fixtures so the
//! test isolates the decoder from upstream numerical drift:
//!
//! ```text
//! t_en           = load text_encoder.npy            # [1, 512, 7]
//! pred_dur       = load pred_dur.npy                # [7]
//! pred_aln_trg   = build_alignment(pred_dur)        # [7, 56]
//! asr            = t_en @ pred_aln_trg              # [1, 512, 56]
//! F0             = load predictor_F0_proj.npy       # [1, 1, 112] -> [1, 112]
//! N              = load predictor_N_proj.npy        # [1, 1, 112] -> [1, 112]
//! style          = ref_s[:, :128]                   # [1, 128]
//!
//! # Deterministic stack (no SineGen):
//! F0_down   = decoder.F0_conv(F0.unsqueeze(1))       # [1, 1, 56]
//! N_down    = decoder.N_conv(N.unsqueeze(1))         # [1, 1, 56]
//! asr_res   = decoder.asr_res(asr)                   # [1, 64, 56]
//! x0        = decoder.encode(cat([asr, F0_down, N_down]), s)
//!                                                    # [1, 1024, 56]
//! x1..x4    = decoder.decode[i](cat([x, asr_res, F0_down, N_down]), s)
//!
//! # Stochastic stack (has SineGen randomness):
//! audio     = decoder.generator(x4, s, F0)
//! ```
//!
//! Each deterministic layer is compared against its Python fixture with a
//! tight tolerance. The final audio is compared with a much looser
//! tolerance because the iSTFTNet generator adds random phase and noise
//! via `SineGen`, so it is non-deterministic across runs even in Python.
//! The audio check serves as a "produced, finite, non-zero" smoke test
//! until Python and Rust are forced to share a deterministic RNG seed.

#![cfg(feature = "weights")]

mod common;

use std::error::Error;

use ferrocarril_core::{tensor::Tensor, weights_binary::BinaryWeightLoader, LoadWeightsBinary};
use ferrocarril_nn::Forward;
use ferrocarril_nn::vocoder::Decoder;

/// Absolute tolerance floor for the deterministic decoder layers.
const ABS_TOLERANCE: f32 = 1e-3;
/// Relative tolerance ceiling for scale-dependent drifts.
const REL_TOLERANCE: f32 = 5e-4;

fn layer_ok(max_diff: f32, py_mean_abs: f32) -> bool {
    let rel_cap = (REL_TOLERANCE * py_mean_abs).max(ABS_TOLERANCE);
    max_diff < rel_cap
}

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
    let py_mean = common::mean_abs(py);
    let rel = if py_mean > 0.0 { max_diff / py_mean } else { 0.0 };
    println!(
        "{:<30} shape={:?} Rust mean|x|={:.6} Py mean|x|={:.6} max|Δ|={:.6e} rel={:.6e}",
        label,
        rust.shape(),
        common::mean_abs(rust_data),
        py_mean,
        max_diff,
        rel
    );
    (max_diff, py_mean)
}

/// Build the `pred_aln_trg [seq_len, total_frames]` one-hot alignment
/// matrix from a per-position duration vector.
fn build_alignment(pred_dur: &[i64], seq_len: usize) -> Tensor<f32> {
    assert_eq!(pred_dur.len(), seq_len, "pred_dur length mismatch");
    for (i, &d) in pred_dur.iter().enumerate() {
        assert!(d >= 0, "pred_dur[{}] = {} is negative", i, d);
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

/// Compute `[B, C, T1] @ [T1, S] -> [B, C, S]`.
fn matmul_bct_by_ts(x: &Tensor<f32>, y: &Tensor<f32>) -> Tensor<f32> {
    let (b, c, t1) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let (t2, s) = (y.shape()[0], y.shape()[1]);
    assert_eq!(t1, t2, "inner dim mismatch: x={:?} y={:?}", x.shape(), y.shape());
    let mut out = vec![0.0f32; b * c * s];
    for bb in 0..b {
        for cc in 0..c {
            for ss in 0..s {
                let mut acc = 0.0f32;
                for k in 0..t1 {
                    acc += x[&[bb, cc, k]] * y[&[k, ss]];
                }
                out[bb * c * s + cc * s + ss] = acc;
            }
        }
    }
    Tensor::from_data(out, vec![b, c, s])
}

/// Concatenate 3D tensors along the channel dim. Mirrors
/// `Decoder::concat_channels` but inline here so the test can drive the
/// decoder stack manually.
fn concat_channels(tensors: &[&Tensor<f32>]) -> Tensor<f32> {
    assert!(!tensors.is_empty());
    let first_shape = tensors[0].shape();
    let batch = first_shape[0];
    let time = first_shape[2];
    let total_channels: usize = tensors.iter().map(|t| t.shape()[1]).sum();
    let mut result = vec![0.0f32; batch * total_channels * time];
    let mut channel_offset = 0usize;
    for tensor in tensors {
        let channels = tensor.shape()[1];
        for b in 0..batch {
            for c in 0..channels {
                for t in 0..time {
                    let src_idx = b * channels * time + c * time + t;
                    let dst_idx = b * total_channels * time + (channel_offset + c) * time + t;
                    result[dst_idx] = tensor.data()[src_idx];
                }
            }
        }
        channel_offset += channels;
    }
    Tensor::from_data(result, vec![batch, total_channels, time])
}

/// Root-mean-square of a sample buffer.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut ss = 0.0f64;
    for &s in samples {
        ss += (s as f64) * (s as f64);
    }
    (ss / samples.len() as f64).sqrt() as f32
}

/// Pearson correlation coefficient for two vectors.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "pearson: length mismatch");
    let n = a.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for i in 0..a.len() {
        let xa = a[i] - mean_a;
        let xb = b[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let denom = (da * db).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    num / denom
}

#[test]
fn test_decoder_golden_vs_python_audio() -> Result<(), Box<dyn Error>> {
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

    // ---- Inputs from fixtures --------------------------------------------
    let t_en_fixture = common::npy::load(format!("{}/text_encoder.npy", fixtures_path));
    assert_eq!(t_en_fixture.shape, vec![1, 512, 7]);
    let t_en = Tensor::from_data(t_en_fixture.as_f32().to_vec(), vec![1, 512, 7]);

    let pred_dur_fixture = common::npy::load(format!("{}/pred_dur.npy", fixtures_path));
    assert_eq!(pred_dur_fixture.shape, vec![7]);
    let pred_dur: Vec<i64> = pred_dur_fixture.as_i64().to_vec();

    let alignment = build_alignment(&pred_dur, 7);
    assert_eq!(alignment.shape(), &[7, 56]);

    let asr = matmul_bct_by_ts(&t_en, &alignment);
    assert_eq!(asr.shape(), &[1, 512, 56]);

    let f0_fixture = common::npy::load(format!("{}/predictor_F0_proj.npy", fixtures_path));
    let n_fixture = common::npy::load(format!("{}/predictor_N_proj.npy", fixtures_path));
    assert_eq!(f0_fixture.shape, vec![1, 1, 112]);
    assert_eq!(n_fixture.shape, vec![1, 1, 112]);
    // Python does `F0_curve.unsqueeze(1)` inside the Decoder, so the
    // external shape is [B, T]. Reshape our [1, 1, 112] fixture into
    // [1, 112] to match.
    let f0_curve = Tensor::from_data(f0_fixture.as_f32().to_vec(), vec![1, 112]);
    let n_curve = Tensor::from_data(n_fixture.as_f32().to_vec(), vec![1, 112]);

    let ref_s_fixture = common::npy::load(format!("{}/ref_s.npy", fixtures_path));
    assert_eq!(ref_s_fixture.shape, vec![1, 256]);
    let mut style_data = vec![0.0f32; 128];
    style_data.copy_from_slice(&ref_s_fixture.as_f32()[..128]);
    let style = Tensor::from_data(style_data, vec![1, 128]);

    // ---- Decoder ---------------------------------------------------------
    let mut decoder = Decoder::new(
        512,                                                   // dim_in
        128,                                                   // style_dim
        80,                                                    // dim_out
        vec![3, 7, 11],                                        // resblock_kernel_sizes
        vec![10, 6],                                           // upsample_rates
        512,                                                   // upsample_initial_channel
        vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],     // resblock_dilation_sizes
        vec![20, 12],                                          // upsample_kernel_sizes
        20,                                                    // gen_istft_n_fft
        5,                                                     // gen_istft_hop_size
    );
    decoder.load_weights_binary(&loader, "decoder", "module")?;

    println!("\n======== Decoder deterministic-stack bisection ========");

    // ---- F0_conv / N_conv ------------------------------------------------
    // Python: self.F0_conv(F0.unsqueeze(1)) and same for N. Build the
    // `[B, 1, T]` input here.
    let f0_bct = Tensor::from_data(f0_curve.data().to_vec(), vec![1, 1, 112]);
    let n_bct = Tensor::from_data(n_curve.data().to_vec(), vec![1, 1, 112]);
    let f0_down = decoder.f0_conv.forward(&f0_bct);
    let n_down = decoder.n_conv.forward(&n_bct);
    assert_eq!(f0_down.shape(), &[1, 1, 56]);
    assert_eq!(n_down.shape(), &[1, 1, 56]);

    let f0_fixture_d = common::npy::load(format!("{}/decoder_F0_conv.npy", fixtures_path));
    let n_fixture_d = common::npy::load(format!("{}/decoder_N_conv.npy", fixtures_path));
    let (f0_diff, f0_py) = check_layer("decoder.F0_conv", &f0_down, &f0_fixture_d);
    let (n_diff, n_py) = check_layer("decoder.N_conv", &n_down, &n_fixture_d);

    // ---- asr_res ---------------------------------------------------------
    let asr_res = decoder.asr_res.forward(&asr);
    assert_eq!(asr_res.shape(), &[1, 64, 56]);
    let asr_res_fixture = common::npy::load(format!("{}/decoder_asr_res.npy", fixtures_path));
    let (asr_res_diff, asr_res_py) = check_layer("decoder.asr_res", &asr_res, &asr_res_fixture);

    // ---- encode(cat([asr, F0_down, N_down]), s) --------------------------
    let encode_in = concat_channels(&[&asr, &f0_down, &n_down]);
    assert_eq!(encode_in.shape(), &[1, 514, 56]);
    let encode_out = decoder.encode.forward(&encode_in, &style);
    assert_eq!(encode_out.shape(), &[1, 1024, 56]);
    let encode_fixture = common::npy::load(format!("{}/decoder_encode.npy", fixtures_path));
    let (encode_diff, encode_py) = check_layer("decoder.encode", &encode_out, &encode_fixture);

    // ---- decode.0..3 -----------------------------------------------------
    let mut x = encode_out;
    let mut res = true;
    let mut decode_diffs = Vec::new();
    for i in 0..decoder.decode.len() {
        if res {
            x = concat_channels(&[&x, &asr_res, &f0_down, &n_down]);
        }
        x = decoder.decode[i].forward(&x, &style);
        let fixture_path = format!("{}/decoder_decode_{}.npy", fixtures_path, i);
        let fixture = common::npy::load(&fixture_path);
        let (diff, py_mean) = check_layer(&format!("decoder.decode.{}", i), &x, &fixture);
        decode_diffs.push((i, diff, py_mean));
        if decoder.decode[i].is_upsample() {
            res = false;
        }
    }

    // ---- Generator bisection: conv_post RMS comparison -------------------
    // The Generator's pre-iSTFT `conv_post` output is captured in the
    // `decoder_generator_conv_post.npy` fixture (shape [1, 22, 6721]).
    // It is still stochastic because `noise_convs` / `noise_res` consume
    // `har_source` from SineGen, but its global RMS is stable across
    // runs. If this matches Python, any final-audio drift is inside the
    // iSTFT inverse; if it doesn't match, the bug is in the upsample/
    // resblock/noise chain upstream.
    println!("\n======== Generator conv_post (pre-iSTFT) bisection ========");
    let conv_post_out = decoder
        .generator
        .forward_to_conv_post(&x, &style, &f0_curve)
        .map_err(|e| format!("Generator.forward_to_conv_post failed: {}", e))?;
    assert_eq!(
        conv_post_out.shape(),
        &[1, 22, 6721],
        "conv_post output shape mismatch (expected [1, 22, 6721])"
    );
    let conv_post_fixture = common::npy::load(format!(
        "{}/decoder_generator_conv_post.npy",
        fixtures_path
    ));
    assert_eq!(conv_post_fixture.shape, vec![1, 22, 6721]);
    let conv_post_rust_rms = rms(conv_post_out.data());
    let conv_post_py_rms = rms(conv_post_fixture.as_f32());
    let conv_post_rms_rel =
        (conv_post_rust_rms - conv_post_py_rms).abs() / conv_post_py_rms.max(1e-8);
    println!(
        "Generator conv_post RMS  Rust={:.6} Py={:.6} rel diff={:.4} (tol 0.10)",
        conv_post_rust_rms, conv_post_py_rms, conv_post_rms_rel
    );
    assert!(
        conv_post_out.data().iter().all(|v| v.is_finite()),
        "Generator conv_post contains NaN or Inf"
    );
    assert!(
        conv_post_rms_rel < 0.10,
        "Generator conv_post RMS rel diff {:.4} exceeds 10% \
         (Rust={:.6}, Py={:.6}). This means the Generator's \
         upsample/resblock/noise_res chain is structurally wrong BEFORE the \
         iSTFT inverse step — bisect inside the upsample loop to find which \
         resblock introduces the amplification.",
        conv_post_rms_rel,
        conv_post_rust_rms,
        conv_post_py_rms
    );

    // ---- Generator (stochastic) ------------------------------------------
    println!("\n======== Generator phase-invariant audio comparison ========");
    let audio = decoder
        .generator
        .forward(&x, &style, &f0_curve)
        .map_err(|e| format!("Generator forward failed: {}", e))?;
    let audio_fixture = common::npy::load(format!("{}/audio.npy", fixtures_path));
    assert_eq!(audio_fixture.shape, vec![33600]);
    let py_audio = audio_fixture.as_f32();
    assert_eq!(
        audio.data().len(),
        py_audio.len(),
        "audio length mismatch: Rust {} vs Python {}",
        audio.data().len(),
        py_audio.len()
    );

    let rust_audio = audio.data();

    // Sanity: no NaN/Inf, non-zero.
    assert!(
        rust_audio.iter().all(|x| x.is_finite()),
        "generator audio contains NaN or Inf"
    );
    assert!(
        !rust_audio.iter().all(|&x| x.abs() < 1e-8),
        "generator audio is all zeros - functionally dead"
    );

    // ---- 1) Global RMS within 10 % ---------------------------------------
    let rust_rms = rms(rust_audio);
    let py_rms = rms(py_audio);
    let rms_rel_diff = (rust_rms - py_rms).abs() / py_rms.max(1e-8);
    println!(
        "Global RMS            Rust={:.6} Py={:.6} rel diff={:.4}",
        rust_rms, py_rms, rms_rel_diff
    );
    assert!(
        rms_rel_diff < 0.10,
        "Global audio RMS relative diff {:.4} exceeds 10% (Rust={:.6}, Py={:.6})",
        rms_rel_diff,
        rust_rms,
        py_rms
    );

    // ---- 2) Per-segment RMS profile (diagnostic + correlation gate) -----
    // 60 non-overlapping segments × 560 samples ≈ 23 ms per segment at 24 kHz.
    let segment_len = 560usize;
    let n_segments = rust_audio.len() / segment_len;
    assert_eq!(
        n_segments, 60,
        "expected 60 segments of {} samples each, got {}",
        segment_len, n_segments
    );

    let mut rust_seg_rms = Vec::with_capacity(n_segments);
    let mut py_seg_rms = Vec::with_capacity(n_segments);
    for seg in 0..n_segments {
        let start = seg * segment_len;
        let end = start + segment_len;
        rust_seg_rms.push(rms(&rust_audio[start..end]));
        py_seg_rms.push(rms(&py_audio[start..end]));
    }

    println!(
        "\nPer-segment RMS profile ({} segments × {} samples, diagnostic only):",
        n_segments, segment_len
    );
    println!("  seg |  Rust RMS  |  Py RMS   |  abs diff |  rel");
    println!("  ----+------------+-----------+-----------+------");
    for seg in (0..n_segments).step_by(5) {
        let r = rust_seg_rms[seg];
        let p = py_seg_rms[seg];
        let abs_diff = (r - p).abs();
        let rel = if p > 0.0 { abs_diff / p } else { f32::INFINITY };
        println!(
            "  {:3} | {:10.6} | {:9.6} | {:9.6} | {:.4}",
            seg, r, p, abs_diff, rel
        );
    }

    // ---- 3) Pearson correlation of per-segment RMS profile ≥ 0.95 -------
    let correlation = pearson(&rust_seg_rms, &py_seg_rms);
    println!("\nPer-segment RMS profile correlation = {:.6}", correlation);
    assert!(
        correlation >= 0.95,
        "Per-segment RMS profile correlation {:.6} < 0.95 (Rust and Python envelopes disagree)",
        correlation
    );

    println!("\n=== Generator audio: phase-invariant comparison passes ===");

    println!("\n======== Summary of deterministic layer drifts ========");
    let ordered: Vec<(String, f32, f32)> = [
        ("decoder.F0_conv".to_string(), f0_diff, f0_py),
        ("decoder.N_conv".to_string(), n_diff, n_py),
        ("decoder.asr_res".to_string(), asr_res_diff, asr_res_py),
        ("decoder.encode".to_string(), encode_diff, encode_py),
    ]
    .into_iter()
    .chain(
        decode_diffs
            .into_iter()
            .map(|(i, d, pm)| (format!("decoder.decode.{}", i), d, pm)),
    )
    .collect();
    for (name, diff, py_mean) in &ordered {
        let rel = if *py_mean > 0.0 { diff / py_mean } else { 0.0 };
        println!(
            "{:<18}: max|Δ|={:.3e}  rel={:.3e}  py_mean|x|={:.3}",
            name, diff, rel, py_mean
        );
    }

    if let Some((first_bad, drift, py_mean)) = ordered
        .iter()
        .find(|(_, d, pm)| !layer_ok(*d, *pm))
        .map(|(n, d, pm)| (n.clone(), *d, *pm))
    {
        println!(
            "\n>>> FIRST FAILING LAYER: {} (drift {:.6e}, py_mean|x|={:.6})",
            first_bad, drift, py_mean
        );
    } else {
        println!(
            "\n>>> ALL DETERMINISTIC LAYERS PASS (abs tol {:.0e} / rel tol {:.0e})",
            ABS_TOLERANCE, REL_TOLERANCE
        );
    }

    for (name, diff, py_mean) in ordered {
        assert!(
            layer_ok(diff, py_mean),
            "{} drift {:.6e} exceeds tolerance (py_mean|x|={:.6})",
            name,
            diff,
            py_mean
        );
    }

    println!("\n=== Decoder golden: DETERMINISTIC LAYERS PASS (generator is stochastic) ===");
    Ok(())
}