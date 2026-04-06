//! End-to-end real-voice inference smoke test.
//!
//! Exercises the full `FerroModel::infer_with_phonemes` production path
//! with the real Kokoro-82M weights, the real `af_heart` voice pack,
//! and the canonical 5-phoneme IPA input that matches the
//! `tests/fixtures/kmodel/audio.npy` Python reference. This is the only
//! end-to-end smoke test of the production API; rigorous numerical
//! correctness is enforced by the per-component golden tests
//! (bert / text_encoder / duration_encoder / bert_encoder /
//! prosody_predictor / decoder).
//!
//! What this test does NOT do:
//!   - It does not run G2P. It feeds the phoneme string `hɛlqʊ`
//!     directly so the test is independent of the G2P implementation.
//!   - It does not compare raw samples against Python because the
//!     iSTFTNet generator is stochastic.
//!
//! What this test DOES do:
//!   - Load the real Kokoro weights via `FerroModel::load_binary`.
//!   - Load the real `af_heart` voice pack (`[510, 256]` raw tensor).
//!   - Run `infer_with_phonemes` on the canonical input.
//!   - Assert the audio is finite, non-zero, the right length, and has
//!     a global RMS within ±30 % of the Python fixture's ~0.0456.
//!   - Save the WAV to `test_output/end_to_end_af_heart.wav` for
//!     listening.

#![cfg(feature = "weights")]

mod common;

use std::error::Error;
use std::fs;
use std::path::Path;

use ferrocarril::model::FerroModel;
use ferrocarril_core::{tensor::Tensor, Config};

/// Expected audio length and RMS, derived from the Python kmodel
/// fixture. The fixture was dumped with the same input phonemes via
/// `scripts/validate_kmodel.py`.
const EXPECTED_AUDIO_LEN: usize = 33600;
const EXPECTED_PYTHON_RMS: f32 = 0.0456;
const RMS_REL_TOL: f32 = 0.30;

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

#[test]
fn test_end_to_end_real_voice_inference() -> Result<(), Box<dyn Error>> {
    let weights_path = common::find_weights_path().ok_or_else(|| -> Box<dyn Error> {
        "ferrocarril_weights not found; run `python3 weight_converter.py \
         --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights` first"
            .into()
    })?;
    println!("Loading weights from {}", weights_path);

    // Load the real Kokoro config from the converter output.
    let config_path = format!("{}/config.json", weights_path);
    let config = Config::from_json(&config_path)?;
    println!(
        "Config: vocab_size={}, hidden_dim={}, style_dim={}, n_layer={}",
        config.n_token, config.hidden_dim, config.style_dim, config.n_layer
    );

    // Build the model with real weights.
    let model = FerroModel::load_binary(&weights_path, config)?;

    // Load the real `af_heart` voice pack. `load_voice` returns the raw
    // `[510, 256]` tensor; `infer_with_phonemes` indexes into it
    // internally based on the input phoneme count.
    let voice_pack = model.load_voice("af_heart")?;
    println!("Loaded af_heart voice pack: shape={:?}", voice_pack.shape());
    assert!(
        voice_pack.shape() == &[510, 256] || voice_pack.shape() == &[510, 1, 256],
        "Unexpected voice pack shape {:?}",
        voice_pack.shape()
    );

    // Canonical 5-phoneme input — matches the input the Python
    // `scripts/validate_kmodel.py` uses to generate `audio.npy`:
    //
    //   input_ids = [0, 50, 86, 54, 59, 135, 0]
    //             = [BOS, h, ɛ, l, q, ʊ, EOS]   // 'q' is vocab index 59
    //
    // The IPA string `hɛlqʊ` is phonetically nonsense but it is
    // deterministic and lets us reuse the same reference audio length
    // and RMS from the Python kmodel fixture. The production
    // `infer_with_phonemes` tokenizer walks the string char-by-char and
    // looks up each IPA scalar in the vocab, so this is bit-equivalent
    // to passing those raw token IDs directly.
    let phonemes = "hɛlqʊ";
    println!("Running inference on phonemes: {:?}", phonemes);

    let audio = model.infer_with_phonemes(phonemes, &voice_pack, 1.0)?;

    // Sanity checks ----------------------------------------------------------
    println!("Generated audio: {} samples", audio.len());
    assert!(
        !audio.is_empty(),
        "Inference produced no audio samples"
    );
    assert_eq!(
        audio.len(),
        EXPECTED_AUDIO_LEN,
        "Audio length mismatch: expected {} samples (matching kmodel/audio.npy), got {}",
        EXPECTED_AUDIO_LEN,
        audio.len()
    );
    assert!(
        audio.iter().all(|s| s.is_finite()),
        "Audio contains NaN or Inf — Generator pipeline broken"
    );
    assert!(
        audio.iter().any(|&s| s.abs() > 1e-6),
        "Audio is all zeros — model is functionally dead"
    );

    // Global RMS check vs Python fixture (loose because the production
    // path uses our own G2P-tokenized phonemes and the iSTFTNet
    // generator is stochastic).
    let observed_rms = rms(&audio);
    let rms_rel_diff = (observed_rms - EXPECTED_PYTHON_RMS).abs() / EXPECTED_PYTHON_RMS;
    let max_abs = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!(
        "Audio stats: rms={:.6} (Python ~{:.6}, rel diff={:.4}) max|x|={:.6}",
        observed_rms, EXPECTED_PYTHON_RMS, rms_rel_diff, max_abs
    );
    assert!(
        rms_rel_diff < RMS_REL_TOL,
        "Audio global RMS rel diff {:.4} exceeds tolerance {:.4} \
         (Rust={:.6}, Python ref={:.6})",
        rms_rel_diff,
        RMS_REL_TOL,
        observed_rms,
        EXPECTED_PYTHON_RMS
    );

    // Save the WAV to disk for listening.
    let output_dir = Path::new("test_output");
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("end_to_end_af_heart.wav");
    let audio_tensor = Tensor::from_data(audio.clone(), vec![audio.len()]);
    ferrocarril_dsp::save_wav(&audio_tensor, output_path.to_str().unwrap())?;
    println!(
        "Saved generated audio to {} (rms={:.6}, max|x|={:.6})",
        output_path.display(),
        observed_rms,
        max_abs
    );

    Ok(())
}