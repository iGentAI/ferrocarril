//! Example: Text-to-Speech Synthesis with Ferrocarril
//! 
//! This example demonstrates how to use the Ferrocarril TTS system to synthesize speech
//! from text using the Kokoro model architecture.
//! 
//! Usage:
//! ```bash
//! cargo run --example tts_synthesis --features weights
//! ```

use ferrocarril_tts::{FerroModel, FerroError};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Convert f32 audio samples to 16-bit PCM for WAV file
fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&s| {
        let clamped = s.max(-1.0).min(1.0);
        (clamped * 32767.0) as i16
    }).collect()
}

/// Write audio samples to a simple WAV file
fn write_wav(filename: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let i16_samples = f32_to_i16(samples);
    
    let mut file = File::create(filename)?;
    
    // WAV header
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + i16_samples.len() as u32 * 2).to_le_bytes())?; // File size - 8
    file.write_all(b"WAVE")?;
    
    // Format chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Chunk size
    file.write_all(&1u16.to_le_bytes())?;  // Audio format (PCM)
    file.write_all(&1u16.to_le_bytes())?;  // Number of channels
    file.write_all(&sample_rate.to_le_bytes())?; // Sample rate
    file.write_all(&(sample_rate * 2).to_le_bytes())?; // Byte rate
    file.write_all(&2u16.to_le_bytes())?;  // Block align
    file.write_all(&16u16.to_le_bytes())?; // Bits per sample
    
    // Data chunk
    file.write_all(b"data")?;
    file.write_all(&(i16_samples.len() as u32 * 2).to_le_bytes())?; // Data size
    
    // Write samples
    for sample in i16_samples {
        file.write_all(&sample.to_le_bytes())?;
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🎙️ Ferrocarril TTS Synthesis Example");
    println!("=====================================\n");
    
    // Check if weights directory exists
    let weights_dir = "./ferrocarril_weights";
    let config_path = "./ferrocarril_weights/config.json";
    
    if !Path::new(weights_dir).exists() {
        eprintln!("❌ Weights directory not found: {}", weights_dir);
        eprintln!("   Please download and convert the Kokoro weights first.");
        eprintln!("   See README for instructions.");
        return Err("Weights not found".into());
    }
    
    // Load the model
    println!("📦 Loading Ferrocarril model...");
    let mut model = FerroModel::from_weights_dir(weights_dir, config_path)?;
    
    // Get model info
    let (name, sample_rate, params) = model.info();
    println!("✅ Model loaded: {}", name);
    println!("   Parameters: {}", params);
    println!("   Sample rate: {}Hz\n", sample_rate);
    
    // List available voices
    println!("🎤 Available voices:");
    let voices = model.list_voices()?;
    for voice in &voices {
        println!("   - {}", voice);
    }
    println!();
    
    // Example texts to synthesize
    let examples = vec![
        ("Hello world", "af_heart", 1.0),
        ("Welcome to Ferrocarril text to speech", "af_bella", 1.0),
        ("This is fast neural speech synthesis in Rust", "af_sarah", 0.9),
    ];
    
    println!("🎯 Synthesizing {} examples...\n", examples.len());
    
    for (i, (text, voice, speed)) in examples.iter().enumerate() {
        println!("Example {}: \"{}\"", i + 1, text);
        println!("   Voice: {}, Speed: {:.1}x", voice, speed);
        
        match model.synthesize(text, voice, *speed) {
            Ok(audio) => {
                let duration = audio.len() as f32 / sample_rate as f32;
                println!("   ✓ Generated {:.2}s of audio ({} samples)", duration, audio.len());
                
                // Save to WAV file
                let filename = format!("ferrocarril_example_{}.wav", i + 1);
                write_wav(&filename, &audio, sample_rate)?;
                println!("   ✓ Saved to: {}", filename);
            }
            Err(e) => {
                println!("   ✗ Error: {}", e);
            }
        }
        println!();
    }
    
    // Demonstrate batch synthesis
    println!("🎯 Batch synthesis example...");
    let batch_texts = vec![
        "First sentence in the batch.",
        "Second sentence here.",
        "Third and final sentence.",
    ];
    
    match model.synthesize_batch(&batch_texts, "af_heart", 1.0) {
        Ok(results) => {
            println!("✅ Batch complete: {} audio clips", results.len());
            for (i, audio) in results.iter().enumerate() {
                let duration = audio.len() as f32 / sample_rate as f32;
                println!("   [{}] {:.2}s ({} samples)", i + 1, duration, audio.len());
            }
        }
        Err(e) => {
            println!("❌ Batch error: {}", e);
        }
    }
    
    println!("\n✨ Synthesis complete!");
    Ok(())
}