//! Main executable for Ferrocarril TTS engine

mod model;

use ferrocarril_core::{Config, tensor::Tensor, PhonesisG2P};
use ferrocarril_dsp;
use ferrocarril_nn::vocoder::{Generator, Decoder};
use std::error::Error;
use std::path::Path;
use clap::{Parser, Subcommand};
use crate::model::FerroModel;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on text
    Infer {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,
        
        /// Output WAV file path
        #[arg(short, long)]
        output: String,
        
        /// Model path for binary weights (optional)
        #[arg(short, long)]
        model: Option<String>,
        
        /// Voice name (optional)
        #[arg(short, long)]
        voice: Option<String>,
        
        /// Speech speed factor (optional, default: 1.0)
        #[arg(short, long, default_value = "1.0")]
        speed: f32,
    },
    /// Run a demo to test components
    Demo,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Some(Commands::Infer { text, output, model, voice, speed }) => {
            // Load configuration
            let config = Config::from_json("config.json")?;
            
            // Load model
            let model = match model {
                Some(model_path) => {
                    #[cfg(feature = "weights")]
                    {
                        FerroModel::load_binary(&model_path, config)?
                    }
                    #[cfg(not(feature = "weights"))]
                    {
                        return Err("Binary weight loading requires the 'weights' feature".into());
                    }
                }
                None => FerroModel::load("config.json", config)?
            };
            
            // Perform inference
            let audio_data = match voice {
                Some(voice_name) => {
                    let voice_embedding = model.load_voice(&voice_name)?;
                    model.infer_with_voice(&text, &voice_embedding, speed)?
                }
                None => model.infer(&text)?
            };
            
            // Create audio tensor
            let audio_tensor = Tensor::from_data(audio_data.clone(), vec![audio_data.len()]);
            
            // Save WAV file
            ferrocarril_dsp::save_wav(&audio_tensor, &output)?;
            
            println!("Audio generated and saved to: {}", output);
        }
        Some(Commands::Demo) => {
            println!("Running Ferrocarril TTS demo...");
            
            // Test vocoder components
            test_vocoder_demo()?;
            
            // Test G2P component - using directly integrated Phonesis
            test_g2p_demo()?;
            
            // Test basic audio generation
            generate_test_audio()?;
            
            println!("Demo completed successfully!");
        }
        None => {
            println!("Welcome to Ferrocarril TTS!");
            println!("Use --help to see available commands.");
        }
    }
    
    Ok(())
}

fn test_vocoder_demo() -> Result<(), Box<dyn Error>> {
    println!("\n=== Testing Vocoder Components ===");
    
    // Create a small Generator for testing
    let generator = Generator::new(
        64,                         // style_dim
        vec![3, 7, 11],            // resblock_kernel_sizes
        vec![2, 2],                // upsample_rates
        128,                       // upsample_initial_channel
        vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]], // resblock_dilation_sizes
        vec![4, 4],                // upsample_kernel_sizes
        16,                        // gen_istft_n_fft
        4,                         // gen_istft_hop_size
    );
    
    // Create input tensors
    let batch_size = 1;
    let time_dim = 16;
    
    // Input features
    let x = Tensor::from_data(vec![0.1; batch_size * 128 * time_dim], vec![batch_size, 128, time_dim]);
    
    // Style vector
    let s = Tensor::from_data(vec![0.1; batch_size * 64], vec![batch_size, 64]);
    
    // F0 curve with realistic values
    let mut f0_values = vec![0.0; batch_size * time_dim];
    for t in 0..time_dim {
        f0_values[t] = 440.0 + (t as f32 * 10.0).sin() * 50.0; // Varying around 440Hz
    }
    let f0 = Tensor::from_data(f0_values, vec![batch_size, time_dim]);
    
    println!("Running Generator forward pass...");
    let output = generator.forward(&x, &s, &f0);
    
    println!("Generator output shape: {:?}", output.shape());
    println!("Generator test passed!");
    
    // Test Decoder
    let decoder = Decoder::new(
        128,                       // dim_in
        64,                        // style_dim
        80,                        // dim_out
        vec![3, 7, 11],           // resblock_kernel_sizes
        vec![2, 2],               // upsample_rates
        128,                      // upsample_initial_channel
        vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]], // resblock_dilation_sizes
        vec![4, 4],               // upsample_kernel_sizes
        16,                       // gen_istft_n_fft
        4,                        // gen_istft_hop_size
    );
    
    // Test ASR features
    let asr = Tensor::from_data(vec![0.1; batch_size * 128 * time_dim], vec![batch_size, 128, time_dim]);
    
    // Noise curve
    let noise = Tensor::from_data(vec![0.01; batch_size * time_dim], vec![batch_size, time_dim]);
    
    println!("Running Decoder forward pass...");
    let decoder_output = decoder.forward(&asr, &f0, &noise, &s);
    
    println!("Decoder output shape: {:?}", decoder_output.shape());
    println!("Decoder test passed!");
    
    Ok(())
}

fn test_g2p_demo() -> Result<(), Box<dyn Error>> {
    println!("\n=== Testing G2P Component (Phonesis) ===");
    
    // Create PhonesisG2P instance directly from ferrocarril-core
    let g2p = PhonesisG2P::new("en-us")?;
    
    let test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Pineapple pizza is actually quite nice.",
        "I scream, you scream, we all scream for ice cream!",
    ];
    
    for text in &test_texts {
        let phonemes = g2p.convert(text)?;
        println!("Text: '{}'", text);
        println!("Phonemes: '{}'", phonemes);
    }
    
    println!("G2P test passed!");
    Ok(())
}

fn generate_test_audio() -> Result<(), Box<dyn Error>> {
    println!("\n=== Generating Test Audio ===");
    
    // Generate a simple sine wave
    let sample_rate = 24000;
    let duration = 1.0; // seconds
    let frequency = 440.0; // Hz (A4)
    
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut sine_wave = vec![0.0f32; num_samples];
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        sine_wave[i] = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
    }
    
    // Create audio tensor
    let audio_tensor = Tensor::from_data(sine_wave.clone(), vec![sine_wave.len()]);
    
    // Save to WAV file
    ferrocarril_dsp::save_wav(&audio_tensor, "demo_audio.wav")?;
    
    println!("Test audio saved to: demo_audio.wav");
    println!("Audio generation test passed!");
    
    Ok(())
}