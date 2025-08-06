//! Main executable for Ferrocarril TTS engine

mod model;

use ferrocarril_core::tensor::Tensor;
use ferrocarril_dsp;
use std::error::Error;
use clap::{Parser, Subcommand};
use crate::model::KokoroInference;

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
        Some(Commands::Infer { text, output, model: _model, voice, speed }) => {
            let model = KokoroInference::load_from_weights("ferrocarril_weights", "ferrocarril_weights/config.json")?;
            
            // Perform real neural inference
            let audio_data = match voice {
                Some(voice_name) => {
                    let mut model = model;
                    model.infer_text(&text, &voice_name, speed)?
                }
                None => {
                    let mut model = model;
                    model.infer_text(&text, "af_heart", speed)?
                }
            };
            
            // Save real neural speech
            let audio_tensor = Tensor::from_data(audio_data.clone(), vec![audio_data.len()]);
            ferrocarril_dsp::save_wav(&audio_tensor, &output)?;
            
            println!("Real neural speech generated: {}", output);
        }
        Some(Commands::Demo) => {
            println!("NO DEMO - REAL TTS ONLY");
            println!("Use: cargo run --bin ferrocarril-tts -- infer --text \"Your text\" --output speech.wav");
        }
        None => {
            println!("Ferrocarril TTS - Real neural speech with 81.8M parameters");
            println!("Use --help for commands.");
        }
    }
    
    Ok(())
}