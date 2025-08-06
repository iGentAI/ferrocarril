//! Ferrocarril - Fast Rust Neural Network Inference
//! 
//! A high-performance text-to-speech system implemented in pure Rust,
//! based on the Kokoro/StyleTTS2 architecture.
//! 
//! # Production Status
//! 
//! **CURRENT LIMITATIONS**: This implementation uses basic spectral processing
//! rather than full FFT/iFFT. Production deployment requires proper FFT library
//! integration for optimal audio quality.
//! 
//! **WEIGHT REQUIREMENTS**: Requires real Kokoro-82M weights (81.8M parameters)
//! converted using the weight_converter.py tool. No synthetic weights supported.
//! 
//! # Quick Start
//! 
//! ```no_run
//! use ferrocarril_tts::FerroModel;
//! 
//! // Load model with real weights
//! let mut model = FerroModel::from_weights_dir("./ferrocarril_weights", "./config.json")?;
//! 
//! // Synthesize speech
//! let audio = model.synthesize("Hello world", "af_heart", 1.0)?;
//! 
//! // Audio is f32 samples at 24kHz
//! println!("Generated {} audio samples", audio.len());
//! ```

pub mod model;

// Re-export the main API types
pub use model::FerroModel;
pub use model::load_default_model;

// Re-export KokoroInference for direct access when needed
pub use model::kokoro_inference::KokoroInference;

// Re-export core types that users might need
pub use ferrocarril_core::{Config, FerroError};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exports() {
        // Ensure our main exports are available
        let _ = std::any::type_name::<FerroModel>();
        let _ = std::any::type_name::<FerroError>();
        let _ = std::any::type_name::<Config>();
    }
}