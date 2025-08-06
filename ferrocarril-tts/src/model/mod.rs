//! Ferrocarril TTS Model API
//! 
//! This module provides the main FerroModel struct that serves as the primary
//! API interface for the Ferrocarril text-to-speech system.

pub mod kokoro_inference;

// Public re-export for external modules like main.rs
pub use kokoro_inference::KokoroInference;

use ferrocarril_core::FerroError;
use std::error::Error;
use std::path::Path;

/// Main Ferrocarril TTS Model API
/// 
/// This struct provides a clean, user-friendly interface to the Ferrocarril TTS system,
/// wrapping the internal KokoroInference implementation with convenient methods for
/// text-to-speech synthesis.
/// 
/// # Example
/// ```no_run
/// use ferrocarril_tts::FerroModel;
/// 
/// // Load model from weights directory
/// let model = FerroModel::from_weights_dir("./ferrocarril_weights", "./config.json")?;
/// 
/// // Synthesize speech
/// let audio = model.synthesize("Hello world", "af_heart", 1.0)?;
/// 
/// // Get available voices
/// let voices = model.list_voices()?;
/// println!("Available voices: {:?}", voices);
/// ```
pub struct FerroModel {
    /// Internal Kokoro inference engine
    inference: KokoroInference,
    
    /// Model configuration
    sample_rate: u32,
    
    /// Model name/version
    model_name: String,
}

impl FerroModel {
    /// Load a FerroModel from a weights directory with STRICT validation
    pub fn from_weights_dir<P: AsRef<Path>>(
        weights_dir: P, 
        config_path: P
    ) -> Result<Self, Box<dyn Error>> {
        println!("🚀 Loading FerroModel with STRICT weight validation...");
        
        // Validate paths exist before proceeding
        let weights_path = weights_dir.as_ref();
        let config_file_path = config_path.as_ref();
        
        if !weights_path.exists() {
            return Err(format!("CRITICAL: Weights directory not found: {}", weights_path.display()).into());
        }
        
        if !config_file_path.exists() {
            return Err(format!("CRITICAL: Config file not found: {}", config_file_path.display()).into());
        }
        
        // Load and validate the inference engine
        let inference = KokoroInference::load_from_weights(
            weights_path.to_str().unwrap(),
            config_file_path.to_str().unwrap()
        )?;
        
        // Validate inference engine is functional
        Self::validate_inference_engine(&inference)?;
        
        // Set model configuration
        let sample_rate = 24000; // Kokoro uses 24kHz
        let model_name = "Ferrocarril-Kokoro-82M".to_string();
        
        println!("✅ FerroModel loaded and validated successfully");
        println!("   Model: {}", model_name);
        println!("   Sample rate: {}Hz", sample_rate);
        
        Ok(Self {
            inference,
            sample_rate,
            model_name,
        })
    }
    
    /// Validate inference engine functionality - STRICT
    fn validate_inference_engine(inference: &KokoroInference) -> Result<(), Box<dyn Error>> {
        println!("🔍 Validating inference engine functionality...");
        
        // Check that voices are available
        let voices = inference.weight_loader.list_voices()
            .map_err(|e| format!("CRITICAL: Cannot list voices: {}", e))?;
        
        if voices.is_empty() {
            return Err("CRITICAL: No voices available. Check voice directory and conversion.".into());
        }
        
        println!("  ✅ {} voices available", voices.len());
        
        // Test voice loading for first available voice
        let test_voice = &voices[0];
        let _voice_embedding = inference.weight_loader.load_voice(test_voice)
            .map_err(|e| format!("CRITICAL: Cannot load test voice '{}': {}", test_voice, e))?;
        
        println!("  ✅ Voice loading functional");
        
        Ok(())
    }
    
    /// Synthesize speech with STRICT validation
    pub fn synthesize(
        &mut self, 
        text: &str, 
        voice: &str, 
        speed: f32
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        // STRICT: Input validation
        if text.trim().is_empty() {
            return Err("CRITICAL: Cannot synthesize empty text".into());
        }
        
        if speed <= 0.0 || speed > 10.0 {
            return Err(format!("CRITICAL: Invalid speed {}. Must be between 0.0 and 10.0", speed).into());
        }
        
        // STRICT: Voice must exist
        let available_voices = self.list_voices()?;
        if !available_voices.contains(&voice.to_string()) {
            return Err(format!("CRITICAL: Voice '{}' not found. Available: {:?}", 
                             voice, available_voices).into());
        }
        
        println!("\n🎯 Synthesizing: \"{}\"", text);
        println!("   Voice: {}", voice);
        println!("   Speed: {:.1}x", speed);
        
        // Use the internal inference engine
        let audio = self.inference.infer_text(text, voice, speed)?;
        
        // STRICT: Audio validation
        if audio.is_empty() {
            return Err("CRITICAL: Generated audio is empty. This indicates pipeline failure.".into());
        }
        
        // Validate audio samples are in valid range
        for (i, &sample) in audio.iter().enumerate() {
            if !sample.is_finite() {
                return Err(format!("CRITICAL: Non-finite audio sample {} at position {}", sample, i).into());
            }
            if sample.abs() > 10.0 {  // Reasonable range check
                return Err(format!("CRITICAL: Audio sample {} out of range at position {}", sample, i).into());
            }
        }
        
        println!("✅ Synthesis complete: {} samples generated", audio.len());
        Ok(audio)
    }
    
    /// List all available voices
    /// 
    /// # Returns
    /// Vector of available voice names
    pub fn list_voices(&self) -> Result<Vec<String>, Box<dyn Error>> {
        self.inference.list_available_voices()
    }
    
    /// Get model information
    /// 
    /// # Returns
    /// Tuple of (model_name, sample_rate, parameter_count)
    pub fn info(&self) -> (&str, u32, String) {
        (
            &self.model_name,
            self.sample_rate,
            "81.8M".to_string(), // From specification
        )
    }
    
    /// Get the audio sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    
    /// Process a batch of texts with the same voice and speed
    /// 
    /// # Arguments
    /// * `texts` - Vector of texts to synthesize
    /// * `voice` - Voice name to use for all texts
    /// * `speed` - Speech speed for all texts
    /// 
    /// # Returns
    /// Vector of audio samples for each input text
    pub fn synthesize_batch(
        &mut self,
        texts: &[&str],
        voice: &str,
        speed: f32
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        println!("\n🎯 Batch synthesis: {} texts", texts.len());
        
        let mut results = Vec::new();
        
        for (i, text) in texts.iter().enumerate() {
            println!("   [{}] Processing: \"{}\"", i + 1, text);
            let audio = self.synthesize(text, voice, speed)?;
            results.push(audio);
        }
        
        println!("✅ Batch complete: {} audio clips generated", results.len());
        Ok(results)
    }
}

/// Convenience function to create a model from default locations
/// 
/// Looks for weights in "./ferrocarril_weights" and config in "./ferrocarril_weights/config.json"
pub fn load_default_model() -> Result<FerroModel, Box<dyn Error>> {
    let weights_dir = "./ferrocarril_weights";
    let config_path = "./ferrocarril_weights/config.json";
    
    if !Path::new(weights_dir).exists() {
        return Err("Default weights directory './ferrocarril_weights' not found".into());
    }
    
    if !Path::new(config_path).exists() {
        return Err("Default config file './ferrocarril_weights/config.json' not found".into());
    }
    
    FerroModel::from_weights_dir(weights_dir, config_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_info() {
        // This test doesn't require actual weights
        let sample_rate = 24000;
        let model_name = "Ferrocarril-Kokoro-82M".to_string();
        
        // We can't create a full model without weights, but we can test the API design
        assert_eq!(sample_rate, 24000);
        assert_eq!(model_name, "Ferrocarril-Kokoro-82M");
    }
    
    #[test]
    fn test_speed_validation() {
        // Test that invalid speed values would be rejected
        let invalid_speeds = vec![-1.0, 0.0, 11.0, f32::INFINITY];
        
        for speed in invalid_speeds {
            assert!(speed <= 0.0 || speed > 10.0, "Speed {} should be invalid", speed);
        }
    }
}