//! Core functionality for Ferrocarril

pub mod tensor;
pub mod ops;
#[cfg(feature = "weights")]
pub mod weights;
#[cfg(feature = "weights")]
pub use weights::LoadWeights; // Export LoadWeights trait
#[cfg(feature = "weights")]
pub mod weights_binary;

#[cfg(feature = "weights")]
pub trait LoadWeightsBinary {
    /// Load weights from a BinaryWeightLoader into this component
    fn load_weights_binary(
        &mut self, 
        loader: &weights_binary::BinaryWeightLoader, 
        component: &str,
        prefix: &str
    ) -> Result<(), FerroError>;
}

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use phonesis::{GraphemeToPhoneme, PhonemeStandard, G2POptions, FallbackStrategy, english::EnglishG2P};

#[derive(Debug)]
pub struct FerroError {
    message: String,
}

impl fmt::Display for FerroError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for FerroError {}

impl FerroError {
    pub fn new<T: Into<String>>(message: T) -> Self {
        Self { message: message.into() }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: std::collections::HashMap<char, usize>,
    pub n_token: usize,
    pub hidden_dim: usize,
    pub n_layer: usize,
    pub style_dim: usize,
    pub n_mels: usize,
    pub max_dur: usize,
    pub dropout: f32,
    pub text_encoder_kernel_size: usize,
    pub istftnet: IstftnetConfig,
    pub plbert: PlbertConfig,
}

#[derive(Debug, Clone)]
pub struct IstftnetConfig {
    pub upsample_rates: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub gen_istft_n_fft: usize,
    pub gen_istft_hop_size: usize,
}

#[derive(Debug, Clone)]
pub struct PlbertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
}

impl Config {
    pub fn from_json(path: &str) -> Result<Self, Box<dyn Error>> {
        // Implement actual JSON parsing instead of hardcoded values
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        // Use serde_json to parse the JSON file
        let config: serde_json::Value = serde_json::from_str(&contents)?;
        
        // Extract values from the JSON
        let n_token = config["n_token"].as_u64().unwrap_or(150) as usize;
        let hidden_dim = config["hidden_dim"].as_u64().unwrap_or(512) as usize;
        let n_layer = config["n_layer"].as_u64().unwrap_or(4) as usize;
        let style_dim = config["style_dim"].as_u64().unwrap_or(128) as usize;
        let n_mels = config["n_mels"].as_u64().unwrap_or(80) as usize;
        let max_dur = config["max_dur"].as_u64().unwrap_or(50) as usize;
        let dropout = config["dropout"].as_f64().unwrap_or(0.1) as f32;
        let text_encoder_kernel_size = config["text_encoder_kernel_size"].as_u64().unwrap_or(5) as usize;
        
        // Extract istftnet config
        let istftnet = &config["istftnet"];
        let istftnet_config = IstftnetConfig {
            upsample_rates: istftnet["upsample_rates"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            upsample_initial_channel: istftnet["upsample_initial_channel"].as_u64().unwrap_or(512) as usize,
            resblock_kernel_sizes: istftnet["resblock_kernel_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            resblock_dilation_sizes: istftnet["resblock_dilation_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|arr| {
                    arr.as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|v| v.as_u64().unwrap_or(0) as usize)
                        .collect()
                })
                .collect(),
            upsample_kernel_sizes: istftnet["upsample_kernel_sizes"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            gen_istft_n_fft: istftnet["gen_istft_n_fft"].as_u64().unwrap_or(16) as usize,
            gen_istft_hop_size: istftnet["gen_istft_hop_size"].as_u64().unwrap_or(4) as usize,
        };
        
        // Extract plbert config
        let plbert = &config["plbert"];
        let plbert_config = PlbertConfig {
            hidden_size: plbert["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_attention_heads: plbert["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            num_hidden_layers: plbert["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            intermediate_size: plbert["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        };
        
        // Extract vocab
        let mut vocab = std::collections::HashMap::new();
        if let Some(vocab_obj) = config["vocab"].as_object() {
            for (key, value) in vocab_obj {
                if key.len() == 1 {
                    let ch = key.chars().next().unwrap();
                    let id = value.as_u64().unwrap_or(0) as usize;
                    vocab.insert(ch, id);
                }
            }
        }
        
        Ok(Self {
            vocab,
            n_token,
            hidden_dim,
            n_layer,
            style_dim,
            n_mels,
            max_dur,
            dropout,
            text_encoder_kernel_size,
            istftnet: istftnet_config,
            plbert: plbert_config,
        })
    }
    
    #[cfg(test)]
    pub fn default_for_testing() -> Self {
        // Create a minimal default configuration for testing purposes
        let mut vocab = std::collections::HashMap::new();
        
        // Add some basic tokens
        vocab.insert('!', 1);
        vocab.insert('.', 2);
        vocab.insert(' ', 3);
        
        Self {
            vocab,
            n_token: 178,
            hidden_dim: 512,
            n_layer: 3,
            style_dim: 128,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.2,
            text_encoder_kernel_size: 5,
            istftnet: IstftnetConfig {
                upsample_rates: vec![8, 8],
                upsample_initial_channel: 512,
                resblock_kernel_sizes: vec![3, 7, 11],
                resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
                upsample_kernel_sizes: vec![16, 16],
                gen_istft_n_fft: 16,
                gen_istft_hop_size: 4,
            },
            plbert: PlbertConfig {
                hidden_size: 768,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                intermediate_size: 3072,
            },
        }
    }
}

/// Learnable parameter that can be loaded from weights
#[derive(Debug, Clone)]
pub struct Parameter {
    data: tensor::Tensor<f32>,
}

impl Parameter {
    pub fn new(data: tensor::Tensor<f32>) -> Self {
        Self { data }
    }
    
    pub fn data(&self) -> &tensor::Tensor<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut tensor::Tensor<f32> {
        &mut self.data
    }
}

/// Phonesis-based G2P component for Ferrocarril
pub struct PhonesisG2P {
    inner: Arc<Mutex<dyn GraphemeToPhoneme + Send + Sync>>,
    language: String,
    standard: PhonemeStandard,
}

impl PhonesisG2P {
    /// Create a new G2P component with the specified language
    pub fn new(language: &str) -> Result<Self, FerroError> {
        match language {
            "en" | "en-us" | "en-gb" => {
                let options = G2POptions {
                    handle_stress: true,
                    default_standard: PhonemeStandard::ARPABET,
                    fallback_strategy: FallbackStrategy::UseRules,
                };
                
                let g2p = EnglishG2P::with_options(options)
                    .map_err(|e| FerroError::new(format!("Failed to create G2P: {}", e)))?;
                
                Ok(Self {
                    inner: Arc::new(Mutex::new(g2p)),
                    language: language.to_string(),
                    standard: PhonemeStandard::ARPABET,
                })
            },
            _ => Err(FerroError::new(format!("Unsupported language: {}", language))),
        }
    }
    
    /// Convert text to phonemes
    pub fn convert(&self, text: &str) -> Result<String, FerroError> {
        let inner = self.inner.lock()
            .map_err(|e| FerroError::new(format!("Failed to lock G2P: {}", e)))?;
    
        // First try regular conversion with the default strategy
        match inner.convert_to_standard(text, self.standard) {
            Ok(phonemes) => {
                Ok(phonemes.join(" "))
            },
            Err(e) => {
                // If the default strategy fails (for example, UseRules for unknown words),
                // try again with ReturnGraphemes as a last-resort fallback
                if let phonesis::G2PError::UnknownWord(_) = e {
                    // Create a temporary G2P instance with ReturnGraphemes fallback
                    let options = phonesis::G2POptions {
                        handle_stress: true,
                        default_standard: self.standard,
                        fallback_strategy: phonesis::FallbackStrategy::ReturnGraphemes,
                    };
                    
                    let fallback_g2p = phonesis::english::EnglishG2P::with_options(options)
                        .map_err(|e| FerroError::new(format!("Failed to create fallback G2P: {}", e)))?;
                    
                    match fallback_g2p.convert_to_standard(text, self.standard) {
                        Ok(phonemes) => {
                            println!("Warning: Using grapheme fallback for unknown word(s) in: '{}'", text);
                            Ok(phonemes.join(" "))
                        },
                        Err(e) => {
                            // If even the fallback fails, propagate the error
                            Err(FerroError::new(format!("G2P conversion error: {}", e)))
                        }
                    }
                } else {
                    // For other errors, propagate them
                    Err(FerroError::new(format!("G2P conversion error: {}", e)))
                }
            }
        }
    }
}