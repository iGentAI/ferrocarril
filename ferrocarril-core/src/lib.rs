//! Core functionality for Ferrocarril

pub mod tensor;
pub mod ops;  
pub mod g2p;
#[cfg(feature = "weights")]
pub mod weights_binary;
#[cfg(feature = "weights")]
pub mod weights_binary_trait;

use std::error::Error;
use std::fmt;
use tensor::Tensor;

// Re-export the PhonesisG2P wrapper
pub use g2p::PhonesisG2P;

#[cfg(feature = "weights")]
pub use weights_binary_trait::LoadWeightsBinary;

// Add Forward trait that neural network components need
pub trait Forward {
    type Output;
    fn forward(&self, input: &Tensor<f32>) -> Self::Output;
}

// Add basic weights module for compatibility
pub mod weights {
    use super::*;
    
    pub struct PyTorchWeightLoader;
    
    pub trait LoadWeights {
        fn load_weights(
            &mut self,
            loader: &PyTorchWeightLoader,
            prefix: Option<&str>,
        ) -> Result<(), FerroError>;
    }
}

#[derive(Debug)]
pub struct FerroError {
    pub message: String,
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

/// Learnable parameter that can be loaded from weights
#[derive(Debug, Clone)]
pub struct Parameter {
    data: Tensor<f32>,
}

impl Parameter {
    pub fn new(data: Tensor<f32>) -> Self {
        Self { data }
    }
    
    pub fn data(&self) -> &Tensor<f32> {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut Tensor<f32> {
        &mut self.data
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: std::collections::HashMap<String, usize>,
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
    /// Create a default configuration for testing purposes
    pub fn default_for_testing() -> Self {
        // Create a simple test vocabulary
        let mut vocab = std::collections::HashMap::new();
        // Add basic characters and phonemes
        for (i, c) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?-".chars().enumerate() {
            vocab.insert(c.to_string(), i);
        }
        
        // Default istftnet configuration
        let istftnet = IstftnetConfig {
            upsample_rates: vec![8, 8, 2, 2],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            gen_istft_n_fft: 16,
            gen_istft_hop_size: 4,
        };
        
        // Default PLBERT configuration
        let plbert = PlbertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
        };
        
        Self {
            vocab,
            n_token: 150,
            hidden_dim: 512,
            n_layer: 4,
            style_dim: 128,
            n_mels: 80,
            max_dur: 50,
            dropout: 0.1,
            text_encoder_kernel_size: 5,
            istftnet,
            plbert,
        }
    }
    
    /// Load configuration from a KModel config.json file - STRICT VERSION
    pub fn from_json(path: &str) -> Result<Self, Box<dyn Error>> {
        use std::fs::File;
        use std::io::Read;
        
        // STRICT: File must exist
        if !std::path::Path::new(path).exists() {
            return Err(format!("CRITICAL: Config file not found: {}", path).into());
        }
        
        let mut file = File::open(path)
            .map_err(|e| format!("CRITICAL: Cannot open config file '{}': {}", path, e))?;
        
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("CRITICAL: Cannot read config file '{}': {}", path, e))?;
        
        if contents.trim().is_empty() {
            return Err(format!("CRITICAL: Config file '{}' is empty", path).into());
        }
        
        let config: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| format!("CRITICAL: Invalid JSON in config file '{}': {}", path, e))?;
        
        // STRICT: Extract required values with validation
        let n_token = config["n_token"].as_u64()
            .ok_or_else(|| format!("CRITICAL: Missing or invalid 'n_token' in config"))? as usize;
        let hidden_dim = config["hidden_dim"].as_u64()
            .ok_or_else(|| format!("CRITICAL: Missing or invalid 'hidden_dim' in config"))? as usize;
        let style_dim = config["style_dim"].as_u64()
            .ok_or_else(|| format!("CRITICAL: Missing or invalid 'style_dim' in config"))? as usize;
        
        // STRICT: Validate reasonable ranges
        if n_token == 0 || n_token > 10000 {
            return Err(format!("CRITICAL: Invalid n_token value: {}", n_token).into());
        }
        if hidden_dim == 0 || hidden_dim > 10000 {
            return Err(format!("CRITICAL: Invalid hidden_dim value: {}", hidden_dim).into());
        }
        if style_dim == 0 || style_dim > 1000 {
            return Err(format!("CRITICAL: Invalid style_dim value: {}", style_dim).into());
        }
        
        // Extract other required values
        let n_layer = config["n_layer"].as_u64().unwrap_or(3) as usize;
        let n_mels = config["n_mels"].as_u64().unwrap_or(80) as usize;
        let max_dur = config["max_dur"].as_u64().unwrap_or(50) as usize;
        let dropout = config["dropout"].as_f64().unwrap_or(0.1) as f32;
        let text_encoder_kernel_size = config["text_encoder_kernel_size"].as_u64().unwrap_or(5) as usize;
        
        // STRICT: Extract vocab with proper string key handling
        let mut vocab = std::collections::HashMap::new();
        if let Some(vocab_obj) = config["vocab"].as_object() {
            for (key, value) in vocab_obj {
                let id = value.as_u64()
                    .ok_or_else(|| format!("CRITICAL: Invalid vocab ID for key '{}': {}", key, value))? as usize;
                vocab.insert(key.clone(), id);
            }
        } else {
            return Err("CRITICAL: Missing or invalid 'vocab' in config".into());
        }
        
        if vocab.is_empty() {
            return Err("CRITICAL: Empty vocabulary in config".into());
        }
        
        // STRICT: Extract required nested configurations
        let istftnet = config["istftnet"].as_object()
            .ok_or_else(|| "CRITICAL: Missing 'istftnet' config")?;
        let plbert = config["plbert"].as_object()
            .ok_or_else(|| "CRITICAL: Missing 'plbert' config")?;
        
        // Extract istftnet config with validation
        let upsample_rates: Vec<usize> = istftnet["upsample_rates"]
            .as_array()
            .ok_or_else(|| "CRITICAL: Missing 'upsample_rates' in istftnet config")?
            .iter()
            .map(|v| v.as_u64()
                .ok_or_else(|| format!("CRITICAL: Invalid upsample rate: {}", v))
                .map(|x| x as usize))
            .collect::<Result<Vec<_>, _>>()?;
        
        if upsample_rates.is_empty() {
            return Err("CRITICAL: Empty upsample_rates in istftnet config".into());
        }
        
        let istftnet_config = IstftnetConfig {
            upsample_rates,
            upsample_initial_channel: istftnet["upsample_initial_channel"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'upsample_initial_channel'")? as usize,
            resblock_kernel_sizes: istftnet["resblock_kernel_sizes"]
                .as_array()
                .ok_or_else(|| "CRITICAL: Missing 'resblock_kernel_sizes'")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            resblock_dilation_sizes: istftnet["resblock_dilation_sizes"]
                .as_array()
                .ok_or_else(|| "CRITICAL: Missing 'resblock_dilation_sizes'")?
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
                .ok_or_else(|| "CRITICAL: Missing 'upsample_kernel_sizes'")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect(),
            gen_istft_n_fft: istftnet["gen_istft_n_fft"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'gen_istft_n_fft'")? as usize,
            gen_istft_hop_size: istftnet["gen_istft_hop_size"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'gen_istft_hop_size'")? as usize,
        };
        
        // Extract plbert config with validation
        let plbert_config = PlbertConfig {
            hidden_size: plbert["hidden_size"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'hidden_size' in plbert config")? as usize,
            num_attention_heads: plbert["num_attention_heads"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'num_attention_heads' in plbert config")? as usize,
            num_hidden_layers: plbert["num_hidden_layers"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'num_hidden_layers' in plbert config")? as usize,
            intermediate_size: plbert["intermediate_size"].as_u64()
                .ok_or_else(|| "CRITICAL: Missing 'intermediate_size' in plbert config")? as usize,
        };
        
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

    /// Real Kokoro-82M configuration - VENDORED for production
    /// NO external file dependencies - authentic neural configuration only
    pub fn kokoro_82m_config() -> Self {        
        // Complete Real Kokoro vocabulary - AUTHENTIC NEURAL MAPPING
        let mut vocab = std::collections::HashMap::new();
        vocab.insert(";".to_string(), 1);
        vocab.insert(":".to_string(), 2);
        vocab.insert(",".to_string(), 3);
        vocab.insert(".".to_string(), 4);
        vocab.insert("!".to_string(), 5);
        vocab.insert("?".to_string(), 6);
        vocab.insert("—".to_string(), 9);
        vocab.insert("…".to_string(), 10);
        vocab.insert("\"".to_string(), 11);
        vocab.insert("(".to_string(), 12);
        vocab.insert(")".to_string(), 13);
        vocab.insert(" ".to_string(), 16);
        vocab.insert("ʣ".to_string(), 18);
        vocab.insert("ʥ".to_string(), 19);
        vocab.insert("ʦ".to_string(), 20);
        vocab.insert("ʨ".to_string(), 21);
        vocab.insert("ᵝ".to_string(), 22);
        vocab.insert("A".to_string(), 24);
        vocab.insert("I".to_string(), 25);
        vocab.insert("O".to_string(), 31);
        vocab.insert("Q".to_string(), 33);
        vocab.insert("S".to_string(), 35);
        vocab.insert("T".to_string(), 36);
        vocab.insert("W".to_string(), 39);
        vocab.insert("Y".to_string(), 41);
        vocab.insert("ᵊ".to_string(), 42);
        vocab.insert("a".to_string(), 43);
        vocab.insert("b".to_string(), 44);
        vocab.insert("c".to_string(), 45);
        vocab.insert("d".to_string(), 46);
        vocab.insert("e".to_string(), 47);
        vocab.insert("f".to_string(), 48);
        vocab.insert("h".to_string(), 50);
        vocab.insert("i".to_string(), 51);
        vocab.insert("j".to_string(), 52);
        vocab.insert("k".to_string(), 53);
        vocab.insert("l".to_string(), 54);
        vocab.insert("m".to_string(), 55);
        vocab.insert("n".to_string(), 56);
        vocab.insert("o".to_string(), 57);
        vocab.insert("p".to_string(), 58);
        vocab.insert("q".to_string(), 59);
        vocab.insert("r".to_string(), 60);
        vocab.insert("s".to_string(), 61);
        vocab.insert("t".to_string(), 62);
        vocab.insert("u".to_string(), 63);
        vocab.insert("v".to_string(), 64);
        vocab.insert("w".to_string(), 65);
        vocab.insert("x".to_string(), 66);
        vocab.insert("y".to_string(), 67);
        vocab.insert("z".to_string(), 68);
        vocab.insert("ɑ".to_string(), 69);
        vocab.insert("ɐ".to_string(), 70);
        vocab.insert("ɒ".to_string(), 71);
        vocab.insert("æ".to_string(), 72);
        vocab.insert("β".to_string(), 75);
        vocab.insert("ɔ".to_string(), 76);
        vocab.insert("ɕ".to_string(), 77);
        vocab.insert("ç".to_string(), 78);
        vocab.insert("ɖ".to_string(), 80);
        vocab.insert("ð".to_string(), 81);
        vocab.insert("ʤ".to_string(), 82);
        vocab.insert("ə".to_string(), 83);
        vocab.insert("ɚ".to_string(), 85);
        vocab.insert("ɛ".to_string(), 86);
        vocab.insert("ɜ".to_string(), 87);
        vocab.insert("ɟ".to_string(), 90);
        vocab.insert("ɡ".to_string(), 92);
        vocab.insert("ɥ".to_string(), 99);
        vocab.insert("ɨ".to_string(), 101);
        vocab.insert("ɪ".to_string(), 102);
        vocab.insert("ʝ".to_string(), 103);
        vocab.insert("ɯ".to_string(), 110);
        vocab.insert("ɰ".to_string(), 111);
        vocab.insert("ŋ".to_string(), 112);
        vocab.insert("ɳ".to_string(), 113);
        vocab.insert("ɲ".to_string(), 114);
        vocab.insert("ɴ".to_string(), 115);
        vocab.insert("ø".to_string(), 116);
        vocab.insert("ɸ".to_string(), 118);
        vocab.insert("θ".to_string(), 119);
        vocab.insert("œ".to_string(), 120);
        vocab.insert("ɹ".to_string(), 123);
        vocab.insert("ɾ".to_string(), 125);
        vocab.insert("ɻ".to_string(), 126);
        vocab.insert("ʁ".to_string(), 128);
        vocab.insert("ɽ".to_string(), 129);
        vocab.insert("ʂ".to_string(), 130);
        vocab.insert("ʃ".to_string(), 131);
        vocab.insert("ʈ".to_string(), 132);
        vocab.insert("ʧ".to_string(), 133);
        vocab.insert("ʊ".to_string(), 135);
        vocab.insert("ʋ".to_string(), 136);
        vocab.insert("ʌ".to_string(), 138);
        vocab.insert("ɣ".to_string(), 139);
        vocab.insert("ɤ".to_string(), 140);
        vocab.insert("χ".to_string(), 142);
        vocab.insert("ʎ".to_string(), 143);
        vocab.insert("ʒ".to_string(), 147);
        vocab.insert("ʔ".to_string(), 148);
        vocab.insert("ˈ".to_string(), 156);
        vocab.insert("ˌ".to_string(), 157);
        vocab.insert("ː".to_string(), 158);
        vocab.insert("ʰ".to_string(), 162);
        vocab.insert("ʲ".to_string(), 164);
        vocab.insert("↓".to_string(), 169);
        vocab.insert("→".to_string(), 171);
        vocab.insert("↗".to_string(), 172);
        vocab.insert("↘".to_string(), 173);
        vocab.insert("ᵻ".to_string(), 177);
        
        // Real istftnet configuration from authentic Kokoro-82M
        let istftnet = IstftnetConfig {
            upsample_rates: vec![10, 6],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_kernel_sizes: vec![20, 12],
            gen_istft_n_fft: 20,
            gen_istft_hop_size: 5,
        };
        
        // Real PLBERT configuration
        let plbert = PlbertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 2048,
        };
        
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
            istftnet,
            plbert,
        }
    }
}
