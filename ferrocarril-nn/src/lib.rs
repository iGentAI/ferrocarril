//! Ferrocarril Neural Network Components
//! 
//! This crate contains specialized neural network implementations
//! optimized for the Kokoro TTS architecture.

// Re-export core traits
pub use ferrocarril_core::{Parameter, Forward, LoadWeightsBinary, FerroError};

// Basic layer implementations (needed by specialized variants)
pub mod linear;
pub mod conv; 
pub mod adain;

// Specialized layer variants (the quality implementations)
pub mod lstm_variants;
pub mod linear_variants; 
pub mod conv1d_variants;
pub mod adain_variants;

// Utility layers
pub mod activation;
pub mod conv_transpose;

// High-level components
pub mod text_encoder;

// BERT implementation 
pub mod bert {
    pub mod custom_albert;
    pub use custom_albert::{CustomAlbert, CustomAlbertConfig};
}

// Vocoder for audio generation
pub mod vocoder;

// Prosody prediction
pub mod prosody;

// Re-export commonly used components
pub use lstm_variants::{TextEncoderLSTM, ProsodyLSTM, DurationEncoderLSTM};
pub use linear_variants::{BERTLinear, ProjectionLinear, EmbeddingLinear, ProjectionType};
pub use conv1d_variants::{TextEncoderConv1d, PredictorConv1d, DecoderConv1d};
pub use adain_variants::{DecoderAdaIN, GeneratorAdaIN, DurationAdaIN};
pub use text_encoder::TextEncoder;
pub use bert::CustomAlbert;
pub use vocoder::{Generator, Decoder}; 
pub use activation::{relu, leaky_relu, sigmoid, tanh, snake};

// Re-export basic layers for specialized variants
pub use linear::Linear;
pub use conv::Conv1d;
pub use adain::{AdaIN1d, InstanceNorm1d};