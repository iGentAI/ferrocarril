//! Ferrocarril TTS model implementation

// Export the G2P implementation as a submodule
pub mod g2p;

// Export the clean inference implementation
pub mod kokoro_inference;

// Re-export the clean inference for convenience
pub use kokoro_inference::{KokoroInference, run_hello_world_inference};
pub use g2p::G2PHandler;