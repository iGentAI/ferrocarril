//! Ferrocarril TTS model implementation

// Export the G2P implementation as a submodule
pub mod g2p;

// Re-export the G2P handler for convenience
pub use g2p::G2PHandler;

// Export FerroModel
mod ferro_model;
pub use ferro_model::FerroModel;