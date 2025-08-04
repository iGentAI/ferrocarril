//! LoadWeightsBinary trait definition for Ferrocarril weight loading

use crate::FerroError;
use crate::weights_binary::BinaryWeightLoader;

/// Trait for loading weights from the binary weight format
/// 
/// This trait should be implemented by all neural network components that need
/// to load weights from the converted PyTorch binary format.
pub trait LoadWeightsBinary {
    /// Load weights from binary format
    /// 
    /// # Arguments
    /// * `loader` - The binary weight loader
    /// * `component` - Component name (e.g., "bert", "text_encoder", "predictor")
    /// * `prefix` - Module prefix within the component (e.g., "module")
    /// 
    /// # Returns
    /// Result indicating success or failure with error details
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError>;
}