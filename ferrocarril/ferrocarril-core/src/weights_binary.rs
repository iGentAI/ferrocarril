//! Binary weight loading functionality for Ferrocarril.
//! 
//! This module provides a BinaryWeightLoader that can load PyTorch weights
//! converted to a simpler binary format using the weight_converter.py script.

use crate::tensor::Tensor;
use crate::{Parameter, FerroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Metadata for a single tensor in the binary format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorMetadata {
    file: String,
    shape: Vec<usize>,
    dtype: String,
    byte_size: usize,
}

/// Metadata for a single component (e.g., bert, decoder)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComponentMetadata {
    parameters: HashMap<String, TensorMetadata>,
}

/// Complete metadata for a converted model
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    format_version: String,
    original_file: String,
    components: HashMap<String, ComponentMetadata>,
}

/// Metadata for a single voice
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoiceMetadata {
    name: String,
    file: String,
    shape: Vec<usize>,
    dtype: String,
    byte_size: usize,
}

/// Complete metadata for all voices
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoicesMetadata {
    format_version: String,
    voices: HashMap<String, VoiceMetadata>,
}

/// Loader for converted binary weights - handles both model weights and voices
pub struct BinaryWeightLoader {
    base_path: PathBuf,
    model_metadata: ModelMetadata,
    voices_metadata: Option<VoicesMetadata>,
}

impl BinaryWeightLoader {
    /// Load a model from a converted directory
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self, FerroError> {
        let base_path = path.as_ref().to_path_buf();
        let metadata_path = base_path.join("model").join("metadata.json");
        
        // Load model metadata
        let mut metadata_file = File::open(&metadata_path)
            .map_err(|e| FerroError::new(format!("Failed to open metadata file: {}", e)))?;
        
        let mut metadata_content = String::new();
        metadata_file.read_to_string(&mut metadata_content)
            .map_err(|e| FerroError::new(format!("Failed to read metadata file: {}", e)))?;
        
        let model_metadata: ModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| FerroError::new(format!("Failed to parse metadata: {}", e)))?;
        
        // Try to load voices metadata if it exists
        let voices_metadata_path = base_path.join("voices").join("voices.json");
        let voices_metadata = if voices_metadata_path.exists() {
            let mut voices_file = File::open(&voices_metadata_path)
                .map_err(|e| FerroError::new(format!("Failed to open voices metadata: {}", e)))?;
            
            let mut voices_content = String::new();
            voices_file.read_to_string(&mut voices_content)
                .map_err(|e| FerroError::new(format!("Failed to read voices metadata: {}", e)))?;
            
            let metadata: VoicesMetadata = serde_json::from_str(&voices_content)
                .map_err(|e| FerroError::new(format!("Failed to parse voices metadata: {}", e)))?;
            
            Some(metadata)
        } else {
            None
        };
        
        Ok(Self {
            base_path,
            model_metadata,
            voices_metadata,
        })
    }
    
    /// Load a component and parameter by name
    pub fn load_component_parameter(&self, component: &str, param: &str) -> Result<Tensor<f32>, FerroError> {
        // Find the component
        let component_meta = self.model_metadata.components.get(component)
            .ok_or_else(|| FerroError::new(format!("Component '{}' not found", component)))?;
        
        // Find the parameter
        let tensor_meta = component_meta.parameters.get(param)
            .ok_or_else(|| FerroError::new(format!("Parameter '{}' not found in component '{}'", param, component)))?;
        
        // Get file path
        let file_path = self.base_path.join("model").join(&tensor_meta.file);
        
        // Support for f32 tensors
        if tensor_meta.dtype == "float32" || tensor_meta.dtype == "torch.float32" || tensor_meta.dtype == "dtype('float32')" {
            // Calculate number of elements
            let num_elements: usize = tensor_meta.shape.iter().product();
            let expected_bytes = num_elements * 4; // 4 bytes per f32
            
            if tensor_meta.byte_size != expected_bytes {
                return Err(FerroError::new(format!(
                    "Byte size mismatch: metadata says {} but calculated {} for tensor {}",
                    tensor_meta.byte_size, expected_bytes, param
                )));
            }
            
            // Read binary data from file
            let mut file = File::open(&file_path)
                .map_err(|e| FerroError::new(format!("Failed to open file '{}': {}", file_path.display(), e)))?;
            
            let mut bytes = vec![0u8; expected_bytes];
            file.read_exact(&mut bytes)
                .map_err(|e| FerroError::new(format!("Failed to read tensor data from '{}': {}", file_path.display(), e)))?;
            
            // Convert to f32
            let mut data_vec = Vec::with_capacity(num_elements);
            for i in 0..num_elements {
                let start = i * 4;
                let end = start + 4;
                let float_bytes = &bytes[start..end];
                let value = f32::from_le_bytes([float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]]);
                data_vec.push(value);
            }
            
            Ok(Tensor::from_data(data_vec, tensor_meta.shape.clone()))
        } else {
            Err(FerroError::new(format!("Unsupported tensor dtype: {}", tensor_meta.dtype)))
        }
    }
    
    /// Load a tensor by name
    pub fn load_tensor(&self, name: &str) -> Result<Tensor<f32>, FerroError> {
        // Split the name into component and parameter parts if it contains a dot
        let parts: Vec<&str> = name.split('.').collect();
        
        if parts.len() > 1 {
            // Handle component.parameter format
            let component = parts[0];
            let param = &name[component.len() + 1..]; // +1 for the dot
            self.load_component_parameter(component, param)
        } else {
            // Try to find the parameter in any component
            for component_name in self.list_components() {
                if let Ok(parameters) = self.list_parameters(&component_name) {
                    if parameters.contains(&name.to_string()) {
                        return self.load_component_parameter(&component_name, name);
                    }
                }
            }
            
            Err(FerroError::new(format!("Tensor '{}' not found in any component", name)))
        }
    }
    
    /// Load a voice by name
    pub fn load_voice(&self, voice_name: &str) -> Result<Tensor<f32>, FerroError> {
        // Make sure we have voices metadata
        let voices_meta = self.voices_metadata.as_ref()
            .ok_or_else(|| FerroError::new("No voices metadata available"))?;
        
        // Find the voice
        let voice_meta = voices_meta.voices.get(voice_name)
            .ok_or_else(|| FerroError::new(format!("Voice '{}' not found", voice_name)))?;
        
        // Get file path
        let file_path = self.base_path.join(&voice_meta.file);
        
        // Calculate number of elements
        let num_elements: usize = voice_meta.shape.iter().product();
        let expected_bytes = num_elements * 4; // 4 bytes per f32
        
        // Read binary data from file
        let mut file = File::open(&file_path)
            .map_err(|e| FerroError::new(format!("Failed to open file '{}': {}", file_path.display(), e)))?;
        
        let mut bytes = vec![0u8; expected_bytes];
        file.read_exact(&mut bytes)
            .map_err(|e| FerroError::new(format!("Failed to read voice data from '{}': {}", file_path.display(), e)))?;
        
        // Convert to f32
        let mut data_vec = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let end = start + 4;
            let float_bytes = &bytes[start..end];
            let value = f32::from_le_bytes([float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]]);
            data_vec.push(value);
        }
        
        Ok(Tensor::from_data(data_vec, voice_meta.shape.clone()))
    }
    
    // Helper functions
    
    /// List all available components
    pub fn list_components(&self) -> Vec<String> {
        self.model_metadata.components.keys().cloned().collect()
    }
    
    /// List all parameters in a component
    pub fn list_parameters(&self, component: &str) -> Result<Vec<String>, FerroError> {
        let component_meta = self.model_metadata.components.get(component)
            .ok_or_else(|| FerroError::new(format!("Component '{}' not found", component)))?;
        
        Ok(component_meta.parameters.keys().cloned().collect())
    }
    
    /// List all available voices
    pub fn list_voices(&self) -> Result<Vec<String>, FerroError> {
        let voices_meta = self.voices_metadata.as_ref()
            .ok_or_else(|| FerroError::new("No voices metadata available"))?;
        
        Ok(voices_meta.voices.keys().cloned().collect())
    }
    
    /// Check if the loader is empty (no tensors)
    pub fn is_empty(&self) -> bool {
        self.model_metadata.components.is_empty()
    }
    
    /// Load a parameter for a component by name with the LoadWeights trait interface
    pub fn load_weight_into_parameter(
        &self,
        param: &mut Parameter,
        component: &str,
        name: &str,
    ) -> Result<(), FerroError> {
        let tensor = self.load_component_parameter(component, name)?;
        *param = Parameter::new(tensor);
        Ok(())
    }
}

/// Implementation of the LoadWeights trait for BinaryWeightLoader
#[cfg(feature = "weights")]
impl crate::weights::LoadWeights for BinaryWeightLoader {
    fn load_weights(
        &mut self,
        _loader: &crate::weights::PyTorchWeightLoader,
        prefix: Option<&str>,
    ) -> Result<(), FerroError> {
        // Convert prefix to component name
        let _component = prefix.unwrap_or("model");
        
        // This is a stub implementation - actual implementation would
        // load weights into components based on the component type
        Ok(())
    }
}