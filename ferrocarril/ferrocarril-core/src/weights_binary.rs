//! Binary weight loading functionality for Ferrocarril.
//! 
//! This module provides a BinaryWeightLoader that can load PyTorch weights
//! converted to a simpler binary format using the weight_converter.py script.

use crate::tensor::Tensor;
use crate::{Parameter, FerroError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
    /// Original base path provided by the user
    original_base_path: PathBuf,
    /// Base path for model weights (directory containing metadata.json)
    model_base_path: PathBuf,
    /// Base path for voice weights (typically original_base_path/voices)
    voices_base_path: Option<PathBuf>,
    /// Model metadata
    model_metadata: ModelMetadata,
    /// Voices metadata
    voices_metadata: Option<VoicesMetadata>,
}

impl BinaryWeightLoader {
    /// Load a model from a converted directory
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self, FerroError> {
        let original_base_path = path.as_ref().to_path_buf();
        
        println!("Initializing BinaryWeightLoader from: {}", original_base_path.display());
        
        // Try different potential locations for the metadata.json file
        let possible_metadata_paths = vec![
            original_base_path.join("metadata.json"),               // Direct in base directory
            original_base_path.clone(),                            // Path is the metadata file itself
            original_base_path.join("model").join("metadata.json"), // In model subdirectory
        ];
        
        // Try each path until we find a valid metadata file
        let mut metadata_path = None;
        for path in &possible_metadata_paths {
            if path.exists() && path.is_file() {
                println!("Found metadata.json at: {}", path.display());
                metadata_path = Some(path.clone());
                break;
            }
        }
        
        // If still not found, try searching for it
        if metadata_path.is_none() {
            println!("Metadata file not found at standard locations. Searching recursively...");
            if let Ok(entries) = std::fs::read_dir(&original_base_path) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                            let subdir_path = entry.path().join("metadata.json");
                            if subdir_path.exists() && subdir_path.is_file() {
                                metadata_path = Some(subdir_path);
                                println!("Found metadata.json at: {}", subdir_path.display());
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // If still not found, return an error
        let metadata_path = metadata_path.ok_or_else(|| 
            FerroError::new(format!("Metadata file not found in or under directory: {}", original_base_path.display()))
        )?;
        
        // Set model_base_path to the directory containing metadata.json
        let model_base_path = metadata_path.parent().unwrap_or(&original_base_path).to_path_buf();
        println!("Using model base path: {}", model_base_path.display());
        
        // Load model metadata
        let mut metadata_file = File::open(&metadata_path)
            .map_err(|e| FerroError::new(format!("Failed to open metadata file: {}", e)))?;
        
        let mut metadata_content = String::new();
        metadata_file.read_to_string(&mut metadata_content)
            .map_err(|e| FerroError::new(format!("Failed to read metadata file: {}", e)))?;
        
        let model_metadata: ModelMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| FerroError::new(format!("Failed to parse metadata: {}", e)))?;
        
        // Determine the voices base path - this should be at the root level
        let voices_base_path = original_base_path.join("voices");
        let voices_metadata_path = voices_base_path.join("voices.json");
        
        // Try to load voices metadata if it exists
        let voices_metadata = if voices_metadata_path.exists() {
            println!("Loading voices metadata from: {}", voices_metadata_path.display());
            let mut voices_file = File::open(&voices_metadata_path)
                .map_err(|e| FerroError::new(format!("Failed to open voices metadata: {}", e)))?;
            
            let mut voices_content = String::new();
            voices_file.read_to_string(&mut voices_content)
                .map_err(|e| FerroError::new(format!("Failed to read voices metadata: {}", e)))?;
            
            let metadata: VoicesMetadata = serde_json::from_str(&voices_content)
                .map_err(|e| FerroError::new(format!("Failed to parse voices metadata: {}", e)))?;
            
            Some(metadata)
        } else {
            println!("No voices metadata found at: {}", voices_metadata_path.display());
            None
        };
        
        Ok(Self {
            original_base_path: original_base_path.clone(),
            model_base_path,
            voices_base_path: if voices_metadata_path.exists() { Some(voices_base_path) } else { None },
            model_metadata,
            voices_metadata,
        })
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

    /// Load a component and parameter by name
    pub fn load_component_parameter(&self, component: &str, param: &str) -> Result<Tensor<f32>, FerroError> {
        // Find the component
        let component_meta = self.model_metadata.components.get(component)
            .ok_or_else(|| FerroError::new(format!("Component '{}' not found", component)))?;
        
        // Find the parameter
        let tensor_meta = component_meta.parameters.get(param)
            .ok_or_else(|| FerroError::new(format!("Parameter '{}' not found in component '{}'", param, component)))?;
        
        // Get file path from metadata
        let file_path = self.model_base_path.join(&tensor_meta.file);
        
        // Check if file exists
        if !file_path.exists() {
            // Try alternative paths
            let alt_paths = vec![
                // Try direct component/param path
                self.model_base_path.join(component).join(format!("{}.bin", param.replace(".", "_"))),
                // Try from original base path
                self.original_base_path.join(&tensor_meta.file),
                // Try in model subdirectory of original path
                self.original_base_path.join("model").join(&tensor_meta.file),
            ];
            
            for alt_path in &alt_paths {
                if alt_path.exists() {
                    println!("Using alternative path: {}", alt_path.display());
                    return self.load_tensor_from_path(alt_path, &tensor_meta.shape);
                }
            }
            
            return Err(FerroError::new(format!(
                "File not found at '{}' or any alternative locations", 
                file_path.display()
            )));
        }
        
        self.load_tensor_from_path(&file_path, &tensor_meta.shape)
    }
    
    /// Helper method to load tensor from a specific path
    fn load_tensor_from_path(&self, file_path: &Path, shape: &Vec<usize>) -> Result<Tensor<f32>, FerroError> {
        // Read binary data from file
        let mut file = File::open(file_path)
            .map_err(|e| FerroError::new(format!("Failed to open file '{}': {}", file_path.display(), e)))?;
        
        // Calculate number of elements
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * 4; // 4 bytes per f32
        
        // Get actual file size
        let file_size = std::fs::metadata(file_path)
            .map_err(|e| FerroError::new(format!("Failed to get file size for '{}': {}", file_path.display(), e)))?
            .len() as usize;
        
        if expected_bytes != file_size {
            println!("Warning: Expected {} bytes based on shape {:?}, but file is {} bytes. Adjusting.",
                expected_bytes, shape, file_size);
        }
        
        // Use the smaller of expected and actual size to avoid reading past EOF
        let tensor_bytes = std::cmp::min(file_size, expected_bytes);
        let actual_elements = tensor_bytes / 4;
        
        let mut bytes = vec![0u8; tensor_bytes];
        file.read_exact(&mut bytes)
            .map_err(|e| FerroError::new(format!("Failed to read tensor data from '{}': {}", file_path.display(), e)))?;
        
        // Convert to f32
        let mut data_vec = Vec::with_capacity(actual_elements);
        for i in 0..actual_elements {
            let start = i * 4;
            let end = start + 4;
            let float_bytes = &bytes[start..end];
            let value = f32::from_le_bytes([float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]]);
            data_vec.push(value);
        }
        
        // Adjust shape if needed
        let mut adjusted_shape = shape.clone();
        if shape.iter().product::<usize>() > actual_elements {
            println!("Warning: Shape {:?} requires {} elements but only {} available. Adjusting shape.", 
                shape, shape.iter().product::<usize>(), actual_elements);
            
            // Try to infer a reasonable shape
            if shape.len() == 2 {
                // For 2D tensors, keep the first dimension and adjust the second
                let first_dim = shape[0];
                let second_dim = actual_elements / first_dim;
                adjusted_shape = vec![first_dim, second_dim];
                println!("Adjusted shape to: {:?}", adjusted_shape);
            }
        }
        
        Ok(Tensor::from_data(data_vec, adjusted_shape))
    }
    
    /// Load a voice by name
    pub fn load_voice(&self, voice_name: &str) -> Result<Tensor<f32>, FerroError> {
        // Make sure we have voices metadata
        let voices_meta = self.voices_metadata.as_ref()
            .ok_or_else(|| FerroError::new("No voices metadata available"))?;
        
        // Find the voice
        let voice_meta = voices_meta.voices.get(voice_name)
            .ok_or_else(|| FerroError::new(format!("Voice '{}' not found", voice_name)))?;
        
        // Get voices base path, defaulting to original/voices if not set
        let voices_path = match &self.voices_base_path {
            Some(path) => path.clone(),
            None => self.original_base_path.join("voices"),
        };
        
        // Get file path
        let file_path = voices_path.join(&voice_meta.file);
        
        // Check if file exists
        if !file_path.exists() {
            // Try alternative paths
            let alt_paths = vec![
                // Try with filename directly
                voices_path.join(format!("{}.bin", voice_name)),
                // Try from original base path
                self.original_base_path.join("voices").join(&voice_meta.file),
            ];
            
            for alt_path in &alt_paths {
                if alt_path.exists() {
                    println!("Using alternative voice path: {}", alt_path.display());
                    return self.load_tensor_from_path(alt_path, &voice_meta.shape);
                }
            }
            
            return Err(FerroError::new(format!("Voice file not found at '{}' or any alternative locations", file_path.display())));
        }
        
        self.load_tensor_from_path(&file_path, &voice_meta.shape)
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
impl crate::weights::LoadWeights for BinaryWeightLoader {
    fn load_weights(
        &mut self,  // Keep &mut self to match the trait definition
        _loader: &crate::weights::PyTorchWeightLoader,
        prefix: Option<&str>
    ) -> Result<(), FerroError> {
        // Convert prefix to component name
        let _component = prefix.unwrap_or("model");
        
        // This is a stub implementation - actual implementation would
        // load weights into components based on the component type
        Ok(())
    }
}