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
                                println!("Found metadata.json at: {}", subdir_path.display());
                                metadata_path = Some(subdir_path);
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
    
    
    /// Load a tensor by name - FAIL-FAST VERSION
    pub fn load_tensor(&self, name: &str) -> Result<Tensor<f32>, FerroError> {
        // Split the name into component and parameter parts if it contains a dot
        let parts: Vec<&str> = name.split('.').collect();
        
        if parts.len() > 1 {
            // Handle component.parameter format
            let component = parts[0];
            let param = &name[component.len() + 1..]; // +1 for the dot
            self.load_component_parameter(component, param)
        } else {
            // STRICT: No silent searching - parameter must be explicit
            return Err(FerroError::new(format!(
                "CRITICAL: Ambiguous parameter name '{}'. Must specify component.parameter format. \
                Available components: {:?}", 
                name, self.list_components()
            )));
        }
    }

    /// Load a component and parameter by name with CRITICAL validation - NO SILENT FAILURES
    pub fn load_component_parameter(&self, component: &str, param_name: &str) -> Result<Tensor<f32>, FerroError> {
        // STRICT: Component must exist
        let component_meta = self.model_metadata.components.get(component)
            .ok_or_else(|| FerroError::new(format!(
                "CRITICAL: Component '{}' not found. Available components: {:?}", 
                component, self.list_components()
            )))?;
        
        // STRICT: Parameter must exist in component
        let tensor_meta = component_meta.parameters.get(param_name)
            .ok_or_else(|| {
                // Provide detailed error with available parameters for debugging
                let available_params: Vec<String> = component_meta.parameters.keys().map(|k| format!("\"{}\"", k)).collect();
                FerroError::new(format!(
                    "CRITICAL: Parameter '{}' not found in component '{}'. Available parameters: [{}]",
                    param_name, component, available_params.join(", ")
                ))
            })?;
        
        // CRITICAL: Validate tensor metadata integrity
        if tensor_meta.shape.is_empty() {
            return Err(FerroError::new(format!(
                "CRITICAL: Parameter '{}' in component '{}' has empty shape", 
                param_name, component
            )));
        }
        
        // STRICT: Validate tensor shape is reasonable
        let expected_elements: usize = tensor_meta.shape.iter().product();
        if expected_elements == 0 {
            return Err(FerroError::new(format!(
                "CRITICAL: Parameter '{}' in component '{}' has zero-sized shape {:?}", 
                param_name, component, tensor_meta.shape
            )));
        }
        
        // Get file path from metadata
        let file_path = self.model_base_path.join(&tensor_meta.file);
        
        // STRICT: File must exist - NO ALTERNATIVE PATHS
        if !file_path.exists() {
            return Err(FerroError::new(format!(
                "CRITICAL: Weight file not found: '{}'. \
                This indicates incomplete weight conversion or corrupted weight directory. \
                NO FALLBACKS PERMITTED.", 
                file_path.display()
            )));
        }
        
        // Load tensor with comprehensive validation
        let tensor = self.load_tensor_from_path_with_validation(&file_path, &tensor_meta.shape, component, param_name)?;
        
        println!("✅ Parameter '{}' loaded and validated: shape {:?}, {} elements", 
                 param_name, tensor_meta.shape, tensor.data().len());
        
        Ok(tensor)
    }
    
    /// Helper method to load tensor from a specific path with comprehensive validation
    fn load_tensor_from_path_with_validation(&self, file_path: &Path, shape: &Vec<usize>, component: &str, param_name: &str) -> Result<Tensor<f32>, FerroError> {
        // Read binary data from file
        let mut file = File::open(file_path)
            .map_err(|e| FerroError::new(format!(
                "CRITICAL: Cannot open weight file '{}': {}", 
                file_path.display(), e
            )))?;
        
        // Calculate expected size
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * 4; // 4 bytes per f32
        
        // Get actual file size
        let file_size = std::fs::metadata(file_path)
            .map_err(|e| FerroError::new(format!(
                "CRITICAL: Cannot read metadata for '{}': {}", 
                file_path.display(), e
            )))?
            .len() as usize;
        
        // STRICT: File size must match exactly - NO ADJUSTMENTS
        if expected_bytes != file_size {
            return Err(FerroError::new(format!(
                "CRITICAL: Parameter '{}' data/shape mismatch: shape {:?} expects {} elements ({} bytes), got {} bytes in file '{}'",
                param_name, shape, num_elements, expected_bytes, file_size, file_path.display()
            )));
        }
        
        // Read exact amount
        let mut bytes = vec![0u8; expected_bytes];
        file.read_exact(&mut bytes)
            .map_err(|e| FerroError::new(format!(
                "CRITICAL: Failed to read weight data from '{}': {}", 
                file_path.display(), e
            )))?;
        
        // Convert to f32 with strict validation
        let mut data_vec = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let end = start + 4;
            let float_bytes = &bytes[start..end];
            let value = f32::from_le_bytes([float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]]);
            
            // STRICT: Validate all data is finite
            if !value.is_finite() {
                return Err(FerroError::new(format!(
                    "CRITICAL: Parameter '{}' contains non-finite value {} at index {}",
                    param_name, value, i
                )));
            }
            
            data_vec.push(value);
        }
        
        // CRITICAL: Validate tensor data integrity
        if data_vec.is_empty() {
            return Err(FerroError::new(format!(
                "CRITICAL: Parameter '{}' in component '{}' has empty data", 
                param_name, component
            )));
        }
        
        // Create validated tensor
        let tensor = Tensor::from_data(data_vec, shape.clone());
        
        // VALIDATION: Verify tensor has reasonable statistical properties
        let mean: f32 = tensor.data().iter().sum::<f32>() / tensor.data().len() as f32;
        let variance: f32 = tensor.data().iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / tensor.data().len() as f32;
        
        // STRICT: Only check variance for larger tensors - bias vectors can have low variance
        if tensor.data().len() > 1 && variance < 1e-10 {
            return Err(FerroError::new(format!(
                "CRITICAL: Suspiciously low variance ({}) in weights from '{}'. \
                This may indicate zero-initialized or corrupted weights.", 
                variance, file_path.display()
            )));
        }
        
        Ok(tensor)
    }
    
    /// Helper method to load tensor from a specific path - STRICT VERSION
    fn load_tensor_from_path(&self, file_path: &Path, shape: &Vec<usize>) -> Result<Tensor<f32>, FerroError> {
        // Delegate to the comprehensive validation method with generic naming
        self.load_tensor_from_path_with_validation(file_path, shape, "unknown", "unknown")
    }
    
    /// Load a voice by name - NO SILENT FALLBACKS
    pub fn load_voice(&self, voice_name: &str) -> Result<Tensor<f32>, FerroError> {
        // STRICT: Voices metadata must exist
        let voices_meta = self.voices_metadata.as_ref()
            .ok_or_else(|| FerroError::new(
                "CRITICAL: No voices available. Weight directory missing voices/ subdirectory."
            ))?;
        
        // STRICT: Voice must exist
        let voice_meta = voices_meta.voices.get(voice_name)
            .ok_or_else(|| FerroError::new(format!(
                "CRITICAL: Voice '{}' not found. Available voices: {:?}", 
                voice_name, 
                voices_meta.voices.keys().collect::<Vec<_>>()
            )))?;
        
        // Get precise file path
        let voices_path = self.voices_base_path.as_ref()
            .ok_or_else(|| FerroError::new("CRITICAL: Voices base path not initialized"))?;
        
        let file_path = voices_path.join(&voice_meta.file);
        
        // STRICT: Voice file must exist - NO ALTERNATIVES
        if !file_path.exists() {
            return Err(FerroError::new(format!(
                "CRITICAL: Voice file not found: '{}'. \
                This indicates incomplete voice conversion or corrupted voice directory. \
                NO FALLBACKS PERMITTED.", 
                file_path.display()
            )));
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
    
    /// Load a parameter for a component by name
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

