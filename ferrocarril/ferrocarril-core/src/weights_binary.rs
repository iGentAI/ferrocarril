//! Binary weight loading functionality for Ferrocarril.
//! 
//! This module provides a BinaryWeightLoader that can load PyTorch weights
//! converted to a simpler binary format using the `weight_converter.py`
//! script. Blob fetching is abstracted behind the `WeightBlobProvider`
//! trait so the same loader can run on a native filesystem layout or
//! against in-memory buffers (e.g. in WebAssembly).

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
    #[allow(dead_code)]
    format_version: String,
    #[allow(dead_code)]
    original_file: String,
    components: HashMap<String, ComponentMetadata>,
}

/// Metadata for a single voice
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoiceMetadata {
    #[allow(dead_code)]
    name: String,
    file: String,
    shape: Vec<usize>,
    #[allow(dead_code)]
    dtype: String,
    #[allow(dead_code)]
    byte_size: usize,
}

/// Complete metadata for all voices
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoicesMetadata {
    #[allow(dead_code)]
    format_version: String,
    voices: HashMap<String, VoiceMetadata>,
}

/// Trait for fetching tensor blob bytes from arbitrary backends.
///
/// On native targets the canonical implementation is
/// [`FilesystemBlobProvider`], which reads the directory layout produced
/// by `weight_converter.py`. In the browser / WebAssembly, callers can
/// provide their own implementation (for instance an
/// [`MapBlobProvider`] populated from `fetch()` responses) and avoid
/// touching the filesystem entirely.
pub trait WeightBlobProvider {
    /// Fetch the raw little-endian f32 bytes for a model tensor file.
    /// `file` is the value stored in `metadata.json` next to each
    /// parameter (typically `component/param_name.bin`, without any
    /// leading `model/`).
    fn fetch_model_blob(&self, file: &str) -> Result<Vec<u8>, FerroError>;

    /// Fetch the raw little-endian f32 bytes for a voice blob.
    /// `file` is the value stored in `voices.json` (typically
    /// `voices/NAME.bin`, including the leading `voices/`).
    fn fetch_voice_blob(&self, file: &str) -> Result<Vec<u8>, FerroError>;
}

/// Canonical filesystem-backed blob provider.
///
/// Expects a root directory containing:
///   - `model/metadata.json` and per-parameter `.bin` files under
///     `model/` (organised into component subdirectories), and
///   - optionally `voices/voices.json` and per-voice `.bin` files.
pub struct FilesystemBlobProvider {
    base_path: PathBuf,
}

impl FilesystemBlobProvider {
    /// Create a new filesystem provider rooted at `path`.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self { base_path: path.as_ref().to_path_buf() }
    }

    fn read_file(path: &Path) -> Result<Vec<u8>, FerroError> {
        let mut f = File::open(path).map_err(|e| {
            FerroError::new(format!("failed to open '{}': {}", path.display(), e))
        })?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).map_err(|e| {
            FerroError::new(format!("failed to read '{}': {}", path.display(), e))
        })?;
        Ok(bytes)
    }
}

impl WeightBlobProvider for FilesystemBlobProvider {
    fn fetch_model_blob(&self, file: &str) -> Result<Vec<u8>, FerroError> {
        let p = self.base_path.join("model").join(file);
        Self::read_file(&p)
    }

    fn fetch_voice_blob(&self, file: &str) -> Result<Vec<u8>, FerroError> {
        // Voice file entries in `voices.json` already include the
        // leading `voices/` prefix, so we join against the base path
        // directly.
        let p = self.base_path.join(file);
        Self::read_file(&p)
    }
}

/// In-memory blob provider, useful for WebAssembly and tests.
///
/// The caller populates two `HashMap<String, Vec<u8>>` maps (one for
/// model tensor blobs and one for voice blobs) keyed by the `file`
/// strings stored inside `metadata.json` and `voices.json`. The
/// [`BinaryWeightLoader`] then looks up blobs by name without any
/// filesystem access.
pub struct MapBlobProvider {
    model_blobs: HashMap<String, Vec<u8>>,
    voice_blobs: HashMap<String, Vec<u8>>,
}

impl MapBlobProvider {
    /// Create a new in-memory provider from two pre-populated maps.
    pub fn new(
        model_blobs: HashMap<String, Vec<u8>>,
        voice_blobs: HashMap<String, Vec<u8>>,
    ) -> Self {
        Self { model_blobs, voice_blobs }
    }

    /// Create an empty provider that can be populated incrementally.
    pub fn empty() -> Self {
        Self {
            model_blobs: HashMap::new(),
            voice_blobs: HashMap::new(),
        }
    }

    /// Insert (or replace) the bytes for a model tensor file.
    pub fn insert_model_blob(&mut self, file: impl Into<String>, bytes: Vec<u8>) {
        self.model_blobs.insert(file.into(), bytes);
    }

    /// Insert (or replace) the bytes for a voice blob.
    pub fn insert_voice_blob(&mut self, file: impl Into<String>, bytes: Vec<u8>) {
        self.voice_blobs.insert(file.into(), bytes);
    }
}

impl WeightBlobProvider for MapBlobProvider {
    fn fetch_model_blob(&self, file: &str) -> Result<Vec<u8>, FerroError> {
        self.model_blobs
            .get(file)
            .cloned()
            .ok_or_else(|| FerroError::new(format!("no model blob provided for '{}'", file)))
    }

    fn fetch_voice_blob(&self, file: &str) -> Result<Vec<u8>, FerroError> {
        self.voice_blobs
            .get(file)
            .cloned()
            .ok_or_else(|| FerroError::new(format!("no voice blob provided for '{}'", file)))
    }
}

/// Loader for converted binary weights — handles both model weights and
/// voices. Backed by a pluggable [`WeightBlobProvider`] so the same
/// loader can run natively from the filesystem or from in-memory buffers.
pub struct BinaryWeightLoader {
    model_metadata: ModelMetadata,
    voices_metadata: Option<VoicesMetadata>,
    provider: Box<dyn WeightBlobProvider>,
}

impl BinaryWeightLoader {
    /// Load a model from a converted directory. This is a convenience
    /// wrapper around [`BinaryWeightLoader::from_metadata_str`] that
    /// reads `model/metadata.json` and (if present) `voices/voices.json`
    /// from the given path and wires them up to a
    /// [`FilesystemBlobProvider`].
    ///
    /// This method performs filesystem I/O; for environments without a
    /// filesystem (e.g. WebAssembly) use `from_metadata_str` directly.
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self, FerroError> {
        let base_path = path.as_ref().to_path_buf();
        let metadata_path = base_path.join("model").join("metadata.json");

        // Load model metadata
        let mut metadata_file = File::open(&metadata_path)
            .map_err(|e| FerroError::new(format!("Failed to open metadata file: {}", e)))?;

        let mut metadata_content = String::new();
        metadata_file
            .read_to_string(&mut metadata_content)
            .map_err(|e| FerroError::new(format!("Failed to read metadata file: {}", e)))?;

        // Try to load voices metadata if it exists
        let voices_metadata_path = base_path.join("voices").join("voices.json");
        let voices_content = if voices_metadata_path.exists() {
            let mut voices_file = File::open(&voices_metadata_path)
                .map_err(|e| FerroError::new(format!("Failed to open voices metadata: {}", e)))?;

            let mut s = String::new();
            voices_file
                .read_to_string(&mut s)
                .map_err(|e| FerroError::new(format!("Failed to read voices metadata: {}", e)))?;
            Some(s)
        } else {
            None
        };

        let provider: Box<dyn WeightBlobProvider> =
            Box::new(FilesystemBlobProvider::new(&base_path));

        Self::from_metadata_str(
            &metadata_content,
            voices_content.as_deref(),
            provider,
        )
    }

    /// Construct a loader directly from JSON metadata strings and a
    /// user-supplied blob provider. This is the always-available,
    /// WASM-friendly constructor — it performs no filesystem I/O.
    ///
    /// - `model_metadata_json` is the contents of `metadata.json`.
    /// - `voices_metadata_json` is the optional contents of
    ///   `voices.json`; pass `None` to skip voice support.
    /// - `provider` is the [`WeightBlobProvider`] impl responsible for
    ///   returning the bytes of any tensor or voice blob the loader
    ///   subsequently asks for.
    pub fn from_metadata_str(
        model_metadata_json: &str,
        voices_metadata_json: Option<&str>,
        provider: Box<dyn WeightBlobProvider>,
    ) -> Result<Self, FerroError> {
        let model_metadata: ModelMetadata = serde_json::from_str(model_metadata_json)
            .map_err(|e| FerroError::new(format!("Failed to parse model metadata: {}", e)))?;

        let voices_metadata = if let Some(json) = voices_metadata_json {
            let parsed: VoicesMetadata = serde_json::from_str(json)
                .map_err(|e| FerroError::new(format!("Failed to parse voices metadata: {}", e)))?;
            Some(parsed)
        } else {
            None
        };

        Ok(Self {
            model_metadata,
            voices_metadata,
            provider,
        })
    }

    /// Decode the raw little-endian f32 bytes for a tensor and return
    /// a `Tensor<f32>` of the advertised shape. Shared between model
    /// parameters and voice blobs.
    fn decode_f32_blob(
        bytes: &[u8],
        shape: &[usize],
        label: &str,
    ) -> Result<Tensor<f32>, FerroError> {
        let num_elements: usize = shape.iter().product();
        let expected_bytes = num_elements * 4; // f32 = 4 bytes
        if bytes.len() != expected_bytes {
            return Err(FerroError::new(format!(
                "{}: blob has {} bytes, expected {} (shape {:?})",
                label,
                bytes.len(),
                expected_bytes,
                shape,
            )));
        }

        let mut data_vec = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let value = f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ]);
            data_vec.push(value);
        }

        Ok(Tensor::from_data(data_vec, shape.to_vec()))
    }

    /// Load a component and parameter by name
    pub fn load_component_parameter(&self, component: &str, param: &str) -> Result<Tensor<f32>, FerroError> {
        // Find the component
        let component_meta = self.model_metadata.components.get(component)
            .ok_or_else(|| FerroError::new(format!("Component '{}' not found", component)))?;

        // Find the parameter
        let tensor_meta = component_meta.parameters.get(param)
            .ok_or_else(|| FerroError::new(format!("Parameter '{}' not found in component '{}'", param, component)))?;

        // Support for f32 tensors
        if tensor_meta.dtype == "float32"
            || tensor_meta.dtype == "torch.float32"
            || tensor_meta.dtype == "dtype('float32')"
        {
            let num_elements: usize = tensor_meta.shape.iter().product();
            let expected_bytes = num_elements * 4;

            if tensor_meta.byte_size != expected_bytes {
                return Err(FerroError::new(format!(
                    "Byte size mismatch: metadata says {} but calculated {} for tensor {}",
                    tensor_meta.byte_size, expected_bytes, param
                )));
            }

            let bytes = self.provider.fetch_model_blob(&tensor_meta.file)?;
            Self::decode_f32_blob(
                &bytes,
                &tensor_meta.shape,
                &format!("{}.{}", component, param),
            )
        } else {
            Err(FerroError::new(format!(
                "Unsupported tensor dtype: {}",
                tensor_meta.dtype
            )))
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
        let voices_meta = self
            .voices_metadata
            .as_ref()
            .ok_or_else(|| FerroError::new("No voices metadata available"))?;

        // Find the voice
        let voice_meta = voices_meta
            .voices
            .get(voice_name)
            .ok_or_else(|| FerroError::new(format!("Voice '{}' not found", voice_name)))?;

        let bytes = self.provider.fetch_voice_blob(&voice_meta.file)?;
        Self::decode_f32_blob(
            &bytes,
            &voice_meta.shape,
            &format!("voice.{}", voice_name),
        )
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