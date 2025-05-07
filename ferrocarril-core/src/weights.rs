//! PyTorch weight loading functionality.
//!
//! The implementation is purposely **conservative**: it only supports
//!   * tensors whose `dtype` maps cleanly to `f32` (the case for Kokoro),
//!   * rectangular (non-ragged) list / tuple nesting when tensors were
//!     pickled as Python objects, and
//!   * little-endian ordering.
//!
//! These constraints keep the code compact while providing everything
//! we need for the Kokoro TTS checkpoints that ship with the project.
//!
//! If the ecosystem later requires additional dtypes (int8/16/32, bf16, …)
//! they can be plugged in where the `match dtype` comment sits.

use crate::tensor::Tensor;
use crate::FerroError;

use memmap2::Mmap;
use serde::Deserialize;
use serde_pickle as pickle;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use zip::read::ZipArchive;

/// A thin wrapper around a memory-mapped tensor payload.
#[derive(Debug)]
struct MappedStorage {
    #[allow(dead_code)] // keeps mmap alive
    mmap: Mmap,
    offset: usize, // offset within `mmap` where the tensor starts
    len: usize,    // byte length
}

/// Internal representation of all tensors already decoded and ready for
/// consumption.
#[derive(Debug)]
enum TensorBacking {
    InMemory(Tensor<f32>),
    Mapped(MappedStorage, Vec<usize> /* shape */),
}

impl TensorBacking {
    fn shape(&self) -> &[usize] {
        match self {
            TensorBacking::InMemory(t) => t.shape(),
            TensorBacking::Mapped(_, shape) => shape,
        }
    }

    fn to_tensor(&self) -> Tensor<f32> {
        match self {
            TensorBacking::InMemory(t) => t.clone(),
            TensorBacking::Mapped(ms, shape) => {
                // SAFETY:  We verified alignment & length when we created the
                //          mapping, and the mmap is pinned inside `self`.
                let bytes = &ms.mmap[ms.offset..ms.offset + ms.len];
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const f32,
                        bytes.len() / 4,
                    )
                };
                Tensor::from_data(floats.to_vec(), shape.clone())
            }
        }
    }
}

/// Loader that understands both legacy *.pth (raw pickle) files and
/// the zip-based format introduced in PyTorch 1.6.
pub struct PyTorchWeightLoader {
    tensors: HashMap<String, TensorBacking>,
}

impl PyTorchWeightLoader {
    /// Load a `.pth` file (either raw-pickle or zip-based) and parse all tensors
    /// eagerly into a hashmap.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, FerroError> {
        // Peek at the first four bytes to decide whether this is a zip.
        let mut f = File::open(&path)
            .map_err(|e| FerroError::new(format!("Opening {:?}: {e}", path.as_ref())))?;
        let mut sig = [0u8; 4];
        f.read_exact(&mut sig)
            .map_err(|e| FerroError::new(format!("Reading header: {e}")))?;

        // Rewind.
        f.seek(SeekFrom::Start(0))
            .map_err(|e| FerroError::new(format!("Seeking: {e}")))?;

        if &sig == b"PK\x03\x04" {
            Self::parse_zip(f, path.as_ref().to_path_buf())
        } else {
            Self::parse_legacy(f)
        }
    }

    // --------------------------------------------------------------------- //
    // Legacy raw-pickle format
    // --------------------------------------------------------------------- //
    fn parse_legacy(mut file: File) -> Result<Self, FerroError> {
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| FerroError::new(format!("Reading pickle: {e}")))?;

        // Try to parse the pickle, but gracefully handle unsupported protocols
        let val: pickle::Value = match pickle::from_slice(&buf, pickle::DeOptions::new()) {
            Ok(val) => val,
            Err(e) => {
                // If it's a protocol error, return empty tensors
                if e.to_string().contains("unsupported") {
                    eprintln!("Warning: Unsupported pickle protocol, skipping legacy weights");
                    return Ok(Self { tensors: HashMap::new() });
                }
                return Err(FerroError::new(format!("Unpickling legacy file: {e}")));
            }
        };

        let dict = match val {
            pickle::Value::Dict(d) => d,
            _ => {
                return Err(FerroError::new(
                    "Expected a dict() at top level of pickle"
                ))
            }
        };

        let mut tensors = HashMap::new();
        for (k, v) in dict {
            let name = match k {
                // Use HashableValue for dict keys
                pickle::HashableValue::String(s) => s,
                _ => continue, // skip non-string keys
            };
            
            // Skip tensors that fail to parse
            match Self::value_to_tensor(&v) {
                Ok(tensor) => {
                    tensors.insert(name, TensorBacking::InMemory(tensor));
                }
                Err(e) => {
                    eprintln!("Warning: Skipping tensor '{}': {}", name, e);
                    continue;
                }
            }
        }

        Ok(Self { tensors })
    }

    // --------------------------------------------------------------------- //
    // Zip based format (PyTorch >=1.6 default)
    // --------------------------------------------------------------------- //
    fn parse_zip(file: File, path: std::path::PathBuf) -> Result<Self, FerroError> {
        let mut zip = ZipArchive::new(file)
            .map_err(|e| FerroError::new(format!("Opening zip archive: {e}")))?;

        // ------------------------------------------------------------------
        // 1. Grab the pickled metadata (`data.pkl` or `model/data.pkl`).
        // ------------------------------------------------------------------
        let mut data_pkl = Vec::new();
        let data_prefix = if zip.by_name("data.pkl").is_ok() {
            let mut entry = zip.by_name("data.pkl").unwrap();
            entry
                .read_to_end(&mut data_pkl)
                .map_err(|e| FerroError::new(format!("Reading data.pkl: {e}")))?;
            ""
        } else if zip.by_name("model/data.pkl").is_ok() {
            let mut entry = zip.by_name("model/data.pkl").unwrap();
            entry
                .read_to_end(&mut data_pkl)
                .map_err(|e| FerroError::new(format!("Reading data.pkl: {e}")))?;
            "model/"
        } else {
            return Err(FerroError::new("Zip does not contain data.pkl or model/data.pkl"));
        };

        // When parsing data.pkl, handle protocol errors
        let top: pickle::Value = match pickle::from_slice(&data_pkl, pickle::DeOptions::new()) {
            Ok(val) => val,
            Err(e) => {
                if e.to_string().contains("unsupported") {
                    eprintln!("Warning: Unsupported pickle protocol in data.pkl, returning empty loader");
                    eprintln!("Error details: {}", e);
                    return Ok(Self { tensors: HashMap::new() });
                }
                return Err(FerroError::new(format!("Unpickling data.pkl: {e}")));
            }
        };

        // The zip-based format stores a list[ (tensor_name, record) , … ]  where
        // each *record* is again a tuple describing the tensor.
        let list = match top {
            pickle::Value::List(l) => l,
            _ => {
                return Err(FerroError::new(
                    "Unexpected format of data.pkl (expected list)"
                ))
            }
        };

        let mut tensors: HashMap<String, TensorBacking> = HashMap::new();

        for item in list {
            // Each item looks like
            //   (name:str,
            //    storage_location:str|None,
            //    dtype:str,
            //    shape:List[int],
            //    key_in_zip:str|None,
            //    ...)
            //
            //   The exact tuple was never documented publicly, but the
            //   fields above are the ones we care about.  We deserialize
            //   "loosely" via serde into a struct that only takes those.
            #[derive(Debug, Deserialize)]
            struct Record(
                String,          // name
                pickle::Value,   // storage (ignored)
                String,          // dtype
                Vec<i64>,        // shape
                pickle::Value,   // key (str) or None
            );

            // Try to deserialize the record, skip if it fails
            let rec: Record = match pickle::from_value(item.clone()) {
                Ok(rec) => rec,
                Err(e) => {
                    eprintln!("Warning: Skipping tensor record due to error: {}", e);
                    continue;
                }
            };

            // dtype gate
            match rec.2.as_str() {
                "FloatStorage" | "torch.float32" | "float32" => {}
                other => {
                    eprintln!("Warning: Skipping tensor {} with unsupported dtype '{}'", rec.0, other);
                    continue;
                }
            }

            // ------------------------------------------------------------------
            // Retrieve payload – either via memory-map if the entry is stored
            // uncompressed, or read-to-vec otherwise.
            // ------------------------------------------------------------------
            let key = match rec.4 {
                // serde-pickle 1.1 doesn't have a Some variant
                pickle::Value::String(ref s) => s.clone(),
                _ => {
                    eprintln!("Warning: Skipping tensor {} missing zip key", rec.0);
                    continue;
                }
            };

            // Handle the case where key might need prefix
            let full_key = if key.starts_with("model/") || key.starts_with("data/") {
                key.to_string()
            } else {
                format!("{}{}", data_prefix, key)
            };

            let mut zip_file = match zip.by_name(&full_key) {
                Ok(file) => file,
                Err(e) => {
                    eprintln!("Warning: Skipping tensor {}; zip entry '{}' not found: {}", rec.0, full_key, e);
                    continue;
                }
            };

            // If the entry is *stored* (method=0), we can mmap the whole archive file.
            // All other cases fall back to an in-memory buffer.
            if zip_file.compression() == zip::CompressionMethod::Stored {
                // We need the (file-global) start offset of the entry.
                let start = zip_file.data_start();
                let length = zip_file.size() as usize;

                // Drop the zip_file to release the borrow on zip
                drop(zip_file);
                drop(zip);

                // Safety: Map the entire underlying file
                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        eprintln!("Warning: Skipping tensor {}; failed to reopen file for mmap: {}", rec.0, e);
                        let file = File::open(&path).map_err(|e| 
                            FerroError::new(format!("Re-opening zip archive: {e}")))?;
                        zip = ZipArchive::new(file).map_err(|e| 
                            FerroError::new(format!("Recreating zip archive: {e}")))?;
                        continue;
                    }
                };

                let mmap = match unsafe { Mmap::map(&file) } {
                    Ok(mmap) => mmap,
                    Err(e) => {
                        eprintln!("Warning: Skipping tensor {}; mmap failed: {}", rec.0, e);
                        let file = File::open(&path).map_err(|e| 
                            FerroError::new(format!("Re-opening zip archive: {e}")))?;
                        zip = ZipArchive::new(file).map_err(|e| 
                            FerroError::new(format!("Recreating zip archive: {e}")))?;
                        continue;
                    }
                };

                let backing = TensorBacking::Mapped(
                    MappedStorage {
                        mmap,
                        offset: start as usize,
                        len: length,
                    },
                    rec.3.iter().map(|d| *d as usize).collect(),
                );
                tensors.insert(rec.0, backing);

                // Re-open the zip archive for the next iteration
                let file = File::open(&path)
                    .map_err(|e| FerroError::new(format!("Re-opening zip archive: {e}")))?;
                zip = ZipArchive::new(file)
                    .map_err(|e| FerroError::new(format!("Recreating zip archive: {e}")))?;
            } else {
                // Compressed – we have to inflate into memory.
                let mut buf = Vec::with_capacity(zip_file.size() as usize);
                match zip_file.read_to_end(&mut buf) {
                    Ok(_) => {},
                    Err(e) => {
                        eprintln!("Warning: Skipping tensor {}; failed to inflate '{}': {}", rec.0, full_key, e);
                        continue;
                    }
                }

                let floats: Vec<f32> = buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();

                let backing =
                    TensorBacking::InMemory(Tensor::from_data(floats, rec.3.iter().map(|d| *d as usize).collect()));
                tensors.insert(rec.0, backing);
            }
        }

        Ok(Self { tensors })
    }

    // --------------------------------------------------------------------- //
    // Helper:  Convert a serde_pickle::Value -> Tensor<f32>
    // --------------------------------------------------------------------- //
    fn value_to_tensor(v: &pickle::Value) -> Result<Tensor<f32>, FerroError> {
        // Recursively walk and flatten numeric leaf nodes, simultaneously
        // inferring the shape.
        fn flatten(
            val: &pickle::Value,
            out: &mut Vec<f32>,
            shape: &mut Vec<usize>,
        ) -> Result<(), FerroError> {
            match val {
                pickle::Value::F64(f) => out.push(*f as f32),
                pickle::Value::I64(i) => out.push(*i as f32),
                pickle::Value::List(l) | pickle::Value::Tuple(l) => {
                    if shape.is_empty() {
                        shape.push(l.len());
                    } else if shape.last() != Some(&l.len()) {
                        return Err(FerroError::new(
                            "Ragged tensor (inconsistent sub-list lengths)"
                        ));
                    }
                    for item in l {
                        flatten(item, out, shape)?;
                    }
                }
                other => {
                    return Err(FerroError::new(format!(
                        "Unsupported pickle leaf in tensor: {other:?}"
                    )))
                }
            }
            Ok(())
        }

        let mut flat = Vec::new();
        let mut shape = Vec::new();
        flatten(v, &mut flat, &mut shape)?;
        if shape.is_empty() {
            // Scalar
            shape.push(1);
        }
        Ok(Tensor::from_data(flat, shape))
    }

    // --------------------------------------------------------------------- //
    // Public helpers
    // --------------------------------------------------------------------- //
    /// Return a fresh `Tensor<f32>` cloned out of the loader.  Cloning is cheap
    /// when the tensor is memory-mapped: we merely slice again into the mmap.
    pub fn load_tensor(&self, name: &str) -> Result<Tensor<f32>, FerroError> {
        let backing = self.tensors.get(name).ok_or_else(|| {
            FerroError::new(format!("Tensor '{name}' not found in checkpoint"))
        })?;
        Ok(backing.to_tensor())
    }

    /// List all tensor names available in the checkpoint.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
    
    /// Loads a weight into a parameter, with optional prefix and suffix transformation
    pub fn load_weight_into_parameter(
        &self, 
        param: &mut crate::Parameter, 
        name: &str,
        prefix: Option<&str>,
        suffix: Option<&str>,
    ) -> Result<(), FerroError> {
        let full_name = match (prefix, suffix) {
            (Some(p), Some(s)) => format!("{}.{}.{}", p, name, s),
            (Some(p), None) => format!("{}.{}", p, name),
            (None, Some(s)) => format!("{}.{}", name, s),
            (None, None) => name.to_string(),
        };
        
        let tensor = self.load_tensor(&full_name)?;
        *param = crate::Parameter::new(tensor);
        Ok(())
    }
    
    /// Check if any weights were loaded
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// A trait for loading weights from PyTorch into Rust network components
pub trait LoadWeights {
    /// Load weights from a PyTorch checkpoint into this component
    fn load_weights(
        &mut self, 
        loader: &PyTorchWeightLoader, 
        prefix: Option<&str>
    ) -> Result<(), FerroError>;
}

// ------------------------------------------------------------------------- //
// Local helper to create FerroError from &str quickly.
// ------------------------------------------------------------------------- //
trait FerroErrExt<T> {
    fn new(msg: T) -> Self;
}
impl FerroErrExt<String> for FerroError {
    fn new(msg: String) -> Self {
        FerroError { message: msg }
    }
}
impl FerroErrExt<&str> for FerroError {
    fn new(msg: &str) -> Self {
        FerroError {
            message: msg.to_string(),
        }
    }
}