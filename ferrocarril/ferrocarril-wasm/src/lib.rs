//! WebAssembly bindings for Ferrocarril TTS.
//!
//! This crate exposes a pair of `#[wasm_bindgen]` types that let a
//! JavaScript or TypeScript caller run Kokoro-82M inference fully
//! inside the browser:
//!
//! 1. [`WasmFerroModelBuilder`] accumulates the JSON metadata for the
//!    model plus the raw bytes for every tensor and voice blob the
//!    caller has fetched. It performs no network or filesystem I/O
//!    itself — the caller decides how the bytes arrive (HTTP `fetch`,
//!    IndexedDB cache, embedded static asset, etc.).
//!
//! 2. Calling [`WasmFerroModelBuilder::build`] produces a
//!    [`WasmFerroModel`] which in turn exposes a `synthesize_ipa`
//!    method that runs the full TTS inference pipeline and returns a
//!    `Float32Array` of 24 kHz mono PCM samples.
//!
//! ```javascript
//! import init, { WasmFerroModelBuilder } from "./ferrocarril_wasm.js";
//! await init();
//!
//! const builder = new WasmFerroModelBuilder(
//!     configJson,
//!     metadataJson,
//!     voicesMetadataJson,
//! );
//! for (const [filename, bytes] of modelBlobs) {
//!     builder.add_model_blob(filename, bytes);
//! }
//! for (const [filename, bytes] of voiceBlobs) {
//!     builder.add_voice_blob(filename, bytes);
//! }
//! const model = builder.build();
//!
//! const audio = model.synthesize_ipa("hɛlqʊ", voicePackBytes, 1.0);
//! ```
//!
//! The IPA phoneme string must be produced on the JavaScript side
//! (using the bundled Phonesis WASM build or any other G2P pipeline);
//! this crate does not perform grapheme-to-phoneme conversion itself.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use ferrocarril::{
    BinaryWeightLoader,
    Config,
    FerroModel,
    MapBlobProvider,
    Tensor,
};

/// Builder that accumulates JSON metadata plus tensor/voice blob bytes
/// before constructing a [`WasmFerroModel`].
#[wasm_bindgen]
pub struct WasmFerroModelBuilder {
    config_json: String,
    metadata_json: String,
    voices_metadata_json: Option<String>,
    model_blobs: HashMap<String, Vec<u8>>,
    voice_blobs: HashMap<String, Vec<u8>>,
}

#[wasm_bindgen]
impl WasmFerroModelBuilder {
    /// Create a new builder from the three JSON metadata strings.
    ///
    /// `voices_metadata_json` may be the empty string to skip voice
    /// support; in that case the caller must still pass a voice pack
    /// directly to `WasmFerroModel::synthesize_ipa`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        metadata_json: &str,
        voices_metadata_json: &str,
    ) -> WasmFerroModelBuilder {
        let voices_opt = if voices_metadata_json.is_empty() {
            None
        } else {
            Some(voices_metadata_json.to_string())
        };
        WasmFerroModelBuilder {
            config_json: config_json.to_string(),
            metadata_json: metadata_json.to_string(),
            voices_metadata_json: voices_opt,
            model_blobs: HashMap::new(),
            voice_blobs: HashMap::new(),
        }
    }

    /// Register the raw little-endian f32 bytes for a model tensor
    /// file. `filename` must match the `file` field inside
    /// `metadata.json` (e.g. `bert/embeddings.word_embeddings.weight.bin`).
    pub fn add_model_blob(&mut self, filename: &str, bytes: Vec<u8>) {
        self.model_blobs.insert(filename.to_string(), bytes);
    }

    /// Register the raw little-endian f32 bytes for a voice blob.
    /// `filename` must match the `file` field inside `voices.json`
    /// (typically `voices/NAME.bin`, including the leading `voices/`).
    pub fn add_voice_blob(&mut self, filename: &str, bytes: Vec<u8>) {
        self.voice_blobs.insert(filename.to_string(), bytes);
    }

    /// Consume the builder and construct the final model. Returns a
    /// `JsValue` error on parse or shape mismatches so the JS caller
    /// can surface a useful message.
    pub fn build(self) -> Result<WasmFerroModel, JsValue> {
        let config = Config::from_json_str(&self.config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse error: {}", e)))?;

        let provider = Box::new(MapBlobProvider::new(
            self.model_blobs,
            self.voice_blobs,
        ));

        let loader = BinaryWeightLoader::from_metadata_str(
            &self.metadata_json,
            self.voices_metadata_json.as_deref(),
            provider,
        )
        .map_err(|e| JsValue::from_str(&format!("binary weight loader error: {}", e)))?;

        let inner = FerroModel::load_from_loader(loader, config)
            .map_err(|e| JsValue::from_str(&format!("model load error: {}", e)))?;

        Ok(WasmFerroModel { inner })
    }
}

/// A loaded Ferrocarril TTS model exposed to JavaScript.
#[wasm_bindgen]
pub struct WasmFerroModel {
    inner: FerroModel,
}

#[wasm_bindgen]
impl WasmFerroModel {
    /// Synthesize audio from IPA phonemes and a raw voice-pack byte
    /// buffer. Returns a flat `Float32Array` of 24 kHz mono PCM.
    ///
    /// `voice_pack` is the raw little-endian f32 bytes of a Kokoro-82M
    /// voice pack. Two shapes are accepted:
    ///   - `[510, 256]`, i.e. 130560 floats, the full voice pack;
    ///   - `[1, 256]`, i.e. 256 floats, a pre-indexed row.
    /// Any other float count returns an error.
    ///
    /// `speed` is the inverse duration scaling factor; `1.0` is the
    /// reference speed. Higher values speed up speech, lower values
    /// slow it down.
    pub fn synthesize_ipa(
        &self,
        ipa_phonemes: &str,
        voice_pack: &[u8],
        speed: f32,
    ) -> Result<Vec<f32>, JsValue> {
        if voice_pack.len() % 4 != 0 {
            return Err(JsValue::from_str(
                "voice_pack byte length must be a multiple of 4 (f32 little-endian)",
            ));
        }
        let num_floats = voice_pack.len() / 4;
        let mut data = Vec::with_capacity(num_floats);
        for i in 0..num_floats {
            let start = i * 4;
            data.push(f32::from_le_bytes([
                voice_pack[start],
                voice_pack[start + 1],
                voice_pack[start + 2],
                voice_pack[start + 3],
            ]));
        }

        // Kokoro voice packs ship as [510, 1, 256]; after flattening
        // they become 510 * 256 = 130560 floats. FerroModel also
        // accepts a single pre-indexed row ([1, 256], 256 floats).
        let (rows, cols) = match num_floats {
            130_560 => (510usize, 256usize),
            256 => (1usize, 256usize),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unexpected voice_pack float count {}; expected 130560 ([510, 256]) or 256 ([1, 256])",
                    num_floats
                )));
            }
        };

        let voice_tensor = Tensor::from_data(data, vec![rows, cols]);

        let audio = self
            .inner
            .infer_with_phonemes(ipa_phonemes, &voice_tensor, speed)
            .map_err(|e| JsValue::from_str(&format!("synthesis error: {}", e)))?;

        Ok(audio)
    }
}