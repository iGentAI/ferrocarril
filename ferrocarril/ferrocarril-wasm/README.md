# ferrocarril-wasm

WebAssembly bindings for [Ferrocarril](..), a pure-Rust port of the
Kokoro-82M text-to-speech model (StyleTTS2 + iSTFTNet, ~82M parameters,
Apache-2.0). This crate wraps the native `FerroModel` inference path in
`wasm-bindgen` glue so the full TTS pipeline can run entirely inside a
browser tab, without an ONNX Runtime or any server round-trips.

## What you get

Compiling this crate with `wasm32-unknown-unknown` produces a single
`.wasm` module plus a thin JavaScript glue file that exposes two
`#[wasm_bindgen]` classes:

```typescript
class WasmFerroModelBuilder {
    constructor(
        config_json: string,
        metadata_json: string,
        voices_metadata_json: string, // "" to skip
    );
    add_model_blob(filename: string, bytes: Uint8Array): void;
    add_voice_blob(filename: string, bytes: Uint8Array): void;
    build(): WasmFerroModel;
}

class WasmFerroModel {
    synthesize_ipa(
        ipa_phonemes: string,
        voice_pack: Uint8Array,
        speed: number,
    ): Float32Array; // 24 kHz mono PCM
}
```

The builder accepts the three JSON metadata strings (`config.json`,
`model/metadata.json`, `voices/voices.json`) plus the raw bytes of every
tensor and voice blob referenced inside them. Blob fetching is entirely
the caller's responsibility — this crate never touches the network or
filesystem — so you can use `fetch()`, an IndexedDB cache, an embedded
data URL bundle, or any other source.

Under the hood the builder hands an in-memory `MapBlobProvider` to
`BinaryWeightLoader::from_metadata_str`, then passes the resulting
loader to `FerroModel::load_from_loader`. Everything else is the exact
same inference code the native `ferrocarril` binary runs; this crate is
pure plumbing.

## Building

The build has two stages: a `cargo` build that produces a raw `.wasm`
binary, and a `wasm-bindgen` post-processing pass that emits the
JavaScript glue and the wasm-bindgen-processed module.

```bash
# Stage 1: compile Rust → wasm32
cargo build --release \
    --target wasm32-unknown-unknown \
    --lib \
    -p ferrocarril-wasm

# Stage 2: generate JS bindings (optional: also run wasm-opt)
wasm-bindgen \
    --target web \
    --out-dir demo/pkg \
    target/wasm32-unknown-unknown/release/ferrocarril_wasm.wasm

# Optional: shrink further with wasm-opt
wasm-opt -Oz \
    -o demo/pkg/ferrocarril_wasm_bg.wasm \
    demo/pkg/ferrocarril_wasm_bg.wasm
```

The `demo/build.sh` script in this directory bundles all three commands
behind one entry point. Note that `wasm-bindgen` CLI and the
`wasm-bindgen` crate dependency must be the same version; this crate
currently pins `wasm-bindgen = "=0.2.100"`.

## Usage from JavaScript

```javascript
import init, { WasmFerroModelBuilder } from "./pkg/ferrocarril_wasm.js";

async function loadModel() {
    await init();

    // Fetch the three JSON metadata files
    const [configJson, metadataJson, voicesJson] = await Promise.all([
        fetch("/weights/config.json").then(r => r.text()),
        fetch("/weights/model/metadata.json").then(r => r.text()),
        fetch("/weights/voices/voices.json").then(r => r.text()),
    ]);

    const builder = new WasmFerroModelBuilder(
        configJson, metadataJson, voicesJson,
    );

    // Fetch every model tensor blob. The `file` fields in metadata.json
    // are relative to `weights/model/`.
    const metadata = JSON.parse(metadataJson);
    for (const [_cname, comp] of Object.entries(metadata.components)) {
        for (const [_pname, tmeta] of Object.entries(comp.parameters)) {
            const bytes = new Uint8Array(
                await fetch(`/weights/model/${tmeta.file}`)
                    .then(r => r.arrayBuffer()),
            );
            builder.add_model_blob(tmeta.file, bytes);
        }
    }

    // Fetch the voice pack for the voice you want to use
    const voices = JSON.parse(voicesJson);
    const voiceName = "af_heart";
    const voiceMeta = voices.voices[voiceName];
    const voiceBytes = new Uint8Array(
        await fetch(`/weights/${voiceMeta.file}`)
            .then(r => r.arrayBuffer()),
    );
    builder.add_voice_blob(voiceMeta.file, voiceBytes);

    const model = builder.build();
    return { model, voicePack: voiceBytes };
}

async function speak(ipa) {
    const { model, voicePack } = await loadModel();
    const samples = model.synthesize_ipa(ipa, voicePack, 1.0);
    // samples is a Float32Array of 24 kHz mono PCM — feed it into
    // an AudioBufferSourceNode to play it back.
    return samples;
}
```

## G2P (text → IPA)

This crate does **not** perform grapheme-to-phoneme conversion. You
must produce the IPA phoneme string on the JavaScript side before
calling `synthesize_ipa`. Options:

- Compile the in-tree `phonesis` crate to a separate WASM module and
  use it from the same page.
- Use [phonemize.js](https://github.com/tsx-awesome-g2p/phonemize) or
  similar JS G2P libraries.
- Pre-generate IPA strings on the server and ship them alongside the
  text.

## Size and performance

### Artefact size

On a typical release build with `wasm-opt -Oz`:

| Artefact                        | Size    |
|---------------------------------|---------|
| `ferrocarril_wasm_bg.wasm`      | ~2.8 MB |
| `ferrocarril_wasm_bg.wasm.gz`   | ~790 KB |
| `ferrocarril_wasm.js` (glue)    | ~17 KB  |
| `ferrocarril_wasm.d.ts`         | ~4 KB   |

The 2.8 MB module contains the complete Kokoro-82M inference stack:
BERT, BertEncoder, TextEncoder, DurationEncoder, ProsodyPredictor,
Decoder, and the iSTFTNet Generator. The model *weights* themselves
live in the ~340 MB of `.bin` files that the Rust `weight_converter.py`
script produces; those are fetched separately by the JS side at load
time. A future work item is int8 / fp16 weight packing to shrink the
weight payload by 2–4×.

### Real-time factor (current baseline)

Measured on a 2 vCPU x86_64 sandbox against the canonical 1.4 s
`hɛlqʊ` test fixture using `bench_wasm.js` at the repo root:

| Runtime                                 | Inference wall | Audio    | Real-time factor    |
|-----------------------------------------|---------------:|---------:|--------------------:|
| Native Rust (release)                   | ~258 s         | ~1.0 s   | **~0.004× RT** (258×) |
| Node 22.19 wasm (wasm32, no SIMD)       | 368 s          | 1.40 s   | **0.004× RT** (263×)  |
| Bun 1.2.23 wasm (wasm32, no SIMD)       | 278 s          | 1.40 s   | **0.005× RT** (198×)  |

**The current build is ~200-260× slower than real-time.** All three
configurations are bottlenecked by `ferrocarril-core::ops::matmul` —
a plain scalar triple loop with no cache blocking, no SIMD, and no
BLAS. Wasm is not the problem; the unoptimised compute kernel is.
(Bun's JavaScriptCore JIT actually produces slightly faster wasm
code than native Rust release on this loop pattern, because the
Rust `matmul` has no SIMD hints for LLVM to vectorise.)

### Path to real-time (Phase 6)

To make the current model viable for interactive use in the browser,
all four of the following need to land (they compound):

1. **Cache-blocked matmul** in `ferrocarril-core::ops::matmul`. Single
   most impactful change — should bring native into the 10-30× slower
   range.
2. **im2col Conv1d fast path**. Routes the conv1d hot path through the
   same blocked matmul kernel.
3. **wasm SIMD** (`RUSTFLAGS="-C target-feature=+simd128"`). Another
   2-4× for wasm specifically. Supported by every modern browser.
4. **Int8 / fp16 weight quantisation**. Shrinks the memory bandwidth
   bottleneck 2-4× and cuts the browser download from ~340 MB to
   ~80-170 MB.

Realistic post-optimisation target: **5-10× real-time on a modern
laptop, approaching real-time on a recent mobile CPU**.

Until that work lands, use this crate for correctness validation,
demos, and offline batch rendering. It is not yet a viable production
path for interactive TTS.

## Current limitations

- **No G2P inside the module.** Callers must supply IPA. See §"G2P".
- **No weight quantisation yet.** The full f32 weights (~340 MB) are
  fetched at load time. Int8 / fp16 packing is a future work item.
- **No streaming inference.** The whole utterance is synthesised in one
  shot. Long inputs will block the main thread unless you run the
  module in a dedicated Worker.
- **No audio playback helpers.** `synthesize_ipa` returns a raw
  `Float32Array`; the caller is responsible for wiring it up to a
  `WebAudio` node or encoding it as WAV.

See `demo/` for a small browser page that exercises the full load +
synthesis path, and `../../HANDOFF.md` for the broader project
roadmap.