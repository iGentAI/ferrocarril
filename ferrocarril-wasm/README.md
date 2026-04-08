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
live in the ~340 MB of `.bin` files that `weight_converter.py`
produces; those are fetched separately by the JS side at load time.

### Native performance

The native x86_64 target runs **faster than real-time**. Measured
on a 16-vCPU / 8-physical-core Sapphire Rapids sandbox against the
canonical 1.275 s "Hi" input, with the Phase 6 AVX-512 matmul
kernels active:

| N workers | Inference wall | matmul_f32 | Generator forward | Inference RTF |
|-----------|---------------:|-----------:|------------------:|--------------:|
| 1         | 889 ms         | 606 ms     | 661 ms            | 1.43×         |
| 2         | 672 ms         | 379 ms     | 464 ms            | 1.90×         |
| 8         | **596 ms**     | **247 ms** | **402 ms**        | **2.14×**     |

The `matmul_f32` kernel hits ~76 GFLOPS on the dominant Generator
shape (~93 % of the 2-FMA-port peak) via an AVX-512 8×32
register-blocked micro-kernel with BLIS b-panel packing and
3-level cache blocking, gated behind `is_x86_feature_detected!`.
See `../PHASE6_STATUS.md` for the full perf engineering log.

### Wasm performance

The AVX-512 kernels are x86-specific and don't apply to the
`wasm32-unknown-unknown` target; wasm currently runs on the scalar
+ auto-vectorised fallback path, which is **substantially slower
than native** even on modern browser JITs. Wasm benchmark numbers
are not currently tracked because the next step is the SIMD port
below; we'll re-benchmark once the `wasm32::simd128` kernels land.

The remaining wasm-side optimisations needed to make in-browser
inference interactive, in rough impact order:

1. **wasm SIMD hot paths.** Building with
   `RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown`
   should recover most of the native/wasm perf gap for browsers
   that support wasm SIMD (every current Chromium, Firefox,
   and Safari). Requires porting the matmul and Conv1d hot paths
   to use `core::arch::wasm32` intrinsics (or `std::simd` once
   stable on the wasm target).
2. **Int8 / fp16 weight quantisation.** Cuts the ~340 MB weight
   download by 2–4× (to ~85 MB int8 or ~170 MB fp16), and reduces
   the memory-bandwidth pressure on the matmul kernel. Handled in
   `weight_converter.py` on the producer side and in
   `ferrocarril-core::weights_binary::decode_f32_blob` on the
   consumer side.
3. **Streaming inference.** Right now `synthesize_ipa` synthesises
   the whole utterance in one shot, which blocks the main thread
   for long inputs. Chunking the decoder and emitting partial
   audio as it's ready would let long inputs start playing before
   they finish computing.
4. **Dedicated Worker execution.** Mostly a wasm-bindgen
   ergonomics task — moving `WasmFerroModel` into a
   `DedicatedWorkerGlobalScope` lets the page stay responsive
   during inference even on the current scalar kernels.

Until those four land (especially #1), use this crate for
correctness validation, demos, and offline batch rendering. It's
not yet a viable production path for interactive in-browser TTS.

## Current limitations

- **No G2P inside the module.** Callers must supply IPA. See §"G2P".
- **No audio playback helpers.** `synthesize_ipa` returns a raw
  `Float32Array`; the caller is responsible for wiring it up to a
  `WebAudio` node or encoding it as WAV.

See `demo/` for a small browser page that exercises the full load +
synthesis path, and `../HANDOFF.md` for the broader project
roadmap.