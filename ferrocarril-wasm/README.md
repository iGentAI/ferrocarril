# ferrocarril-wasm

WebAssembly bindings for [Ferrocarril](..), a pure-Rust port of the
Kokoro-82M text-to-speech model (StyleTTS2 + iSTFTNet, ~82M parameters,
Apache-2.0). This crate wraps the native `FerroModel` inference path
in `wasm-bindgen` glue so the full TTS pipeline — grapheme-to-phoneme,
acoustic model, and iSTFTNet vocoder — runs entirely inside a browser
tab, with no ONNX Runtime and no server round-trips during inference.

A working public demo lives at
[ferrocarril-wasm/demo/](demo/) and ships to GitHub Pages via
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml); run
it locally with `./demo/build.sh && python3 demo/serve.py`.

## What you get

Compiling this crate with `wasm32-unknown-unknown` produces a single
`.wasm` module plus a thin ES module JavaScript glue file that exposes
two `#[wasm_bindgen]` classes:

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
    /** Primary entry point: plain English text in, 24 kHz PCM out. */
    synthesize_text(
        text: string,
        voice_pack: Uint8Array,
        speed: number,
    ): Float32Array;

    /** Advanced: feed pre-converted IPA, skipping G2P. */
    synthesize_ipa(
        ipa_phonemes: string,
        voice_pack: Uint8Array,
        speed: number,
    ): Float32Array;
}
```

The builder accepts the three JSON metadata strings (`config.json`,
`model/metadata.json`, `voices/voices.json`) plus the raw bytes of
every tensor and voice blob referenced inside them. Blob fetching is
entirely the caller's responsibility — this crate never touches the
network or filesystem — so you can use `fetch()`, an IndexedDB cache,
an embedded data URL bundle, a packed `weights.bin` + manifest, or
any other source.

Under the hood the builder hands an in-memory `MapBlobProvider` to
`BinaryWeightLoader::from_metadata_str`, then passes the resulting
loader to `FerroModel::load_from_loader`. Grapheme-to-phoneme
conversion uses the in-tree `phonesis` library, which embeds the full
~50 000-word CMU Pronouncing Dictionary plus a rule-based fallback at
compile time, so no separate G2P model has to load on the JS side.
Everything else is the exact same inference code the native
`ferrocarril` binary runs; this crate is pure plumbing.

## Building

The build has three stages: a `cargo` build that produces a raw wasm
binary, a `wasm-bindgen` post-processing pass that emits the
JavaScript glue, and an optional `wasm-opt -Oz` shrink. The
`demo/build.sh` script bundles all three:

```bash
# One-shot build with wasm SIMD enabled. Produces ferrocarril-wasm/demo/pkg/.
./ferrocarril-wasm/demo/build.sh
```

Under the hood it runs:

```bash
RUSTFLAGS="-C target-feature=+simd128" \
    cargo build \
    --release \
    --target wasm32-unknown-unknown \
    --lib \
    -p ferrocarril-wasm

wasm-bindgen \
    --target web \
    --out-dir ferrocarril-wasm/demo/pkg \
    target/wasm32-unknown-unknown/release/ferrocarril_wasm.wasm

wasm-opt -Oz --enable-simd \
    -o ferrocarril-wasm/demo/pkg/ferrocarril_wasm_bg.wasm \
    ferrocarril-wasm/demo/pkg/ferrocarril_wasm_bg.wasm
```

Prerequisites:
- Rust stable with the `wasm32-unknown-unknown` target installed
  (`rustup target add wasm32-unknown-unknown` or, on Fedora,
  `sudo dnf install rust-std-static-wasm32-unknown-unknown`).
- `wasm-bindgen-cli == 0.2.100` (must match the `wasm-bindgen` crate
  version pinned in this crate's `Cargo.toml`). Install with
  `cargo install wasm-bindgen-cli --version 0.2.100 --locked`.
- `binaryen` for `wasm-opt` (optional but recommended for size).

The `+simd128` target feature is important: it's what enables the
hand-written wasm SIMD matmul kernel in
`ferrocarril-core/src/ops/matmul.rs`. Without it the build falls
through to a scalar path that is ~2× slower on inference.

## Usage from JavaScript

Minimum-viable "load model once, synthesise":

```javascript
import init, { WasmFerroModelBuilder } from "./pkg/ferrocarril_wasm.js";

async function loadModel(weightsBase) {
    await init();

    const [configJson, metadataJson, voicesJson] = await Promise.all([
        fetch(weightsBase + "config.json").then(r => r.text()),
        fetch(weightsBase + "model/metadata.json").then(r => r.text()),
        fetch(weightsBase + "voices/voices.json").then(r => r.text()),
    ]);

    const builder = new WasmFerroModelBuilder(
        configJson, metadataJson, voicesJson,
    );

    // Fetch every model tensor. Paths in metadata.json are relative
    // to the `model/` directory.
    const metadata = JSON.parse(metadataJson);
    for (const comp of Object.values(metadata.components)) {
        for (const tmeta of Object.values(comp.parameters)) {
            const bytes = new Uint8Array(
                await fetch(weightsBase + "model/" + tmeta.file)
                    .then(r => r.arrayBuffer()),
            );
            // Native loader key is the file-relative-to-model path,
            // which is exactly what `tmeta.file` holds.
            builder.add_model_blob(tmeta.file, bytes);
        }
    }

    // Fetch the voice pack. `voicesMeta.file` is a bare filename like
    // `af_heart.bin` (relative to the `voices/` directory), so the
    // fetch URL needs an explicit `voices/` prefix.
    const voices = JSON.parse(voicesJson);
    const voiceName = "af_heart";
    const voiceMeta = voices.voices[voiceName];
    const voicePack = new Uint8Array(
        await fetch(weightsBase + "voices/" + voiceMeta.file)
            .then(r => r.arrayBuffer()),
    );
    builder.add_voice_blob(voiceMeta.file, voicePack);

    const model = builder.build();
    return { model, voicePack };
}

async function speak(text) {
    const { model, voicePack } = await loadModel("./weights/");
    const samples = model.synthesize_text(text, voicePack, 1.0);
    // samples is a Float32Array of 24 kHz mono PCM. Feed it into
    // an AudioBufferSourceNode or encode it as WAV for playback.
    return samples;
}
```

See [`demo/main.js`](demo/main.js) for the full reference
implementation: it adds a voice picker, IndexedDB caching of every
blob, a progress bar, a smoke-test harness, and layout-aware URL
rewriting so the same page can talk to a nested directory host
(Hugging Face Hub / Cloudflare R2 / local serve.py) or a flat asset
host (GitHub Releases, where asset names can't contain `/`).

## Size and performance

### Artefact size

Release build with `RUSTFLAGS="-C target-feature=+simd128"` and
`wasm-opt -Oz --enable-simd`:

| Artefact                        |   Size |
|---------------------------------|-------:|
| `ferrocarril_wasm_bg.wasm`      | ~4.0 MB |
| `ferrocarril_wasm_bg.wasm.gz`   | ~1.12 MB |
| `ferrocarril_wasm.js` (glue)    | ~15 KB |
| `ferrocarril_wasm.d.ts`         | ~5.6 KB |

The ~4 MB wasm module contains the complete Kokoro-82M inference
stack (BERT, BertEncoder, TextEncoder, DurationEncoder,
ProsodyPredictor, Decoder, iSTFTNet Generator) **plus** the full
phonesis G2P library with the embedded CMU Pronouncing Dictionary.
The model *weights* themselves are a separate ~340 MB of `.bin` files
that `weight_converter.py` produces; those are fetched at runtime by
the JS side and cached in IndexedDB so returning visitors skip the
cold download.

### Native performance

The native x86_64 target runs **faster than real-time**. Measured on
a 16-vCPU / 8-physical-core Sapphire Rapids host against the
canonical 1.275 s "Hi" input, with the AVX-512 matmul kernel active:

| N workers | Inference wall | matmul_f32 | Generator forward | Inference RTF |
|-----------|---------------:|-----------:|------------------:|--------------:|
| 1         | 889 ms         | 606 ms     | 661 ms            | 1.43×         |
| 2         | 672 ms         | 379 ms     | 464 ms            | 1.90×         |
| 8         | **596 ms**     | **247 ms** | **402 ms**        | **2.14×**     |

The `matmul_f32` kernel hits ~76 GFLOPS on the dominant Generator
shape (~93 % of the 2-FMA-port peak) via an AVX-512 8×32
register-blocked micro-kernel with BLIS b-panel packing and
3-level cache blocking, gated behind `is_x86_feature_detected!`.
See [`../PHASE6_STATUS.md`](../PHASE6_STATUS.md) for the full perf
engineering log.

### Wasm performance

Wasm inference now has its own hand-written SIMD matmul hot path
that mirrors the AVX2 kernel structure but uses
`core::arch::wasm32::v128` intrinsics (4-wide f32, 4×8 register-
blocked micro-kernel, `(KC, NC) = (256, 256)` cache blocking). It is
enabled automatically by the build script's `+simd128` target
feature.

Measured on a 2-vCPU Sapphire Rapids sandbox running Node 22
against the same inputs:

| Build path                          | "Hi" (1.4 s audio) |   RTF | Speedup |
|-------------------------------------|-------------------:|------:|--------:|
| Wasm scalar (no `+simd128`)         |            32.31 s | 0.043×|    0.5× |
| Wasm `+simd128`, LLVM auto-vec      |            15.42 s | 0.091×|    1.0× |
| Wasm `+simd128`, hand-written kernel |        **6.67 s** | **0.210×** | **2.31×** |
| Native AVX-512 (single-thread)      |             2.85 s | 0.49× |       – |

The wasm build is now **~2.3× slower than native AVX-512
single-thread** on this hardware (down from 5.4× slower before the
hand-written kernel). A modern desktop browser typically runs
1.5–2× faster than this 2-vCPU sandbox for compute-bound wasm
workloads, so the projected user experience on a desktop Chrome /
Firefox / Safari is roughly:

| Input                | Audio    | Projected desktop wall |
|----------------------|---------:|-----------------------:|
| `"Hi"`               |  1.23 s  |               ~3–4 s   |
| `"Hello, world."`    |  1.65 s  |               ~4–5 s   |
| Short sentence       |  1.80 s  |               ~5–6 s   |
| Pangram (~85 char)   |  3.25 s  |              ~10–13 s  |

That's in the "click button, watch the spinner, hear the result"
regime — comparable to other browser-based ML demos like kokoro.js
and transformers.js.

An end-to-end headless test (playwright chromium against
`demo/serve.py`) runs the full flow — smoke test → 340 MB blob
fetch → model construction → `synthesize_text("Hello, world.")` →
WAV playback — in about 18 seconds including the blob download.

### Headroom for further speedups

In rough order of impact-per-effort:

1. **`f32x4_relaxed_madd`** (relaxed-simd target feature): a true
   fused multiply-add in one instruction instead of the current
   strict `add(mul(a, b), c)`. Probably another 10–15 %. Gated on
   Chrome ≥ 114 / Firefox ≥ 120 / Safari ≥ 16.4 / Node ≥ 22.
2. **BLIS-style explicit b-panel packing** for the wasm path,
   mirroring what the AVX-512 kernel does. Another ~15–25 % on
   shapes where the b panel doesn't fit in wasm linear memory's
   effective L2 working set.
3. **Wasm threads + `SharedArrayBuffer`**
   (`-C target-feature=+atomics,+bulk-memory`, plus a JS worker
   pool and the `Cross-Origin-Opener-Policy: same-origin` /
   `Cross-Origin-Embedder-Policy: require-corp` Pages headers): the
   m-dimension parallel split could give another 1.5–1.8× on a
   4-core consumer machine. Moderate JS-side complexity; the
   matmul code itself is already structured for it on the native
   side.
4. **fp16 weight quantisation** in `weight_converter.py` +
   `ferrocarril-core::weights_binary::decode_f32_blob`: cuts the
   cold-load weight transfer from ~340 MB to ~170 MB and reduces
   memory-bandwidth pressure on the wasm matmul (where bandwidth
   is more of a bottleneck than compute). Probably 20–30 %
   inference improvement on top of the download-size win.
5. **int8 quantisation**: larger code change (per-channel scales,
   dequant in the inner loop), larger reward (~85 MB downloads,
   possibly faster inference).
6. **Streaming inference**: doesn't change wall time but lets long
   inputs start playing before they finish synthesising.
7. **Dedicated Worker execution**: keeps the page responsive during
   inference on the current kernels. Pure JS-side ergonomics work.

None of those is required to ship the current demo; they're the
list of next steps if interactive single-keystroke latency becomes a
goal.

## Current limitations

- **No audio playback helpers.** `synthesize_text` /
  `synthesize_ipa` return a raw `Float32Array` of 24 kHz mono PCM;
  the caller is responsible for wiring it up to a `WebAudio` node
  or encoding it as WAV. `demo/main.js` has a working WAV encoder
  example.
- **Synchronous inference on the main thread.** The `synthesize_*`
  calls block until the whole utterance is synthesised. For long
  inputs consider running them inside a `DedicatedWorker` so the
  page stays responsive.
- **No relaxed-simd fma yet.** The matmul kernel uses strict
  `add(mul(a, b), c)` for portability across any browser that
  supports baseline `simd128`. Browsers that support
  `relaxed-simd` (Chrome 114+, Firefox 120+, Safari 16.4+) would
  benefit from switching to `f32x4_relaxed_madd` once the
  relaxed-simd feature is stable on the Rust wasm target.

See [`demo/`](demo/) for a full working browser page that exercises
the end-to-end flow, and [`../HANDOFF.md`](../HANDOFF.md) for the
broader project roadmap.