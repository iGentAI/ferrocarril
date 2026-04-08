# ferrocarril

A pure-Rust, zero-GPU, WebAssembly-capable port of the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech
model (StyleTTS2 + iSTFTNet, ~82M parameters, Apache-2.0).

## Status

- **Numerically validated** against the Python reference at f32
  precision for every transformer and vocoder component (BERT,
  TextEncoder, DurationEncoder, ProsodyPredictor, Decoder, Generator).
- **Quality parity** with Python kokoro across a 25-sample diverse
  regression suite — Rust mean WER 1.40 %, Python mean WER 1.40 %, gap
  **0.00 %**.
- **Fully deterministic** synthesis: identical md5 across runs at any
  thread count.
- **Tests**: `cargo test --workspace --release --no-fail-fast` →
  **173 passed / 0 failed / 0 ignored / 0 warnings** across 34 test
  binaries (24 ferrocarril integration tests + 20 ferrocarril-nn
  lib unit tests + 129 phonesis lib + integration tests).
- **Performance**: ~2 s wall for the canonical "Hi" inference (~130×
  vs the pre-Phase 6 baseline), ~2.14× real-time at N=8 workers on
  Sapphire Rapids; AVX-512 matmul peaks at ~76 GFLOPS per core.
- **WebAssembly**: all four library crates compile to
  `wasm32-unknown-unknown`; `ferrocarril-wasm` exposes the inference
  pipeline through `wasm-bindgen` with a working browser demo.

## Repo Layout

```
ferrocarril/                       # repo root = workspace root
├── Cargo.toml                     # workspace + main binary manifest
├── src/                           # ferrocarril CLI binary + FerroModel
├── ferrocarril-core/              # Tensor, ops, weights, BinaryWeightLoader
├── ferrocarril-dsp/               # Custom STFT/iSTFT, windows, WAV I/O
├── ferrocarril-nn/                # Linear, Conv1d, LSTM, AdaIN, BERT,
│                                  #   TextEncoder, ProsodyPredictor, Decoder
├── ferrocarril-wasm/              # wasm-bindgen crate + browser demo
├── tests/                         # integration tests + Python golden fixtures
├── phonesis/                      # vendored G2P library (CMU dictionary)
├── phonesis_data/                 # dictionary build inputs
├── kokoro/                        # upstream Python reference (read-only)
├── scripts/                       # validate_*.py golden-fixture extractors
├── weight_converter.py            # PyTorch → binary weight converter
├── bench_wasm.js                  # node/bun wasm benchmark harness
├── PLAN.md                        # living analysis + roadmap
├── HANDOFF.md                     # tactical session handoff doc
└── PHASE6_STATUS.md               # performance optimization log
```

## Quick Start

```bash
# 1. Convert real Kokoro weights (one-time, ~5 min via HF CDN)
python3 weight_converter.py --huggingface hexgrad/Kokoro-82M \
    --output ferrocarril_weights

# 2. Build + test
cargo build --release --features weights
cargo test  --workspace --release --no-fail-fast

# 3. Inference via the CLI
./target/release/ferrocarril infer \
    --text "Hello world" \
    --output /tmp/hello.wav \
    --model ferrocarril_weights \
    --voice af_heart
```

## Library Usage

### Native (filesystem)

```rust
use ferrocarril::{Config, FerroModel};

let config = Config::from_json("ferrocarril_weights/config.json")?;
let model  = FerroModel::load_binary("ferrocarril_weights", config)?;
let voice  = model.load_voice("af_heart")?;
let audio  = model.infer_with_phonemes("hɛlqʊ", &voice, 1.0)?;
```

### WebAssembly (in-browser)

```javascript
import init, { WasmFerroModelBuilder } from "./pkg/ferrocarril_wasm.js";
await init();

const builder = new WasmFerroModelBuilder(configJson, metadataJson, voicesMetadataJson);
for (const [name, bytes] of modelBlobs) builder.add_model_blob(name, bytes);
for (const [name, bytes] of voiceBlobs) builder.add_voice_blob(name, bytes);
const model   = builder.build();
const samples = model.synthesize_ipa("hɛlqʊ", voicePackBytes, 1.0);
```

The wasm crate builds with `./ferrocarril-wasm/demo/build.sh`. The
demo page at `ferrocarril-wasm/demo/index.html` is self-running once
served (e.g. via `python3 ferrocarril-wasm/demo/serve.py`).

## Weights

ferrocarril uses the real production Kokoro-82M checkpoint
(81,763,410 parameters, ~340 MB f32). The `weight_converter.py`
script downloads from HuggingFace and writes a directory of binary
blobs plus a `metadata.json`. The on-disk layout:

```
ferrocarril_weights/
├── bert/                # 25 tensors, 6.3M params
├── bert_encoder/        # 2 tensors,  0.4M params
├── text_encoder/        # 24 tensors, 5.6M params
├── predictor/           # 122 tensors, 16.2M params
├── decoder/             # 375 tensors, 53.3M params
├── voices/              # 54 voice packs, [510, 1, 256] f32 each
├── metadata.json
└── config.json
```

The directory is `.gitignore`d — reproduce with the converter on
demand. Total ~340 MB and must not be checked in or synced through
the filestore on the dev sandbox.

## Where to Read Next

- **`PLAN.md`** — canonical analysis of the codebase, the small-TTS
  landscape, and the phased roadmap. Top-of-file is "current
  thinking".
- **`HANDOFF.md`** — self-contained tactical handoff: per-component
  validation status, target-support matrix, sandbox setup, and
  authoritative APIs.
- **`PHASE6_STATUS.md`** — performance optimization log: matmul
  kernel evolution, GFLOPS history, multi-core scaling on Sapphire
  Rapids, and remaining opportunities.

## License

MIT. See `LICENSE`. The `kokoro/` Python reference and the converted
weights are subject to their upstream Apache-2.0 license.