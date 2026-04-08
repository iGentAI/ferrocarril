# Ferrocarril — Architecture and Roadmap

> A pure-Rust, zero-GPU, WebAssembly-capable port of the
> [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech
> model (StyleTTS2 + iSTFTNet, 82M parameters, Apache-2.0).
>
> **Status (April 2026):** quality parity with Python Kokoro (mean
> WER 1.40% for both systems, gap **0.00%**), fully deterministic
> synthesis, **173 passed / 0 failed / 0 warnings** across 34 test
> binaries, **2.14× real-time** at N=8 workers on Sapphire Rapids.

## 1. What it is

Ferrocarril is a from-scratch Rust reimplementation of the Kokoro-82M
text-to-speech model. No PyTorch, no ONNX Runtime, no external C
dependencies — just Rust. Given a text string (or pre-computed IPA
phoneme string) and a voice embedding, it produces 24 kHz mono PCM
audio that matches the Python reference within f32 precision per
component and within stochastic envelope tolerance end-to-end.

Two deployment targets:

- **Native binary.** `cargo build --release` produces the
  `ferrocarril` CLI, which loads the real 82M-parameter Kokoro-82M
  weights from a directory of binary blobs (produced by
  `weight_converter.py`) and runs inference at real-time on a modern
  x86_64 host.
- **WebAssembly library.** `ferrocarril-wasm` compiles to
  `wasm32-unknown-unknown` and exposes the inference pipeline via
  `wasm-bindgen` for in-browser use. See `ferrocarril-wasm/README.md`.

The in-tree `phonesis` crate provides the grapheme-to-phoneme (G2P)
front-end using the CMU Pronouncing Dictionary (~135k entries) plus
misaki-compatible stress handling, function-word destressing,
possessive demotion, and Unicode punctuation tokenisation.

## 2. Pipeline

```
 ┌────────┐  ┌──────────┐  ┌───────┐  ┌──────┐  ┌───────────┐  ┌───────┐  ┌───────┐
 │  text  │→ │ phonesis │→ │ vocab │→ │ BERT │→ │ TextEnc   │→ │prosody│→ │decoder│→ 24 kHz
 │ string │  │   G2P    │  │ token │  │(Alb- │  │ (conv +   │  │ pred  │  │(iSTFT │  mono
 └────────┘  └──────────┘  └───────┘  │ ert) │  │  BiLSTM)  │  │       │  │ Net)  │  PCM
                                      └──────┘  └───────────┘  └───────┘  └───────┘
                                                      +
                                          voice embedding (style + reference)
```

### Component → crate → validation map

| Layer                 | Crate                | Source                                    | Validated by                              |
|-----------------------|----------------------|-------------------------------------------|-------------------------------------------|
| G2P                   | `phonesis`           | `english/mod.rs`, `dictionary/`           | `phonesis/tests/*`                        |
| BERT (PL-BERT)        | `ferrocarril-nn`     | `bert/*.rs`                               | `tests/bert_golden_test.rs`               |
| BERT encoder (Linear) | `ferrocarril-nn`     | `linear.rs`                               | `tests/bert_encoder_golden_test.rs`       |
| TextEncoder           | `ferrocarril-nn`     | `text_encoder.rs`                         | `tests/text_encoder_golden_test.rs`       |
| DurationEncoder       | `ferrocarril-nn`     | `prosody/duration_encoder.rs`             | `tests/duration_encoder_golden_test.rs`   |
| ProsodyPredictor      | `ferrocarril-nn`     | `prosody/mod.rs`                          | `tests/prosody_predictor_golden_test.rs`  |
| Decoder / Generator   | `ferrocarril-nn`     | `vocoder/{mod,adain_resblk1,adain_resblk1d,sinegen,source_module}.rs` | `tests/decoder_golden_test.rs` |
| STFT / iSTFT          | `ferrocarril-dsp`    | `stft.rs`                                 | indirect via `decoder_golden_test`        |
| Weight loading        | `ferrocarril-core`   | `weights_binary.rs`                       | used by every golden test                 |
| End-to-end pipeline   | top-level            | `src/model/ferro_model.rs`, `src/main.rs` | `tests/end_to_end_real_voice_test.rs`     |

Every transformer and vocoder component has a `.npy` golden fixture
under `tests/fixtures/kmodel/`, extracted from the Python reference
in `kokoro/`. The golden tests load the real Kokoro-82M weights,
run the Rust component on the exact input the Python run used, and
assert max-abs diff within tolerance (typically ~1e-6 for the
deterministic paths; ~1% RMS for the stochastic Generator).

## 3. Why Kokoro

Kokoro-82M remains on the Pareto frontier in April 2026 for the
intersection of (a) small, (b) CPU-only, (c) permissive license,
(d) quality, and (e) architecturally tractable for a pure-Rust
from-scratch port.

| Model                | Params  | License   | Arch                                | Quality vs Kokoro | Tractable in Rust? |
|----------------------|---------|-----------|-------------------------------------|-------------------|--------------------|
| **Kokoro-82M** (this)| 82M     | Apache-2  | StyleTTS2 + iSTFTNet, Albert BERT   | baseline          | Moderate           |
| Piper                | 20-75M  | MIT       | VITS (end-to-end + HiFi-GAN)        | −1 MOS            | High               |
| MeloTTS              | ~50M    | MIT       | VITS with BERT front-end            | ≈ Piper           | High               |
| Matcha-TTS           | 18-50M  | MIT       | Flow matching + HiFi-GAN            | ≈ Piper           | High               |
| Kitten TTS           | 15M     | Apache    | VITS-like, int8                     | −2 MOS            | High               |
| StyleTTS2            | 148M    | MIT       | StyleTTS2 + HiFi-GAN                | ≈ Kokoro          | Moderate           |
| F5-TTS               | 335M    | CC-BY-NC  | Flow matching on DiT                | ≈ Kokoro          | too large / NC     |
| OuteTTS / Orpheus    | 700M-3B | Apache    | Llama-style LM over audio tokens    | > Kokoro          | too large          |
| Parler-TTS Mini      | 600M    | Apache    | T5 + Llama-style LM                 | ≈ Kokoro          | too large          |

Piper is the only alternative with a comparable Rust-implementation
cost, but it trails Kokoro on naturalness by roughly 1 MOS point,
especially on long sentences where StyleTTS2 + Albert phonetic
conditioning pulls ahead. Switching would pessimize the quality
ceiling of the library without much to show for it. We revisit the
decision if a new small TTS ships with materially better quality +
simpler architecture + permissive licensing, or if the iSTFTNet
vocoder turns out to be intractable under the "pure Rust, no BLAS
on wasm" constraint (the most plausible escape hatch; the fallback
would be Piper + HiFi-GAN, not a completely different model).

### Kokoro-specific numbers

- **Sample rate:** 24 kHz output.
- **Phoneme vocabulary:** 178 tokens (IPA + control symbols),
  matched by `phonesis::PhonemeStandard::IPA` output.
- **Voices:** 54 official voices in v1.0, each a `[510, 1, 256]`
  sequence-position-indexed style + reference pack.
- **Weights:** 82M params × 4 bytes = ~340 MB f32 on disk. Browser
  deployment will want int8 (~85 MB) or fp16 (~170 MB) quantisation
  — see roadmap §5.

## 4. Current state

### 4.1 Quality

Broad-test validation across 25 diverse text inputs (statements,
questions, exclamations, contractions, numbers, proper nouns,
technical vocab, dialog, news, poetry, conversational fillers,
multi-questions, negations, subordinate clauses, commands, lists,
em-dashes, semicolons, possessives, modals, idioms, pangrams,
complex conditionals):

```
Rust WER:   mean = 1.40%   median = 0.00%   max = 25.0%
Python WER: mean = 1.40%   median = 0.00%   max = 25.0%
Gap (R-P):  mean = +0.00%  median = +0.00%  max = +0.0%

Samples where Rust > Python by > 5%:   (none)
Samples where Python > Rust by > 5%:   (none)
```

The residual 1.40% WER is exclusively from Whisper normalising
"forty two" → "42" and "three" → "3" in two samples — shared
between both systems, not a ferrocarril bug.

### 4.2 Performance

Measured on a privileged 16-vCPU / 8-physical-core Sapphire Rapids
sandbox against the canonical 1.275 s "Hi" input:

| N workers | Wall (incl. load) | Inference only | matmul_f32 | Generator forward | Inference RTF |
|-----------|------------------:|---------------:|-----------:|------------------:|--------------:|
| 1         | 1424 ms           | 889 ms         | 606 ms     | 661 ms            | 1.43×         |
| 2         | 1210 ms           | 672 ms         | 379 ms     | 464 ms            | 1.90×         |
| 8         | **1122 ms**       | **596 ms**     | **247 ms** | **402 ms**        | **2.14×**     |

The `matmul_f32` kernel hits ~76 GFLOPS on the dominant Generator
shape (~93% of the 2-FMA-port peak) via an AVX-512 8×32 register-
blocked micro-kernel with BLIS b-panel packing and 3-level cache
blocking. See `PHASE6_STATUS.md` for the optimization log and full
per-stage profile breakdown.

### 4.3 Determinism

`md5` of the same input across multiple runs at any thread count is
identical. The two `SineGen` / `SourceModuleHnNSF` randomness sites
are seeded from fixed constants rather than `thread_rng()`, and the
noise distribution uses Box-Muller Gaussian samples instead of a
uniform [-1, 1] distribution with std ~0.577.

### 4.4 Tests

`cargo test --workspace --release --no-fail-fast` →
**173 passed / 0 failed / 0 ignored / 0 warnings** across 34 test
binaries:

- **24** ferrocarril integration tests (BERT / TextEncoder / Prosody
  / Decoder golden tests, end-to-end real voice, G2P integration,
  simple round-trip).
- **20** ferrocarril-nn lib unit tests (LSTM, Conv, AdaIN, BERT,
  TextEncoder, Prosody, Vocoder).
- **129** phonesis tests (dictionary, rules, normaliser, English
  G2P, IPA conversion, TTS edge cases, robustness, Ferrocarril
  adapter).

All tests use the real Kokoro-82M weights for the ferrocarril
crates and the vendored CMU dictionary for phonesis.

### 4.5 Known limitations

1. **510-token hard cap on single-call inference.** Inputs longer
   than ~510 IPA characters (including whitespace and punctuation)
   are truncated because of Kokoro's BERT `max_position_embeddings
   = 512` minus BOS/EOS. Python Kokoro avoids this by internal
   sentence chunking in `KPipeline.__call__`; ferrocarril's CLI
   does not currently chunk. Adding a sentence-boundary splitter
   in `src/model/g2p.rs` or `src/main.rs` is the natural follow-up.
2. **Out-of-vocabulary words** fall through to the phonesis rules
   engine which produces letter-by-letter character-fallback
   output. For example "ferrocarril" itself (not in CMU) comes
   through garbled instead of misaki's `fˈɛɹəkˌæɹᵊl`. Python
   Kokoro has the same gap — both systems score 9.1% WER on the
   sample containing "ferrocarril". Not critical for normal text;
   a future fix is an espeak-ng-style letter-to-sound engine.
3. **Wasm target is not yet real-time** — the AVX-512 matmul
   kernels are gated on `is_x86_feature_detected!` and don't apply
   in the wasm build. See roadmap §5 for the SIMD + quantisation
   path to real-time in the browser.

## 5. Roadmap

The performance and correctness workstreams are complete for the
native x86_64 target. The forward-looking items are all wasm /
browser polish plus the OOV G2P gap and long-input chunking.

1. **Weight quantisation (int8 / fp16).** The wasm module itself
   is 2.8 MB but the weight pack it loads is ~340 MB of f32
   tensors. Packing as int8 at load time would shrink the download
   4× (~85 MB). Both current development sandboxes (Ice Lake and
   Sapphire Rapids) expose `avx512_vnni`, so an int8 matmul kernel
   using `vpdpbusd` is a viable next step. `weight_converter.py`
   is the logical place to add the packing; `MapBlobProvider` /
   `decode_f32_blob` would grow a matching decode path.
2. **wasm SIMD hot paths.** Building `ferrocarril-wasm` with
   `RUSTFLAGS="-C target-feature=+simd128"` should recover most of
   the native/wasm perf gap for browsers that support wasm SIMD.
   Requires porting the matmul and Conv1d hot paths to use
   `core::arch::wasm32` intrinsics (or `std::simd` once stable).
3. **Streaming inference.** Right now `synthesize_ipa` runs the
   whole utterance in one shot. Chunking the decoder and emitting
   partial audio as it's ready would let long inputs start playing
   before they finish computing. Needs work in `ProsodyPredictor`
   + `Decoder` to support a chunked-forward API.
4. **Dedicated Worker support.** The browser demo runs synthesis
   on the main thread. Moving `WasmFerroModel` into a
   `DedicatedWorkerGlobalScope` would let the page stay responsive
   during inference. Mostly a wasm-bindgen ergonomics task.
5. **Long-input sentence chunking.** Port of Python Kokoro's
   `KPipeline.__call__` sentence splitter so inputs above the
   510-token BERT cap no longer truncate. Restores parity on the
   longest test samples (Gettysburg, Frankenstein passages, etc.).
6. **OOV letter-to-sound engine in phonesis.** An espeak-ng-style
   pronunciation predictor for words not in the CMU dictionary
   would eliminate the ~0.1% WER contribution from character
   fallback on OOV inputs.
7. **Persistent thread pool.** The matmul kernel currently
   re-allocates the per-worker packed-B buffer on every
   `std::thread::scope` call. A persistent pool would recover the
   ~50-80 ms per inference currently lost to L2 cache eviction at
   N=8 workers.

## 6. Where to read more

- **`README.md`** — user-facing quick start, install, CLI and
  library usage examples.
- **`HANDOFF.md`** — tactical session handoff: per-component
  validation table, sandbox setup commands, authoritative native
  and wasm APIs, things not to do.
- **`PHASE6_STATUS.md`** — performance optimization log: matmul
  kernel evolution, timing history, multi-core scaling on Sapphire
  Rapids, mutagen cross-sandbox gotcha, deferred perf items.
- **`ferrocarril-wasm/README.md`** — wasm bindings, build steps,
  browser demo walkthrough, JS glue API.
- **`phonesis/README.md`** — G2P library API, phoneme standards,
  Ferrocarril adapter.