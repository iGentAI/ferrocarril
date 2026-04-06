# Ferrocarril Review and Plan — April 2026

> **Phase 3 numerical correctness is COMPLETE** (end of session, April 2026).
> The Rust Kokoro-82M port now matches the Python reference to f32 precision
> across every transformer component (BERT / TextEncoder / DurationEncoder /
> ProsodyPredictor / Decoder) and within ~1 % global RMS + 0.99 envelope
> correlation on the final generator audio. `cargo test --release --features
> weights` runs clean at 22 passed / 0 failed / 0 ignored. The `ferrocarril
> infer` CLI produces intelligible 24 kHz audio from real text via G2P. For
> the tactical summary of what shipped, the test matrix, the sandbox setup
> and the suggested Phase 4 (cleanup) and Phase 5 (WASM) next steps, read
> **`HANDOFF.md`** — it is the authoritative "what's done and what's next"
> document. The rest of this file is retained as historical context
> (especially §3 "Is Kokoro Still the Right Target?" and the phased
> roadmap).

---

> **Goal:** A pure-Rust, zero-GPU, WebAssembly-cross-compilable text-to-speech
> system. Originally conceived as a from-scratch port of the Kokoro-82M model
> built on StyleTTS2 + iSTFTNet, using Phonesis (in-tree) for grapheme-to-phoneme.

This document is the authoritative review-and-roadmap artifact created after a
full audit of the repository on 2026-04-06. It supersedes the historical
`STATUS.md`, `NEXT_STEPS.md`, `IMPLEMENTATION_PROGRESS.md`, and
`FERROCARRIL_TTS_BURNDOWN.md` scattered under `ferrocarril/` for the purpose of
orienting future work. The historical documents are retained unchanged as
archival references; their content is not self-consistent with the code on
disk today.

---

## 1. Executive Summary

1. **The project exists in two partially overlapping copies.** The repo root
   contains a broken, incomplete refactor attempt (top-level
   `ferrocarril-core/`, `ferrocarril-nn/`, `ferrocarril-dsp/`) that does not
   declare a workspace, that has crates importing modules that were never
   written, and that the project gave up on mid-flight. The real codebase is
   the nested workspace at `ferrocarril/ferrocarril/`, which has a full
   `[workspace]` manifest, a main binary, the Kokoro architecture components,
   an integrated `Phonesis` G2P, and a set of integration tests.
2. **The real workspace does not compile.** The Phonesis dependency builds
   clean (with some cosmetic warnings). The nested workspace fails with 22
   compile errors in `ferrocarril-nn`, almost all of which are the vocoder
   module calling `Conv1d::set_weight_norm`, `Conv1d::set_bias`, and
   `LoadWeightsBinary::load_weights_binary` on `Conv1d` / `AdaIN1d` — methods
   and trait impls that were referenced from higher layers but never
   implemented on those layers. One additional error is a missing
   `#[derive(Debug)]` on `Conv1d`.
3. **The documentation overstates the state of the code.** `STATUS.md` and
   `IMPLEMENTATION_PROGRESS.md` claim end-to-end inference, validated
   bidirectional LSTM, validated AdaIN, and working component weight loading.
   None of this can be true in the head commit because the crate does not
   link. The docs appear to reflect an earlier working snapshot that was then
   invalidated by in-flight integration work.
4. **Kokoro is still the right target** for a pure-Rust, CPU-only, WASM-able
   TTS in this size class in April 2026. See §3 for the comparison against
   Piper/VITS, Kitten TTS, MeloTTS, Pocket TTS (Kyutai, 2026), Chatterbox,
   StyleTTS2, F5-TTS, OuteTTS, Matcha-TTS, and Parler-TTS. None of these
   simultaneously satisfy (quality ≥ Kokoro) ∧ (size ≤ 150M params) ∧
   (permissive license) ∧ (architecturally tractable for pure-Rust
   reimplementation). Kokoro remains the Pareto frontier. The only serious
   competitor on *implementability* (VITS-based Piper) is a meaningful step
   down on naturalness and prosody.
5. **A lot of the existing Ferrocarril work is recoverable.** Phonesis is in
   good shape. `ferrocarril-core::weights_binary::BinaryWeightLoader`, the
   Python `weight_converter.py`, the `config.json` schema, the scaffolds for
   every StyleTTS2/iSTFTNet component, the Python reference checkout under
   `kokoro/`, and the Python-side validation scripts are all useful and stay.
6. **The path forward** is a disciplined cleanup + correctness pass, not a
   rewrite: (a) consolidate to one workspace; (b) get a clean `cargo build`
   and `cargo test` baseline; (c) work the existing burndown list against the
   real Python reference while validating numerically per component; (d) only
   then add the `wasm32-unknown-unknown` target. Phases laid out in §5.

---

## 2. Repository Inventory (as of 2026-04-06)

Layout seen on disk at `~/ferrocarril/`:

```
ferrocarril/                                     # repo root
├── README.md                                    # one-paragraph stub
├── LICENSE, *.md (top-level docs)               # legacy, partially out of date
├── ferrocarril-core/        ❌ broken refactor  # no workspace, no `Cargo.toml` at root,
├── ferrocarril-nn/          ❌ broken refactor  # imports modules that do not exist,
├── ferrocarril-dsp/         ❌ broken refactor  # abandoned. To be removed.
├── phonesis/                ✅ builds           # G2P library, ~65k-word dictionary, tests pass
├── phonesis_data/                               # wikipron-derived dictionary sources
├── kokoro/                  📚 reference        # Python reference implementation (upstream)
├── weight_converter.py                          # PyTorch → binary weight converter (canonical)
├── {various scripts}.py / .sh                   # dictionary + weight tooling, mostly OK
├── simple_test.rs, test_*.rs (top level)        # orphan scaffolds, not part of any crate
└── ferrocarril/                                 # **the real workspace**
    ├── Cargo.toml                               # [workspace] with members core/dsp/nn
    ├── src/main.rs, src/model/                  # main binary + FerroModel pipeline
    ├── ferrocarril-core/                        # tensors, ops, weight loaders, FerroError
    ├── ferrocarril-nn/                          # linear, conv, lstm, adain, text_encoder,
    │                                            # prosody, vocoder, bert/CustomAlbert
    ├── ferrocarril-dsp/                         # custom STFT/iSTFT, windows, WAV I/O
    ├── tests/                                   # 15 integration tests
    └── {many .md docs}                          # legacy design/status/burndown docs
```

**Build state (verified empirically with `cargo check`):**

| Crate                                    | Status        | Notes                                    |
|------------------------------------------|---------------|------------------------------------------|
| `phonesis`                               | ✅ compiles   | 25 warnings (static_mut_refs, unused), 119 tests claimed passing in docs |
| `ferrocarril/ferrocarril-core`           | ✅ compiles   | Defines `LoadWeightsBinary` trait, `BinaryWeightLoader`, `Tensor`, `Config`, `FerroError`, `PhonesisG2P` |
| `ferrocarril/ferrocarril-dsp`            | ✅ compiles   | Custom STFT, windows, `save_wav`          |
| `ferrocarril/ferrocarril-nn`             | ❌ **22 errors** | See §2.1                                  |
| `ferrocarril` (main bin)                 | ⛔ blocked    | Blocked by `ferrocarril-nn`              |
| top-level `ferrocarril-core/` (refactor) | ❌ dead       | No workspace, broken imports, to delete  |
| top-level `ferrocarril-nn/` (refactor)   | ❌ dead       | Declares modules that don't exist        |
| top-level `ferrocarril-dsp/` (refactor)  | ❌ dead       | Orphan                                    |

### 2.1 The 22 `ferrocarril-nn` errors

All 22 errors cluster around three root causes in the vocoder layer:

| # | Root cause                                                                                   | Fix                                                                                  |
|---|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1 | `Conv1d` has no `#[derive(Debug)]`; it is used inside a `#[derive(Debug)]` vocoder struct.   | Add `#[derive(Debug)]` to `Conv1d`.                                                  |
| 2 | `Conv1d::set_weight_norm(&g, &v)` is called but not implemented.                             | Add method that reconstructs `w = g * v / ‖v‖₂` along the out-channel dim.          |
| 3 | `Conv1d::set_bias(&b)` is called but not implemented.                                        | Add a setter that stores the bias tensor.                                            |
| 4 | `Conv1d` does not `impl LoadWeightsBinary`. Called at ~12 sites.                             | Add impl that tries `.weight_g/.weight_v`, then `_weight_g/_weight_v`, then `.weight`, plus optional `.bias`. |
| 5 | `AdaIN1d` does not `impl LoadWeightsBinary`. Called at ~4 sites.                             | Delegate to the inner `Linear` under the sub-prefix `{prefix}.fc` to match Kokoro PyTorch state-dict layout. |

These are all mechanical fixes. They do not require algorithmic rewrites.
They are addressed in the first cleanup pass (Phase 1 below, landed together
with this plan document).

### 2.2 Correctness issues beyond compilation

These are known from the existing `FERROCARRIL_TTS_BURNDOWN.md`, plus my own
read of the nested sources. They survive cleanup and become the Phase 3
workload:

- **LSTM**: The nested `lstm.rs` nominally supports bidirectional, but the
  burndown notes that variable-length sequence handling is missing (no
  `pack_padded_sequence`), hidden state updates past sequence end are not
  frozen, and input projection silently pads/truncates on dimension
  mismatches. This is a silent fallback and must be removed in favor of
  assertions.
- **ProsodyPredictor**: Style is concatenated in two places (duration encoder
  and forward), which double-applies style conditioning relative to the
  Python reference. Energy pooling `en` is missing style dims. LSTM input
  ordering does not match the Python `.transpose(-1, -2)` → LSTM →
  `.transpose(-1, -2)` pattern. An extra transpose between channels and
  frames is wrong in the current code.
- **AdaIN1d**: `InstanceNorm1d` must be built with `affine=false` to match
  Kokoro. Channel-mismatch fallback must become a hard assert. γ/β slicing
  must assert `fc.out_features == 2 * num_features`.
- **CustomAlbert / BERT**: Embedding size is hard-coded to 128; must be read
  from config. Feed-forward needs the second projection (`intermediate →
  hidden`). Residual connections need the pre-LayerNorm pattern. Attention
  mask semantics are inverted relative to HuggingFace (rust code says
  "1 = masked", HF convention is "1 = visible").
- **Decoder / iSTFTNet**: `f0_upsamp` output must be transposed to `[B, T, 1]`
  before the source module. Reflection padding must be left-side only.
  AdainResBlk1d upsampling is noted as fixed but worth re-validating against
  Python.
- **Voice embeddings**: The Kokoro voice files ship as `[510, 1, 256]` and
  Ferrocarril flattens them; the flatten-then-split produces wrong reference
  and style halves.
- **Alignment tensor**: `create_alignment_from_durations` in the binary
  target looks correct; but the prosody `forward` path was being called
  twice — once with a temporary identity alignment to extract durations,
  then again with the real alignment. The duration logits path and the
  aligned-features path share a single forward today, which causes the
  redundancy and an extra (incorrect) transpose.

Each of these becomes a Phase 3 ticket with a numerical acceptance test
against the Python reference in `kokoro/`.

---

## 3. Is Kokoro Still the Right Target?

Summary: **yes.** Kokoro-82M remains on the Pareto frontier in April 2026 for
the intersection of (a) small, (b) CPU-only, (c) permissive license, (d) quality,
and (e) architecturally tractable for a pure-Rust from-scratch port. No
alternative that has emerged since Kokoro's December 2024 release better
satisfies all five constraints at once.

### 3.1 Constraints restated

The user's stated goal has very specific consequences:

| Constraint             | Implication                                                                                    |
|------------------------|------------------------------------------------------------------------------------------------|
| **Pure Rust**          | Rules out `ort` (ONNX Runtime), `tract` with C deps, `onnxruntime-rs`, Python FFI, Triton.     |
| **Zero GPU**           | Rules out any model that assumes `fp16`/`bf16`, Flash-Attention, or a GPU-only custom kernel.  |
| **Cross-compiles to WASM** | Rules out any model whose inference depends on `mmap2`, threads, SIMD intrinsics, BLAS, `rayon`, or `std::fs` on the hot path. Weight loading must be architected as "buffer in, buffer out". |
| **Small**              | Soft budget: ≤ 150M params for on-device / browser loading. 80–100M is ideal.                  |
| **Open architecture**  | The paper or reference must exist in enough detail that we can port without guessing.          |

The "pure Rust + WASM" pair is the tight constraint. It forces a from-scratch
implementation of the model, which makes architectural simplicity a
first-order criterion.

### 3.2 Candidate matrix

| Model                 | Params  | License   | Arch                                  | Quality vs Kokoro | Tractable in pure Rust? | Notes                                    |
|-----------------------|---------|-----------|---------------------------------------|-------------------|-------------------------|------------------------------------------|
| **Kokoro-82M v1.0**   | 82M     | Apache-2  | StyleTTS2 + iSTFTNet, Albert phonetic BERT | baseline       | Moderate                | Multi-stage; proven in browser via kokoro.js (ONNX) |
| **Kokoro-82M v1.1**   | 82M     | Apache-2  | same                                  | ≈ baseline        | Moderate                | Minor voice/weight improvements; no arch change |
| **Piper**             | 20–75M  | MIT       | VITS (end-to-end flow + HiFi-GAN)     | −1 MOS            | ✅ High                 | Many ports. Simpler: single forward pass. Lower naturalness. |
| **MeloTTS**           | ~50M    | MIT       | VITS with BERT front-end              | ≈ Piper           | ✅ High                 | Multilingual, similar simplicity        |
| **Matcha-TTS**        | 18–50M  | MIT       | Flow-matching acoustic + HiFi-GAN     | ≈ Piper           | ✅ High                 | Very small, cleanly documented          |
| **Kitten TTS**        | 15M (25MB int8) | Apache | VITS-like, heavily quantized          | −2 MOS            | ✅ High                 | Ultra-edge, noticeably less natural     |
| **StyleTTS2**         | 148M    | MIT       | StyleTTS2 + HiFi-GAN                  | ≈ Kokoro          | Moderate                | Kokoro's parent; larger; does not target edge |
| **Pocket TTS (Kyutai, Feb 2026)** | ~100M | ?       | Undisclosed decoder-only LM-TTS       | ≈ Kokoro          | Unknown                 | No public architecture details; PyTorch-only today |
| **Chatterbox-Turbo**  | 350M    | MIT       | Distilled diffusion, 1-step decoder   | ≈ Kokoro          | ❌ too large            | Needs GPU for real-time                 |
| **F5-TTS**            | 335M    | CC-BY-NC  | Flow matching on DiT                  | ≈ Kokoro          | ❌ too large / NC       | Non-commercial license                  |
| **OuteTTS / Orpheus** | 700M–3B | Apache    | Llama-style LM over audio tokens      | > Kokoro          | ❌ too large            | Requires an LLM runtime; out of scope   |
| **Parler-TTS Mini**   | 600M    | Apache    | T5 + Llama-style LM                   | ≈ Kokoro          | ❌ too large            | Transformer stack too heavy for WASM    |

### 3.3 Why not Piper (the only serious pivot)?

Piper/VITS is the only alternative that is materially simpler to
reimplement than Kokoro:

- **Single end-to-end flow network.** No separate BERT phonetic encoder, no
  separate prosody predictor. Inputs go in one side, mel+waveform comes out
  the other.
- **HiFi-GAN vocoder**, which is well-documented and has many open Rust
  reference ports (notably `candle`'s examples), versus iSTFTNet, which has
  fewer reference implementations.
- **~20M parameters per voice** is genuinely smaller than Kokoro-82M.
- **No AdaIN conditioning tree** to get wrong.

However:

- **Naturalness gap is real**, particularly on long sentences and prosodic
  emphasis. A/B listening tests across the community consistently put Kokoro
  ahead of Piper, and the gap widens on anything above a single sentence.
  This is exactly what StyleTTS2 + Albert phonetic conditioning was designed
  to fix.
- **The existing Ferrocarril investment is Kokoro-shaped.** Phonesis maps to
  Kokoro's phoneme vocabulary. `weight_converter.py`, the metadata schema,
  the voice-file format, the `config.json`, the `plbert` block in
  `Config`, the `IstftnetConfig` — all are Kokoro-specific and would need to
  be replaced wholesale for Piper. The test fixtures assume 24 kHz Kokoro
  output.
- **Switching pessimizes the quality ceiling** for all downstream users of
  Ferrocarril. The whole point of a from-scratch Rust port is to ship the
  good TTS, not a smaller one.
- **Tractability argument for Kokoro is strong enough.** Kokoro-v1 has
  already been shipped in-browser via `kokoro.js` (transformers.js + ONNX,
  running WebAssembly + WebGPU). That is a proof-of-concept that the full
  StyleTTS2 + iSTFTNet inference fits in a browser sandbox at interactive
  speeds. Our job is to match it with a pure-Rust implementation — harder
  than reusing ONNX Runtime, but the feasibility is not in doubt.

**Decision:** continue targeting **Kokoro-82M v1.0** (or v1.1 if weight
conversion is re-run). Revisit if one of the following becomes true:

1. A new small TTS ships with materially better quality AND a simpler
   architecture AND permissive licensing.
2. The iSTFTNet vocoder reimplementation turns out to be intractable under
   the "pure Rust, no BLAS on WASM" constraint. (This is the most plausible
   escape hatch. If it triggers, the fallback is Piper + HiFi-GAN, not a
   completely different model.)

### 3.4 Other architectural notes

- **Sample rate:** Kokoro outputs at 24 kHz. All DSP plumbing in
  `ferrocarril-dsp` must preserve this.
- **Phoneme vocabulary:** 178 tokens including IPA + control symbols. Already
  matched by Phonesis's `PhonemeStandard::IPA` output path.
- **Voice files:** 54 official voices as of v1.0, each `[510, 1, 256]` i.e.
  a sequence-position-indexed style + reference pair. Fixable in Rust
  without touching the model.
- **Weights size:** 82M params × 4 bytes = 328 MB f32. For browser deployment
  we need int8 weight loading (~82 MB) or fp16 (~164 MB). This is a Phase 5
  concern; not in the critical path for correctness.

---

## 4. What Ferrocarril Gets Right Today

Not everything needs throwing out. The following are already load-bearing and
stay:

- **`phonesis` G2P.** Self-contained, Rust, ≥65 k-word dictionary, IPA output,
  tests passing in isolation. Good enough for Phase 2+.
- **`ferrocarril-core::weights_binary::BinaryWeightLoader`**. The split into a
  component-indexed binary-per-tensor format keyed by `metadata.json` is
  sensible and WASM-friendly (one fetch per tensor, no mmap).
- **`weight_converter.py`** produces exactly that format. Validated against
  the 81,763,410-parameter Kokoro-82M checkpoint.
- **Custom STFT/iSTFT** in `ferrocarril-dsp`. Convolution-based, no FFT
  library dependency. Important because FFT libraries with good WASM stories
  are scarce and `realfft`/`rustfft` have SIMD portability quirks.
- **`ferrocarril-dsp::save_wav`** for debug output.
- **`src/model/g2p.rs` `G2PHandler`** wrapping Phonesis for use from the
  main binary.
- **The component scaffolds themselves.** Every layer exists with the right
  names and mostly the right shapes; the failures are in glue code, missing
  setters, and a handful of known-wrong tensor transpositions — *not* in
  the overall decomposition.
- **The 15 integration tests** under `ferrocarril/ferrocarril/tests/`. Most
  are presently dead code because of the compilation break, but they encode
  real acceptance criteria and come back to life as Phase 2 lands.
- **The `kokoro/` Python reference** checked into the repo. Having the exact
  reference alongside the port is invaluable for numerical golden-testing.

---

## 5. Roadmap

### Phase 1 — Consolidate and compile (this turn)

**Goal:** one workspace, one `cargo build` passes, one `cargo test` runs.

- [x] Audit the repo structure and identify the two copies.
- [x] Audit compile state; enumerate the 22 `ferrocarril-nn` errors.
- [x] Audit the small-TTS landscape and decide on a target model.
- [x] Write this `PLAN.md`.
- [x] Delete the dead top-level refactor crates (`ferrocarril-core`,
      `ferrocarril-nn`, `ferrocarril-dsp` at repo root) and the orphan
      top-level `*.rs` test scaffolds.
- [x] Fix `Conv1d`: add `#[derive(Debug)]`, implement `set_weight_norm`,
      `set_bias`, and `LoadWeightsBinary`.
- [x] Fix `AdaIN1d`: add `#[derive(Debug)]`, implement `LoadWeightsBinary`
      (delegating to the inner `Linear` with the `fc.` sub-prefix).
- [x] `cargo build` clean on the nested workspace.
- [x] `cargo test` runs end-to-end (even if a few tests fail — we want the
      harness).

Exit criterion: the nested workspace produces a `ferrocarril` binary and
runs its unit tests without any compilation errors.

### Phase 2 — Canonicalize layout and clean slate docs

**Goal:** one unambiguous place to find things.

- Flatten the nested workspace: promote `ferrocarril/ferrocarril/**` up one
  level so the repo root is the workspace root. Rewrite `path = "../phonesis"`
  to `path = "phonesis"`. The alternative is to leave the nested layout and
  just delete the duplicates at root; that is simpler short-term but more
  confusing long-term.
- Archive `STATUS.md`, `NEXT_STEPS.md`, `IMPLEMENTATION_PROGRESS.md`,
  `FERROCARRIL_TTS_BURNDOWN.md`, `INTEGRATION_SUMMARY.md`, and the
  `CUSTOM_BERT_HANDOVER.md` / `CUSTOM_BERT_IMPLEMENTATION.md` pair under
  `docs/legacy/`. Add a note in each saying "superseded by /PLAN.md and
  /ROADMAP.md".
- Split this document: keep `PLAN.md` as the living "current thinking" doc,
  spin off `docs/ARCHITECTURE.md` with the per-component design, and
  `ROADMAP.md` with the dated checklist.
- Remove `println!` noise and the ad-hoc `INFO:` debug prints from layer
  forwards; replace with a single `tracing` feature-gated debug path (still
  WASM-compatible, `tracing` has a no-std subset).
- Get `clippy` clean or explicitly allow-listed.

### Phase 3 — Numerical correctness per component

**Goal:** every forward pass in Ferrocarril produces tensors that are
numerically within ε of the Python reference on identical inputs, using the
real 81.8M-parameter weights.

Method: for each component, write a golden test that:

1. Loads the real Kokoro weights via `BinaryWeightLoader`.
2. Invokes the Rust component on a fixed phoneme/token input.
3. Compares to a `.npy` fixture extracted from the Python reference under
   `kokoro/` via a small harness script `scripts/extract_golden.py`.
4. Asserts `max_abs_error < 1e-4` and `cosine_similarity > 0.9999`.

Order of attack, cheapest-first, each component blocks the next:

1. **`Linear`** (trivial, validates the load path end-to-end).
2. **`Conv1d`** with and without weight_norm (validates the new
   `set_weight_norm` reconstruction).
3. **`InstanceNorm1d` with `affine=false`** (single line fix but must be
   asserted).
4. **`AdaIN1d`** — fix γ/β slicing and channel-mismatch fallback.
5. **`LSTM` (bidirectional)** — remove silent projection fallback, load
   `*_reverse` weights, verify concatenation order.
6. **`CustomAlbert`** — the biggest gap. Fix config-driven embedding size,
   add the second FFN projection, fix pre-LayerNorm residual, fix attention
   mask convention. Validate against HuggingFace `albert-base-v2` tensor
   shapes pulled from the `bert` component of the converted weights.
7. **`TextEncoder`** — remove silent reshape, verify embedding + CNN + BiLSTM
   chain.
8. **`ProsodyPredictor`** — fix double-applied style, fix energy pooling
   dimension, fix the LSTM transpose pattern. This is the hardest one after
   the BERT port.
9. **`Decoder` / `Generator` / `Source module`** — F0 tensor layout,
   reflection padding direction, AdainResBlk1 upsampling parity,
   `f0_upsamp` transpose.
10. **End-to-end** audio from text, assert similarity vs Python-generated
    WAV via MCD (mel cepstral distortion) or plain RMSE on the log-mel
    spectrogram.

### Phase 4 — Performance and API

- Matrix multiplication: today's naive triple loop in
  `ferrocarril-core::ops::matmul` is the hotspot. Replace with a cache-blocked
  implementation. Keep the same interface. No BLAS dependency.
- Convolution: `Conv1d::conv1d` is also naive; add an im2col + matmul path
  for the common stride=1 case.
- Tensor layout: confirm everything is contiguous row-major and stop
  allocating intermediate `Vec<f32>` for every transpose.
- Thread pool off by default; add `rayon` behind a non-WASM feature gate.
- Publishable crate layout; a single `ferrocarril` crate re-exporting the
  pieces a consumer actually uses.

### Phase 5 — WebAssembly target

- `cargo build --target wasm32-unknown-unknown --no-default-features` clean.
- Feature-gate `memmap2` out; add an alternative `buffer_loader` in
  `ferrocarril-core::weights_binary` that takes `Vec<u8>` blobs directly.
- `wasm-bindgen` crate `ferrocarril-wasm` exposing
  `synthesize(text: &str, voice: &[u8]) -> Vec<f32>`.
- int8/fp16 weight packing to cut browser download size.
- Demo page mirroring the `kokoro.js/demo/` app but hitting the Rust build.

### Phase 6 — Polish

- Multi-language voices (Phonesis already designed for more than English).
- Streaming inference (chunked decoder).
- Documentation on crates.io.
- Benchmarks vs `kokoro.js` and `kokoros` on the same hardware.

---

## 6. End-of-Turn Status Report

This section was rewritten at end-of-turn on 2026-04-06 to reflect what
actually happened, not what was planned.

### 6.1 Delivered — Phase 1 (original scope)

1. **`PLAN.md`** (this document) — canonical analysis, target-model decision,
   and phased roadmap.
2. **Repo consolidated**. Deleted the dead top-level refactor crates
   (`ferrocarril-core/`, `ferrocarril-nn/`, `ferrocarril-dsp/`), the five
   orphan `*.rs` test scaffolds at the repo root, and the five scratch
   binaries under `ferrocarril/ferrocarril/src/bin/`. Added the converted
   weight directories and audio outputs to `.gitignore` so `mutagen` doesn't
   try to sync 341 MB back to the filestore.
3. **`Conv1d`** fixed: `#[derive(Debug)]`, `set_weight_norm` (reconstructs
   `w = g · v / ‖v‖₂` along dim 0), `set_bias`, and full `LoadWeightsBinary`
   that accepts `.weight_g/.weight_v`, `_weight_g/_weight_v`, or plain
   `.weight`, with optional `.bias`.
4. **`AdaIN1d`** fixed: `#[derive(Debug)]`, renamed internal `linear` field
   to `pub fc` to match Kokoro Python naming, added forward-pass shape
   assertions, and added `LoadWeightsBinary` that delegates to the inner
   `Linear` under the `.fc` sub-prefix.
5. **`ConvTranspose1d`** enhanced with `set_weight_norm`, `set_bias`, and a
   weight_norm-aware `LoadWeightsBinary`.
6. **Clean `cargo build`** on the nested workspace. Release build runs in
   ~11 s.
7. **Test harness** compiles and runs. Baseline (before adding the real
   weights directory): 35 passing / 5 weight-dependent failures.

### 6.2 Delivered — Beyond Phase 1

1. **Real Kokoro-82M weights** downloaded from HuggingFace
   (`hexgrad/Kokoro-82M`) and converted to the Ferrocarril binary format via
   the existing `weight_converter.py`. 81,763,410 parameters across 548
   binary files + 54 voice files (326 MB total). The converter and the
   loader both work unchanged.
2. **G2P switched from ARPABET to IPA** in `PhonesisG2P`. Kokoro's vocab is
   IPA-keyed; the previous ARPABET default emitted `HH EH0 L OW0` which did
   not map into the model's vocabulary at all. Now emits
   `h ɛ l oʊ ʊ w ɝ r l d` for "Hello world", matching the reference Python
   phonemizer. Two test assertions were updated from `"HH"` to `"ɛ"`/`"l"`.
3. **BERT attention mask crash fixed**. The old code indexed the mask as
   3D `[B, Q, K]` with `mask.shape()[2]` and inverted the Hugging Face
   convention ("1 = masked"); the new code accepts 2D `[B, seq]` or 3D
   `[B, Q, K]` masks, uses HF convention `1 = visible / 0 = masked`, and
   bounds-checks every access. The 12-layer ALBERT transformer now runs
   end-to-end with real weights.
4. **`DurationEncoder` forward rewritten**. The old code had the LSTM block
   branch transposing x when x was already in BCT format, corrupting the
   tensor layout between block iterations. The new code normalises x to BCT
   before the loop, transposes locally to BTC for each LSTM call, and the
   final return-shape contract matches Python: the encoder genuinely emits
   `[B, T, d_model + style_dim]` because the terminal `AdaLayerNorm` block
   re-adds style and the old Rust assertion that demanded plain `[B, T,
   d_model]` was wrong.
5. **`ProsodyPredictor::forward` rewired to the Python contract**. The old
   code expected `DurationEncoder` to drop style and then manually
   re-concatenated style for the duration LSTM; the new code passes `d_enc`
   directly through the duration LSTM and uses it unchanged for the
   energy-pooling matmul, exactly matching Python `kokoro/modules.py`.
6. **`predict_f0_noise` callers** in `ferro_model.rs` no longer manually
   transpose/truncate the energy-pooled tensor before calling the function.
   `predict_f0_noise` already converts BCF→BFC internally; the caller just
   passes `en` directly.
7. **F0 / noise projection assertions** now read the actual time dimension
   from the projection output instead of the pre-upsample `frames`, because
   the middle F0/N blocks have `upsample=True` and double the time dim.
8. **Decoder uses `istftnet.IstftnetConfig` from the real `config.json`**
   instead of hard-coded `upsample_rates=[8,8]`/`upsample_initial_channel=256`.
9. **New `AdainResBlk1d` class** in `vocoder/adain_resblk1d.rs` that
   faithfully ports Python's decoder-side `AdainResBlk1d`:
   `dim_in`/`dim_out` separation, LeakyReLU(0.2), optional depthwise
   `ConvTranspose1d` pool for upsampling, optional 1×1 learned shortcut
   conv, `(residual + shortcut) * 1/√2` combine, and matching
   `LoadWeightsBinary` paths.
10. **`Decoder` rewired to use `AdainResBlk1d`** instead of the Generator's
    `AdaINResBlock1`. The root-cause bug was that `AdaINResBlock1::new(dim_in + 2, 1024, vec![1,3,5], style_dim)` silently placed `1024` into the
    `kernel_size` slot because the two Python classes have different
    constructor signatures and Rust only had one of them. The Decoder now
    has one encode block and four decode blocks with the correct Kokoro
    shape ladder: `(514 → 1024)`, three `(1090 → 1024)`, one
    `(1090 → 512, upsample=true)`.
11. **`Decoder::forward` rewritten** to match Python's flow exactly, with
    strict fail-fast shape assertions throughout and a hard-failing
    `concat_channels` helper. Fixed weight path names:
    `{prefix}.F0_conv` (capital F), `{prefix}.N_conv` (capital N),
    `{prefix}.asr_res.0` (Python `nn.Sequential` wrapper).
12. **`Generator::forward` F0 shape collapse fixed**. The Python code does
    `f0_upsamp(f0[:, None]).transpose(1, 2)` producing `[B, T_up, 1]`. The
    Rust code was passing `[B, 1, T_up]` directly to `SineGen`, which
    interpreted `shape[1] = 1` as the time dim and collapsed the 15600-sample
    harmonic source to a single sample. Added the missing transpose.
13. **`AdaINResBlock1` Snake1D `alpha` indexing fixed**. The parameter was
    initialized as 1D `[channels]` but the real Kokoro weights ship it as
    `[1, channels, 1]`, so `alpha[&[0]]` panicked on rank mismatch after
    weight loading. The fix reads `alpha[i]` via flat-buffer access and
    applies Snake1D per channel, matching the Python
    `xt + (1/a) * sin(a * xt)**2` semantics correctly.
14. **Generator noise_conv padding off-by-one fixed**: Python uses
    `padding=(stride_f0 + 1) // 2`; Rust was using
    `padding=(kernel_size + 1) / 2 = stride_f0`, producing outputs one
    frame too long for the F0-upsampled STFT branch.
15. **`Generator::forward` reflection padding added**. Python applies
    `nn.ReflectionPad1d((1, 0))` to `x` after the last transpose
    convolution in the upsample loop; the Rust port was missing this,
    leaving `hidden` one frame shorter than `x_source`. Added manual
    reflection pad (`new[0] = old[1]`, shift rest right by 1).

### 6.3 End-to-end inference result

The main binary now produces valid audio output from the real 82M-parameter
Kokoro-82M model loaded through the Rust `BinaryWeightLoader`. Two runs at
end of turn:

```
$ ./target/release/ferrocarril infer \
    --text "Hi" \
    --output /tmp/hi.wav \
    --model ferrocarril_weights \
    --voice af_heart
...
Generated 11400 audio samples
Audio generated and saved to: /tmp/hi.wav
$ file /tmp/hi.wav
/tmp/hi.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit,
mono 24000 Hz
```

```
$ ./target/release/ferrocarril infer \
    --text "Hello world this is a test" \
    --output /tmp/hello.wav \
    --model ferrocarril_weights \
    --voice af_heart
...
Generated 31200 audio samples
```

Both complete without panics or errors. The WAV files are non-zero 16-bit
PCM mono at 24 kHz, which is Kokoro's native sample rate. `/tmp/hi.wav`'s
sample dynamic range is currently narrow (`min=1072`, `max=1638`,
`abs_mean≈1638`) — see §6.4 for why the audio is not yet intelligible.

### 6.4 What still needs doing — Phase 3 numerical correctness

The **architectural pipeline is complete and end-to-end**, but the audio
quality is still Phase 3 work. Known remaining numerical issues, roughly in
order of impact on audio quality:

- **`CustomAlbert` transformer**: still uses the old placeholder forward
  pass with a single shared layer, hard-coded `embedding_size = 128`, an
  incorrect residual connection pattern, and no second FFN projection.
  `hidden_size = 768` and the embedding hidden mapping use the wrong input
  dim. None of these block the shape pipeline but the output is
  not numerically close to Python's Albert. This is the biggest single
  quality dial.
- **`LSTM` bidirectional**: needs `pack_padded_sequence` or a frozen
  hidden state past `input_lengths`. Currently runs over padding tokens.
- **`AdaIN1d`**: `InstanceNorm1d` is configured with `affine=false` which
  matches Kokoro, but the γ/β splitting in `AdaIN1d::forward` needs a
  final check against Python's `torch.chunk(h, chunks=2, dim=1)` semantics.
- **Voice embedding split**: the current loader takes the middle position
  of the `[510, 1, 256]` tensor and duplicates it into a `[1, 256]`
  reference/style pair. Python's `KModel` does something more specific
  (positional indexing by phoneme length). Result: all voices sound the
  same right now.
- **`weight_converter.py` output** drops bias tensors on some
  weight_norm'd convs. The Rust loader then silently leaves the bias at its
  random-init value. This shows up as warnings like
  "Warning: Failed to load f0_conv weights: Parameter 'module.F0_conv.bias'
  not found" in the inference log.
- **`Generator::forward`** harmonic-source merging: `self.source.forward`
  returns `[B, T]` (2D) but the Generator expects `[B, 1, T]` in later
  steps. The current code papers over this with a reshape.
- **STFT hop-size consistency**: Python uses `CustomSTFT` with specific
  padding/window conventions that `ferrocarril-dsp::stft::CustomSTFT` has
  not been numerically validated against.
- **`conv_post` weight loading**: the Generator's final conv_post weights
  go through a "possible path" fallback that currently only succeeds in
  half the cases. The specific weight-norm path in Kokoro's state dict is
  `decoder.module.generator.conv_post.weight_g/_v` but the current code
  also tries underscored variants.

None of these are blockers for the pipeline *running*. They are blockers
for the pipeline *producing intelligible speech*. See §7 for the order to
attack them.

### 6.5 Test suite baseline

`cargo test --no-fail-fast` on a fresh release build:

```
20 passed / 3 failed / 1 ignored
```

The three failures are all in the decoder/end-to-end paths and fail at
runtime because they perform numerical assertions on the Generator's audio
output statistics (variance, max-abs value, non-zero fraction). Those
assertions were calibrated against hand-rolled synthetic inputs before the
Generator was ever exercised end-to-end; they will need to be re-tightened
once §6.4 is addressed. The ignored test is one of the two decoder tests
marked `#[ignore]` pending the Generator numerical-correctness pass.

Failing tests:
- `test_decoder_with_real_kokoro_weights` — hard-coded critical-weights list
  still contains old pre-rename paths.
- `test_decoder_audio_generation` — runtime numerical assertions on audio
  variance.
- `end_to_end_test::test_end_to_end_tts_synthesis` — runtime numerical
  assertions on full-pipeline audio.

### 6.6 Commit log and empirical audio state

Twenty-eight commits landed this turn, starting with "consolidation and
compile" and ending with per-utterance voice selection. The main binary
now successfully runs end-to-end Kokoro inference with the real 82M
parameter checkpoint and produces valid 24 kHz mono WAV output.

Empirical audio stats from the most recent two inference runs:

| Run                       | Text          | Duration | Sample range   | abs_mean | distinct |
|---------------------------|---------------|----------|----------------|----------|----------|
| pre-voice-fix `hello.wav` | "Hello world" | 0.800s   | `-1058..1291`  | 240.8    | 1076     |
| post-voice-fix `hello_voice.wav` | "Hello world" | 0.975s | `-168..367`  | 45.9     | 376      |

Both are real varying PCM waveforms, not silence or saturated constants.
The audio is **not yet intelligible speech**: the Albert BERT forward pass,
several LSTM var-length edge cases, and the numerical correctness of the
AdaIN γ/β splitting still need per-component validation against the Python
reference in `kokoro/`. See §7 for the ordered next-session checklist.

Commit order (roughly):

1. PLAN.md scaffold, delete broken top-level refactor crates, delete stray
   top-level `.rs` scaffolds and the `src/bin/` scratch binaries.
2. `Conv1d` — `#[derive(Debug)]`, `set_weight_norm`, `set_bias`,
   `LoadWeightsBinary`.
3. `AdaIN1d` — `#[derive(Debug)]`, `pub fc`, `LoadWeightsBinary`, forward
   shape assertions.
4. BERT naming aliases (`CustomAlbert` / `CustomAlbertConfig`), add
   `embedding_size` and `dropout_prob` fields.
5. Two `bert.forward` call-site arg fixes in `ferro_model.rs`.
6. Update test imports to use public `ferrocarril_nn::bert` re-exports.
7. Update test `BertConfig` literals to include `embedding_size`.
8. Rename `AdaIN1d.linear` to `pub fc` + shape assertions (supports
   `tests/adain_test.rs`).
9. G2P default standard switched from ARPABET to IPA.
10. Two G2P test assertions updated from `"HH"` to IPA `"ɛ"`.
11. `.gitignore` additions for weight directories and audio outputs to
    avoid mutagen sync overload.
12. BERT `MultiHeadAttention::forward` mask-application fix: support 2D
    `[B, seq]` masks with HF convention `1 = visible`.
13. DurationEncoder loop layout normalized to BCT so the LSTM block no
    longer mis-transposes after AdaLayerNorm.
14. DurationEncoder final output shape contract updated to
    `[B, T, d_model + style_dim]` matching Python.
15. ProsodyPredictor `forward` rewired: feed `d_enc` directly into
    `dur_lstm` and energy pool, no manual style reattachment.
16. Two `predict_f0_noise` callers in `ferro_model.rs` stop mangling `en`
    before the call.
17. F0/Noise projection assertions account for the `upsample=True` middle
    block doubling the time dim.
18. Decoder constructed with real `config.istftnet.*` values instead of
    hardcoded `[8,8]`/`256`/`[16,16]`/`16`/`4`.
19. New `AdainResBlk1d` class created in
    `vocoder/adain_resblk1d.rs`, enhanced `ConvTranspose1d` with
    `set_weight_norm`, `set_bias`, and weight-norm-aware loading.
20. Decoder rewired to use `AdainResBlk1d`, strict shape validation and
    fail-fast `concat_channels`, decoder tests updated and marked `#[ignore]`
    pending the Generator Phase 3 work.
21. Generator F0 shape collapse fixed (missing `.transpose(1, 2)` after
    `f0_upsamp`).
22. `AdaINResBlock1` Snake1D per-channel alpha indexing fixed (real Kokoro
    ships alpha as `[1, C, 1]`, not `[C]`).
23. Generator `noise_convs` padding off-by-one fixed: `(stride_f0 + 1) / 2`
    not `(kernel_size + 1) / 2`.
24. Generator `reflection_pad((1, 0))` added after the last transpose
    convolution.
25. PLAN.md empirical status report + Next Session Notes.
26. Vocab parser `chars().count() == 1` fix so multi-byte IPA characters
    are no longer silently dropped; phoneme tokenizer rewritten to iterate
    over characters instead of whitespace-splitting + first-char.
27. Generator `m_source`/`l_linear`/`conv_post` weight paths fixed to
    match Python Kokoro.
28. Voice pack indexing: `load_voice` now returns the raw `[510, 256]`
    pack and `infer_with_phonemes` picks row `min(num_phonemes - 1, 509)`
    per Python `ref_s = pack[len(ps) - 1]`, accounting for the BOS/EOS
    offset.

The remaining quality levers, ordered by expected impact on audio
intelligibility (see §6.4 for details):

1. CustomAlbert transformer numerical correctness (biggest dial).
2. LSTM bidirectional variable-length sequence handling.
3. AdaIN γ/β splitting verification against Python `torch.chunk`.
4. STFT / iSTFT numerical parity against Kokoro's `CustomSTFT`.
5. Per-component golden fixtures extracted from the Python reference in
   `kokoro/` for drop-in numerical diffing.

---

## 7. Next Session Notes

When resuming:

1. **Validate each component numerically against Python.** Use the in-tree
   `kokoro/` reference. Write a small Python helper that loads the real
   weights, runs `Kokoro` forward to a fixed phoneme input, and dumps the
   intermediate tensors at every layer boundary as `.npy`. Load those
   fixtures from Rust tests and compare with `max_abs_error < 1e-4`. Attack
   components in order of impact (biggest first):
   1. `CustomAlbert` forward (the biggest quality dial).
   2. `TextEncoder` CNN + BiLSTM chain.
   3. `LSTM` bidirectional variable-length handling.
   4. `ProsodyPredictor` end-to-end (already structurally aligned, but
      γ/β splitting and F0/N block internals need a numerical check).
   5. Voice embedding splitter.
   6. Decoder `AdainResBlk1d` forward (already structurally correct, verify
      pool transpose conv weight_norm reconstruction).
   7. `Generator` upsample loop and the source module sine-wave synthesis.
   8. STFT / iSTFT in `ferrocarril-dsp`.

2. **Clean up the warning output.** `cargo build --release` currently emits
   26 warnings in `ferrocarril-nn` (unused fields, dead code, stale
   `println!` debug prints). Strip the noisiest of the `println!`s and
   gate the rest behind a `debug` feature or a `tracing` subscriber.

3. **Re-enable the 3 failing tests** once their numerical assertions make
   sense again. Consider converting them from strict pass/fail to
   similarity-vs-reference tests.

4. **Flatten the nested workspace** (`ferrocarril/ferrocarril/` up to
   `ferrocarril/`). This is a pure refactor and was deferred to avoid
   churning every file path while the Phase 3 work is in flight. Do it
   after the numerical correctness pass so the diffs don't overlap.

5. **Start on Phase 5 (WASM).** `memmap2` is used in
   `ferrocarril-core::weights_binary`; add an alternative
   `buffer_loader` API that takes owned `Vec<u8>` blobs so `cargo build
   --target wasm32-unknown-unknown --no-default-features` can link. Expose
   a `wasm-bindgen` crate `ferrocarril-wasm` with a `synthesize(text,
   voice_bytes) -> Vec<f32>` surface.