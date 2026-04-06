# Ferrocarril Session Handoff — April 2026

> **Read this first when you resume.** It's a self-contained pick-up
> document. The canonical long-form analysis is still `PLAN.md`; this
> handoff is the tactical next-steps summary for the Phase 3 numerical
> correctness work that was in flight when the session paused.

---

## 1. What You're Looking At

Ferrocarril is a pure-Rust, zero-GPU, WASM-targetable port of the
Kokoro-82M text-to-speech model (StyleTTS2 + iSTFTNet architecture, 82M
parameters, Apache-2.0). The repo contains:

- `ferrocarril/` — the canonical nested workspace (Rust, `[workspace]` root
  at `ferrocarril/Cargo.toml`). Crates:
  - `ferrocarril-core` — tensors, ops, binary weight loader, config,
    `PhonesisG2P`, `FerroError`.
  - `ferrocarril-dsp` — custom STFT/iSTFT, windows, WAV I/O.
  - `ferrocarril-nn` — linear, conv, lstm, adain, conv_transpose,
    text_encoder, prosody (duration encoder, resblk1d), vocoder
    (generator, decoder, adain_resblk1, adain_resblk1d, sinegen,
    source_module), bert (custom Albert).
  - `ferrocarril` (root crate) — main `ferrocarril infer` binary.
- `phonesis/` — G2P library, complete, tests pass in isolation.
- `kokoro/` — in-repo Python reference (Kokoro v1.0 source tree).
- `scripts/` — Phase 3 Python golden-reference harnesses.
- `tests/fixtures/` — `.npy` golden fixtures written by the Python
  harnesses for the Rust golden tests to diff against.
- `ferrocarril_weights/` — the converted 341 MB Kokoro-82M binary
  weight pack. Gitignored. Reproduced on demand via
  `python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights`.

`PLAN.md` at the repo root documents the whole project plan, the target-
model decision (Kokoro is still the right target in 2026 for
pure-Rust/WASM in this size class), and the phased roadmap. This handoff
assumes you have read §1–§6 of `PLAN.md` or can re-read them later; it
does not repeat that material.

---

## 2. Current Status Snapshot (end of session 2)

| Layer                | Status          | Tolerance vs Python reference         |
|----------------------|-----------------|---------------------------------------|
| G2P (Phonesis)       | ✅ working       | IPA output, non-numerical              |
| Weight loading       | ✅ working       | weight_norm reconstruction verified    |
| `CustomBert` (plbert)| ✅ **VALIDATED** | **~2e-6** max abs diff on 7-token input |
| `BertEncoder` (proj) | 🟡 probably OK   | Single `nn.Linear(768, 512)`, no dedicated test yet |
| `TextEncoder`        | ✅ **VALIDATED** | **~1e-6** max abs diff (see §6 for the LayerNorm fix this session) |
| `LSTM` (bidirectional) | 🟡 used OK in TextEncoder | No var-length (`pack_padded_sequence`) handling yet |
| `DurationEncoder`    | ❌ **FAILING**   | **0.62** first-time drift, **1.47** last-time drift. ACTIVE DEBUG TARGET. |
| `ProsodyPredictor`   | ⛔ blocked       | Depends on DurationEncoder             |
| `Decoder` / `AdainResBlk1d` | 🟡 runs       | Structural only, no numerical check    |
| `Generator` / iSTFTNet | 🟡 runs        | Structural only, no numerical check    |

**End-to-end inference runs and produces a valid 24 kHz mono WAV.** The
audio is not intelligible yet because of the `DurationEncoder` drift
(and whatever comes after it). Current audio stats on "Hello world":
`duration=0.975s, min=-168, max=367, abs_mean=45.9`.

Build and test suite:

```
$ cargo build --release   # ~11 s clean
$ cargo test --release --no-fail-fast
20 passed / 3 failed / 1 ignored
```

The 3 failing tests are all the runtime-numerical-quality tests in the
decoder/end-to-end paths. The `#[ignore]`'d test is
`test_prosody_predictor_style_conditioning`. The **BERT golden test is
passing without `#[ignore]`** at tolerance 1e-4.

**Build sandbox:** use the privileged `build` sandbox (Ubuntu, 8 CPU,
16 GB). The default sandbox is too small and got reset by mutagen when
the 341 MB weights tried to sync. `.gitignore` now blocks that but use
the `build` sandbox anyway for safety.

---

## 3. What Shipped This Session (Phase 3 Progress)

38+ commits landed on top of the Phase 1 foundation. Ordered by theme:

### 3.1 Pipeline shape correctness (landed before validation phase)
- `Conv1d` / `ConvTranspose1d`: weight_norm reconstruction, `set_bias`,
  full `LoadWeightsBinary` impls.
- `AdaIN1d`: `pub fc` rename, shape asserts, `LoadWeightsBinary` that
  delegates to the inner `Linear`.
- `BertConfig`: new `embedding_size` field; `CustomAlbert` /
  `CustomAlbertConfig` aliases.
- `MultiHeadAttention::forward`: 2D `[B, Seq]` mask handling with HF
  convention (`1 = visible`).
- `DurationEncoder::forward`: loop layout normalised to BCT, final return
  shape contract fixed to `[B, T, d_model + style_dim]` matching Python.
- `ProsodyPredictor::forward`: feed `d_enc` directly into `dur_lstm` and
  energy pool; caller drops manual style expansion and caller-side
  transpose of `en` before `predict_f0_noise`.
- F0/Noise projection assertions account for `upsample=True` doubling.
- `Decoder`: new `AdainResBlk1d` class (correct Python-matching class
  with `dim_in`/`dim_out` + depthwise `pool`), `Decoder::forward` rewired
  to match Python, strict fail-fast `concat_channels`, tests updated and
  marked `#[ignore]` pending Generator Phase 3.
- `Generator`: F0 shape collapse fixed (missing `.transpose(1, 2)` after
  `f0_upsamp`), Snake1D per-channel alpha indexing fixed (alpha is
  `[1, C, 1]` in real weights, not `[C]`), `noise_convs` padding off-by-
  one fixed (`(stride_f0 + 1) / 2`, not `(kernel_size + 1) / 2`),
  `ReflectionPad1d((1, 0))` added after last transpose conv.
- Weight path fixes: `module.F0_conv` / `module.N_conv` (capital F/N),
  `module.asr_res.0` (Sequential wrapper), `m_source.l_linear` in the
  source module.
- Voice pack indexing: `load_voice` returns raw `[510, 256]` pack,
  `infer_with_phonemes` picks row `min(num_phonemes - 1, 509)` per
  Python `ref_s = pack[len(ps) - 1]`, accounting for BOS/EOS offset
  (`seq_len - 3`).
- Vocab parser: `chars().count() == 1` instead of `key.len() == 1` so
  multi-byte IPA chars like `ɛ`/`ʊ` are no longer silently dropped.
- Phoneme tokenizer: rewritten to iterate over Unicode scalars instead
  of whitespace-splitting and taking the first char.

### 3.2 Phase 3 validation infrastructure (new this session)

**Python golden-reference harnesses** under `scripts/`:

1. **`scripts/validate_bert.py`**: downloads Kokoro-82M `.pth`, extracts
   the `bert` sub-state-dict, instantiates `transformers.AlbertModel`
   with Kokoro's plbert config, runs a fixed input
   `[0, 50, 86, 54, 59, 135, 0]` (BOS + h/ɛ/l/o/ʊ + EOS), and dumps
   `last_hidden.npy` + all 13 intermediate `hidden_NN.npy` files under
   `tests/fixtures/bert/`.
2. **`scripts/validate_text_encoder.py`**: faithful pure-PyTorch
   reimplementation of Kokoro's `TextEncoder` (no `kokoro` package
   import, no `misaki`/`phonemizer` deps), loads `text_encoder`
   sub-state-dict, runs same fixed input, dumps `output.npy` under
   `tests/fixtures/text_encoder/`.
3. **`scripts/validate_kmodel.py`**: **the most important harness**.
   Uses PyTorch hooks to instrument the full Kokoro `KModel` (or a
   standalone re-run of each sub-component) and dumps intermediate
   tensors for **every layer** in the TTS pipeline. Fixtures go under
   `tests/fixtures/kmodel/`: `bert`, `bert_encoder`,
   `predictor_text_encoder` (DurationEncoder output `[1, 7, 640]`),
   `predictor_lstm` (Duration LSTM output `[1, 7, 512]`),
   `predictor_duration_proj` (`[1, 7, 50]`), `text_encoder` (`[1, 512, 7]`),
   `pred_dur` (`[18, 2, 2, 3, 4, 9, 18]`), decoder outputs, and the
   final `audio` `[33600]`. Uses voice `af_heart` row 4 (phoneme count 5
   → index 4 after BOS/EOS stripping).

**Rust golden tests** under `ferrocarril/ferrocarril/tests/`:

1. `bert_golden_test.rs` — **passing** at tolerance 1e-4, actual drift
   ~2e-6. Runs in normal `cargo test`, will catch any BERT regression.
2. `text_encoder_golden_test.rs` — **passing** at tolerance 1e-4,
   `#[ignore]`d until we're confident, enable with `--ignored`. Verifies
   TextEncoder numerical correctness on the same 7-token input.
3. `duration_encoder_golden_test.rs` — **failing**. See §5.

### 3.3 The one substantive numerical bug fixed this session

**`text_encoder.rs` LayerNorm**: the Rust code was computing
`(x - mean) / std` across the **time dimension** for each channel (i.e.
an InstanceNorm on BCT-layout tensors). Python's `LayerNorm` in
`kokoro/modules.py` does `x.transpose(1, -1)` (`[B, C, T]` → `[B, T, C]`)
and then `F.layer_norm(x, (channels,))`, which normalizes across the
**channel dimension** for each `(batch, time)` position. Those two
operations produce completely different outputs — the fix was to
normalize across channels-per-time-step for each batch. After this fix,
the `text_encoder_golden_test` Rust output matches the Python reference
to ~1e-6 on every dim tested.

**Lesson:** Kokoro's custom `LayerNorm` class hides a
`transpose(1, -1) → F.layer_norm → transpose(1, -1)` dance. Every time
you see `layer_norm` applied to a `[B, C, T]` tensor in a Python port,
check whether the caller has already transposed. This exact pattern
shows up in at least `DurationEncoder`'s `AdaLayerNorm` (next target)
and likely the `AlbertLayer` full_layer_layer_norm too (which already
works because BERT is BTC layout throughout, so this doesn't bite).

---

## 4. Active Debug Target: `DurationEncoder` Numerical Drift

### 4.1 What the test says

Reproduce:

```
$ cd ~/ferrocarril/ferrocarril
$ cargo test --release --test duration_encoder_golden_test -- --ignored --nocapture
```

Expected to fail with output roughly:

```
DurationEncoder Python reference:
  first time, first 8 dims: [-0.32289, 0.19647, -0.03532, -0.02273, 0.37823, -0.34230, 1.77888, -0.21553]
  last  time, first 8 dims: [1.15667, 0.23253, -1.92759, 0.00752, 0.39949, -0.48881, 1.31379, -0.17638]
  mean |x|                : 0.440264

DurationEncoder Rust output:
  first time, first 8 dims: [-0.29620, 0.23453, -0.03288, -0.09144, 0.40526, -0.19921, 2.40312, -0.18004]
  last  time, first 8 dims: [ 0.74... , ..., ..., ..., ..., ..., ..., ...]   # re-run to capture exact values
  mean |x|                : ~0.415

Diff vs Python reference (tolerance 0.001):
  first time max |Δ|: 0.624236
  last  time max |Δ|: 1.473824
  mean|x| Δ          : 0.025307
```

The shape is correct (`[1, 7, 640]` = `[B, T, d_model + style_dim]`),
which means the Phase 1 shape-contract rewrite of `DurationEncoder` was
right. The bug is **pure numerical** — individual values are close in
sign and order-of-magnitude to Python but drift by up to 0.62 on
individual elements.

### 4.2 What we know about the divergence shape

- First time step, dim 6: Rust **2.403** vs Python **1.779** → drift
  **0.624**. That's one of the biggest single-element drifts.
- First time step, dim 3: Rust **-0.091** vs Python **-0.023** → drift
  **0.068**.
- Most other dims in the first time step: drift **0.02–0.15**.
- Last time step drifts are worse (up to 1.47), suggesting the error
  compounds through the LSTM + AdaLayerNorm stack across time.
- `mean|x|` is very close (0.415 vs 0.440, drift 0.025), so the overall
  scale is roughly right — this is not a complete blowup.

These three features together (same sign, same magnitude band, drift
compounds with time, biggest single drift is a large-magnitude element)
strongly suggest **either**:

1. **A LayerNorm / AdaLayerNorm normalization-axis bug** (same shape as
   the TextEncoder bug fixed earlier this session, just in a different
   file). If `AdaLayerNorm` is normalizing across the time dim instead
   of the channel dim, or applying the `(1 + gamma) * x + beta` scaling
   with gamma/beta from the wrong slice of the `style_fc` output, you
   get exactly this "almost right but drifting by 5–20%" pattern.
2. **The style concatenation re-entry after AdaLayerNorm** is out of
   order. After each `AdaLayerNorm`, the loop body concatenates style
   back onto x along the channel dim. If the order is `[style, x]`
   instead of `[x, style]` (or the gamma/beta split in `AdaLayerNorm`
   is swapped: reading dims `[:C]` as gamma when it should be `[:C]` as
   beta), the LSTM sees a correctly-sized but wrongly-ordered input
   and the drift compounds through the 3 LSTM/Ada pairs.
3. **Bidirectional LSTM forward/reverse ordering**. The Rust `LSTM`
   bidirectional path concatenates forward and backward hidden states.
   If the backward output is not time-reversed before concatenation
   (PyTorch does this automatically; our implementation might not),
   you get a pattern that is "close" because the forward half is right
   but diverges because the backward half is backwards-in-time.

### 4.3 Suspects in priority order

1. **`AdaLayerNorm::forward` in `ferrocarril-nn/src/prosody/duration_encoder.rs`** (around line 60+ based on the file I had open). Check:
   a. It uses `F.layer_norm`-equivalent normalization along the channel dim (Python's `LayerNorm` in `kokoro/modules.py` with the transpose dance).
   b. The style fc output shape is `[B, 2C]` and the split order is
      `gamma = [:C]`, `beta = [C:]` — matching Python's
      `torch.chunk(h, chunks=2, dim=1)`.
   c. The scaling is `(1 + gamma) * x + beta` (Python's
      `x = (1 + gamma) * x + beta`), not plain `gamma * x + beta`.
2. **`LSTM::forward_batch_first` for `bidirectional=True`** in
   `ferrocarril-nn/src/lstm.rs`. Check:
   a. The backward LSTM iterates the input from right-to-left and
      writes its output **time-reversed** before concatenating with the
      forward LSTM's output. Concretely: `backward_output[t] =
      backward_lstm_step(t)`, then `concat[t] = [fwd[t], bwd[T-1-t]]`.
      If you skipped the reversal, the drift looks exactly like §4.2.
   b. The reverse-direction weights (`weight_ih_l0_reverse`,
      `weight_hh_l0_reverse`, `bias_ih_l0_reverse`, `bias_hh_l0_reverse`)
      are being used for the backward pass, not the forward-direction
      weights.
3. **Style re-concatenation inside `DurationEncoder::forward`** in
   `ferrocarril-nn/src/prosody/duration_encoder.rs`. After
   `AdaLayerNorm`, x is BCT `[B, d_model, T]`. The loop re-adds style
   along channel dim to get back to `[B, d_model + style_dim, T]`.
   Check the concat order: Python does `torch.cat([x, s.permute(1, 2, 0)], axis=1)` — `x` first, then style. The Rust
   `concat_channels_bct([&x, &style_tensor])` helper must preserve that
   order.

### 4.4 How to drive the investigation

The reason `validate_kmodel.py` dumped every layer's output is so you
can bisect. Concrete steps:

1. **Extract the AdaLayerNorm output** for each of the three
   AdaLayerNorm blocks. You can run a Python hook-based dump — the
   `validate_kmodel.py` script already has the pattern; just add three
   more hooks on `predictor.text_encoder.lstms[1]`, `lstms[3]`, and
   `lstms[5]` (which are the `AdaLayerNorm` instances in Python).
   Python stores `LSTM, Ada, LSTM, Ada, LSTM, Ada` in `lstms`, so the
   Ada instances are at indices 1, 3, 5. Save the BTC input and the
   BTC output of each Ada.
2. **Write a focused Rust test** that loads the block-by-block fixtures
   and tests `AdaLayerNorm::forward` in isolation:
   - Load `ada_1_input.npy` as the input tensor.
   - Load the real style (from `kmodel/ref_s.npy` or voice pack row 4).
   - Run `adaln.forward(&input, &style)`.
   - Diff against `ada_1_output.npy`.
   - Repeat for blocks 3 and 5.
3. If AdaLayerNorm matches Python, the bug is in the LSTM path.
   Repeat the same approach for `predictor.text_encoder.lstms[0/2/4]`,
   which are the LSTM instances.
4. Once the single bad layer is identified, eyeball the Rust
   implementation against Python and fix the normalization axis,
   split order, activation, or concat order.

---

## 5. The Rust Golden Tests Already In Place

| Test file                          | Enabled?          | Status            | Tolerance |
|------------------------------------|-------------------|-------------------|-----------|
| `bert_golden_test.rs`              | Always            | Passing ~2e-6     | 1e-4      |
| `text_encoder_golden_test.rs`      | `#[ignore]`       | Passing ~1e-6     | 1e-4      |
| `duration_encoder_golden_test.rs`  | `#[ignore]`       | **FAILING 0.62**  | 1e-3      |

Pattern to follow when you add the next golden test: load converted
weights, build the module via its public constructor, load weights via
`LoadWeightsBinary::load_weights_binary`, run the fixed input, compare
a few hand-picked elements AND the global `mean|x|` against hard-coded
Python reference values, print a diff table, then assert on the
tolerance. Mark `#[ignore]` during bring-up, remove `#[ignore]` once
the test passes.

The `find_weights_path()` helper in each test searches
`../ferrocarril_weights`, `ferrocarril_weights`,
`../../ferrocarril_weights`. If none of those exist, make sure the
weights are converted via the top-level
`python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights`
run from the repo root.

---

## 6. Files to Open First On Resumption

In priority order:

1. `ferrocarril/ferrocarril/ferrocarril-nn/src/prosody/duration_encoder.rs`
   — contains `DurationEncoder`, `AdaLayerNorm`, the loop layout fixed
   earlier. This is the file with the bug.
2. `ferrocarril/kokoro/kokoro/modules.py` — the authoritative Python
   reference for `AdaLayerNorm`, `DurationEncoder`, `LayerNorm`, and
   `TextEncoder`. Side-by-side read against (1) is the fastest path to
   the bug.
3. `ferrocarril/ferrocarril/ferrocarril-nn/src/lstm.rs` — for the
   bidirectional-LSTM reversal hypothesis in §4.3.2.
4. `ferrocarril/ferrocarril/tests/duration_encoder_golden_test.rs` —
   the failing test. Note the exact reference values and the fixture
   paths it loads.
5. `ferrocarril/scripts/validate_kmodel.py` — the full-model Python
   harness. Extend this with the per-block AdaLayerNorm hooks per
   §4.4.
6. `ferrocarril/ferrocarril/tests/fixtures/kmodel/` — the .npy fixtures
   already dumped. Contents include
   `predictor_text_encoder.npy` (DurationEncoder output),
   `predictor_lstm.npy`, `predictor_duration_proj.npy`,
   `text_encoder.npy`, `bert.npy`, `bert_encoder.npy`, `ref_s.npy`,
   `pred_dur.npy`, decoder outputs, final `audio.npy`.

---

## 7. How to Re-Establish the Sandbox on Resumption

The `build` sandbox may still be alive, but the default has been shut
down. To recreate everything from scratch:

```bash
# 1. Privileged/high-spec sandbox (don't use default, it gets mutagen-reset by the 341 MB weights)
#    create_sandbox "build" privileged=False cpu_count=8 memory_gb=16

# 2. Python deps for the validation harnesses
pip3 install --break-system-packages --quiet torch huggingface_hub numpy transformers scipy loguru

# 3. Convert the real Kokoro weights (~5 min with HF CDN)
cd ~/ferrocarril
python3 weight_converter.py --huggingface hexgrad/Kokoro-82M --output ferrocarril_weights

# 4. Symlinks so every hard-coded test path resolves
ln -sfn ~/ferrocarril/ferrocarril_weights ~/ferrocarril/ferrocarril/ferrocarril_weights
ln -sfn ~/ferrocarril/ferrocarril_weights ~/ferrocarril/ferrocarril/real_kokoro_weights
ln -sfn ~/ferrocarril/ferrocarril_weights ~/ferrocarril/real_kokoro_weights
sudo mkdir -p /home/sandbox
sudo ln -sfn ~/ferrocarril/ferrocarril_weights /home/sandbox/ferrocarril_weights

# 5. Copy the config.json that the binary expects next to its cwd
cp ~/ferrocarril/ferrocarril_weights/config.json ~/ferrocarril/ferrocarril/config.json

# 6. Baseline build
cd ~/ferrocarril/ferrocarril
cargo build --release          # ~11 s

# 7. Baseline test
cargo test --release --no-fail-fast   # expect 20 pass / 3 fail / 1 ignored

# 8. Re-run the validation harnesses to refresh fixtures
cd ~/ferrocarril
python3 scripts/validate_bert.py
python3 scripts/validate_text_encoder.py
python3 scripts/validate_kmodel.py

# 9. Re-run the failing DurationEncoder test to confirm you're at the same starting point
cd ~/ferrocarril/ferrocarril
cargo test --release --test duration_encoder_golden_test -- --ignored --nocapture
```

---

## 8. Phased Plan Going Forward

Short term (the thing you are doing when you pick this up):

1. Fix the `DurationEncoder` numerical drift per §4.
2. Unignore `duration_encoder_golden_test.rs` and tighten tolerance to
   1e-4.
3. Unignore `text_encoder_golden_test.rs` too — it already passes.
4. Add a `bert_encoder_golden_test.rs` (very easy, it's just a single
   `Linear(768, 512)`).
5. Add a `prosody_predictor_golden_test.rs` that exercises the full
   prosody forward including duration + energy pooling.

Medium term:

6. Add a decoder golden test (`decoder_golden_test.rs`) using the
   `kmodel` fixtures for `asr`, `f0`, `n`, `s` inputs and the
   decoder output.
7. Add a generator golden test against the final audio `[33600]`.
8. Once all components pass 1e-4 golden tolerance, re-enable the three
   currently-failing runtime tests in `decoder_real_weights_test.rs`
   and `end_to_end_test.rs` and tighten their quality thresholds.
9. The audio should be intelligible at this point. Manually listen to
   `/tmp/hi.wav` and `/tmp/hello.wav`.

Long term (Phase 5, WASM):

10. Feature-gate `memmap2` out; add an alternate `buffer_loader` for
    `ferrocarril-core::weights_binary` that takes owned `Vec<u8>` blobs.
11. `cargo build --target wasm32-unknown-unknown --no-default-features`
    clean.
12. `wasm-bindgen` crate `ferrocarril-wasm` exposing
    `synthesize(text: &str, voice_bytes: &[u8]) -> Vec<f32>`.
13. Int8/fp16 weight packing to cut browser download size from ~320 MB
    to ~80–160 MB.
14. Demo page mirroring `kokoro.js/demo/` but hitting the Rust build.

---

## 9. Things Not To Do

- **Don't** rewrite BERT. It's validated to 2e-6; any change is a
  regression. `bert_golden_test` will catch this.
- **Don't** touch `TextEncoder` except to add per-component tests. The
  LayerNorm fix landed this session brought it to ~1e-6 parity.
- **Don't** try to "simplify" the `AdainResBlk1d` class. It's a
  faithful port of Python's `AdainResBlk1d` and deleting any of its
  pieces will re-introduce shape mismatches.
- **Don't** introduce silent fallbacks of any kind. The codebase's
  design discipline is fail-fast: `assert_eq!` on every tensor shape
  contract, `FerroError` on every weight-loading ambiguity, no
  "use minimum dimensions" clamping. The existing silent fallbacks in
  the older code have been load-bearing sources of bugs and are being
  removed as they're encountered.
- **Don't** delete the `[510, 1, 256]` voice pack indexing logic.
  `load_voice` returns the raw pack; `infer_with_phonemes` indexes by
  `seq_len - 3` to match Python's `len(ps) - 1` after BOS/EOS insertion.

---

## 10. Quick Reference Data

**Canonical test input:**
`input_ids = [0, 50, 86, 54, 59, 135, 0]` (BOS, h, ɛ, l, o, ʊ, EOS).
This is the input every golden test uses. `num_phonemes = 5` →
`voice row_idx = 4`.

**Voice:** `af_heart` — `ferrocarril_weights/voices/af_heart.bin` →
`[510, 256]` flat tensor after the loader rewrite.

**Python reference DurationEncoder first-time first-8 dims:**
`[-0.32289, 0.19647, -0.03532, -0.02273, 0.37823, -0.34230, 1.77888, -0.21553]`

**Python reference DurationEncoder last-time first-8 dims:**
`[1.15667, 0.23253, -1.92759, 0.00752, 0.39949, -0.48881, 1.31379, -0.17638]`

**Python reference DurationEncoder mean |x|:** `0.440264`

**Rust DurationEncoder first-time first-8 dims (broken):**
`[-0.29620, 0.23453, -0.03288, -0.09144, 0.40526, -0.19921, 2.40312, -0.18004]`

**Drift (first time):** `0.624` max absolute element diff — the biggest
drift is at dim 6: Rust 2.403 vs Python 1.779.

**Drift (last time):** `1.474` max absolute element diff.

**Drift (mean |x|):** `0.025`.

**Expected pred_dur for this input (from Python):** `[18, 2, 2, 3, 4, 9, 18]`.

**Expected total frames:** `sum(pred_dur) = 56`.

**Expected audio length:** `56 * 300 = 16800` before iSTFT, `33600`
after (2x because the decoder upsamples by 2 via the AdainResBlk1d
upsample block before the generator).

---

## 11. If You Are Reading This Cold

You are a Rust developer who has been handed a pure-Rust port of the
Kokoro-82M text-to-speech model. The port runs end-to-end and produces
valid WAV output from the real 82M-parameter Kokoro weights, but the
audio is not yet intelligible speech because the `DurationEncoder`
component in the prosody predictor diverges numerically from the
Python reference by up to 1.47 in individual tensor elements. The
`DurationEncoder` is in
`ferrocarril/ferrocarril/ferrocarril-nn/src/prosody/duration_encoder.rs`
and the Python reference is in
`ferrocarril/kokoro/kokoro/modules.py`. You have a test at
`ferrocarril/ferrocarril/tests/duration_encoder_golden_test.rs` that
fails, a Python harness at `ferrocarril/scripts/validate_kmodel.py`
that dumps reference tensors for every layer of the full model, and
a set of `.npy` fixtures under
`ferrocarril/ferrocarril/tests/fixtures/kmodel/` that include
`predictor_text_encoder.npy` (the DurationEncoder output you are
trying to match).

Start by reading §4 above, then open `duration_encoder.rs` and
`modules.py` side-by-side, then decide whether to extend
`validate_kmodel.py` with per-AdaLayerNorm hooks (recommended) or to
instrument the Rust side with intermediate tensor dumps. The LayerNorm
bug pattern that bit the TextEncoder (normalizing across the wrong
axis) is the single most likely culprit here — check `AdaLayerNorm`
first. After that, check the bidirectional-LSTM reverse-direction
output ordering in `lstm.rs`.

Good luck.