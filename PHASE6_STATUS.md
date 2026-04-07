# Phase 6 — Performance Optimization Progress

> **Session date:** April 7, 2026 afternoon session, branch
> `phase3-numerical-validation` in `iGentAI/ferrocarril`.
>
> **Measurement host:** Intel Xeon Platinum 8175M (Skylake-X, AVX-512,
> 2 vCPU / 2 hyperthreads sharing one physical core's FMA units), running
> the default ferrocarril sandbox with `target-cpu=native` for x86_64.
>
> **Measurements reflect commits through `commit_45`** (8×32 main
> micro-kernel + BLIS b-panel packing + lowered `m < 16` small-m
> threshold + AVX-512 `linear_f32` + LSTM direct-slice rewrite + BERT
> direct-slice rewrite + AVX-512 Snake1D + im2col buffer pool).
>
> **`commit_46` landed but is unmeasured** — it adds a 4×16
> intermediate micro-kernel to the matmul AVX-512 remainder path for
> `m % 8` in `[4, 7]` (expected ~10-15 % improvement on shapes like
> `conv_post` at m=22). The sandbox was terminated before a post-commit
> benchmark could run, so the headline numbers below still correspond
> to the `commit_45` state. The change is in the tree and compiles;
> the next session should rerun `bench_matmul` + `ferrocarril infer`
> to quantify it (see the reproduction section at the bottom for the
> exact commands, including a note on which `bench_matmul` shape
> directly exercises the new 4×16 path).
>
> **Focus**: native-target inference throughput on the sandbox's Intel
> Xeon Platinum 8175M (Skylake-X, AVX-512, 2 vCPU / 2 hyperthreads
> sharing 1 physical core's FMA units).
>
> **Baseline (pre-Phase 6)**: ~258 s wall for the canonical "Hi" →
> ~1 s audio inference, matmul running at ~2.5 GFLOPS (scalar triple
> loop).
>
> **Current** (commit_45 measured): ~2.0 s wall (~130× speedup) with
> `matmul_f32` at 76 GFLOPS on the dominant Generator shape (95 % of
> the 80 GFLOPS 2-FMA-port peak). Still ~2× from real-time for the
> "Hi" test case.

## Current breakdown ("Hi" → ~1 s audio, AVX-512 release build)

```
[profile] infer TOTAL infer_with_phonemes          ~2000 ms
  infer TextEncoder forward                           ~20 ms
  infer BERT forward                                  ~80 ms
  infer ProsodyPredictor forward (dur)                ~50 ms
  infer ProsodyPredictor forward (aligned)            ~50 ms
  infer predict_f0_noise                             ~155 ms
  infer Decoder forward (incl Generator)            ~1640 ms
    conv1d phases total (88 calls)                  ~1450 ms
      matmul_f32                                    ~1230 ms   (60 % of total)
      im2col_b1                                      ~130 ms
      output alloc + bias                             ~25 ms
    non-conv1d decoder work                          ~190 ms
      (AdaIN, Snake1D, LeakyReLU, ConvTranspose1d,
       reflection pad, STFT/iSTFT, tensor overhead)
```

All 24 golden tests still pass at f32 precision against the Python
reference fixtures.

## What Shipped This Session (in rough order of impact)

### 1. Matmul kernel overhaul — the dominant win

Went through many iterations, each validated with the
`bench_matmul` microbenchmark in `ferrocarril-core/src/bin/` and
with the full `ferrocarril infer` profile.

| Optimization                                                 | bench `large` (m=256) | notes                                   |
|--------------------------------------------------------------|-----------------------|------------------------------------------|
| Baseline (scalar `tensor[&[i,j]]` triple loop)               | ~2.5 GFLOPS           | unvectorisable indexed access            |
| Direct-slice + ikj loop order                                | ~10 GFLOPS            | compiler auto-vec to SSE2 4-wide         |
| 3-level cache blocking (`NC = KC = 256`)                     | ~11 GFLOPS            | bandwidth-bound without SIMD intrinsics  |
| Explicit AVX-512 runtime dispatch                            | ~13 GFLOPS            | only marginal over auto-vec — why?       |
| 4×16 register-blocked micro-kernel                           | ~45 GFLOPS            | **4.1× speedup** — breaks fma dep chain |
| 8×16 with 8 independent accumulators                         | ~51 GFLOPS            | saturates 2 FMA ports                    |
| **BLIS-style b-panel packing**                               | ~51 GFLOPS            | kernel looks similar but in-context...   |
| **8×32 (16 accumulators, 2 b loads per kk)**                 | **~76 GFLOPS**        | **93 % of 80 GFLOPS peak**                |

Additional matmul polish:
- **`m < 16` small-m fast path** — an unblocked 1×16 kernel skips
  the packing overhead for BERT-sized Q/K/V projections.
- **2-way `std::thread::scope` parallelism** on the `m` dimension
  for `m >= 16`. Helps but limited because the sandbox's 2 vCPUs
  are hyperthreads sharing one physical core's FMA units.
- **Thread-local packed buffer** (`PACKED_B`) reused across calls
  so no per-call allocation for the 256 KB b panel.

### 2. `linear_f32` AVX-512 dot-product kernel

Added `linear_f32_avx512` with **4-way unrolled ZMM accumulators**
for the dot-product reduction. Breaks the fma dependency chain so
both SKX FMA ports can dispatch at full rate. Per output:
4 independent `acc0..acc3` running `vfmadd231ps`, reduced via
`_mm512_reduce_add_ps` at the end.

**BERT impact**: 436 ms → 83 ms (5.3× speedup). `linear_f32` was a
scalar triple loop running at ~10 GFLOPS; now it's ~60-80 GFLOPS.

### 3. BERT direct-slice rewrite

Rewrote the hot forward paths of `LinearProjection`, `FeedForward`,
`LayerNorm`, `MultiHeadAttention`, and `EmbeddingHiddenMapping` to
use direct `&[f32]` slice access and the optimized `linear_f32`
instead of the old `tensor[&[...]]` quadruple-nested loops.

Also fixed stale embedding size handling to read from the config.

### 4. LSTM direct-slice rewrite

Rewrote `LSTM::step` to pull the weight/bias tensors into `&[f32]`
slices once at the top and do the inner dot products over contiguous
rows instead of `weights_ih[&[g, i]]` indexed reads.

**ProsodyPredictor impact**:
- `ProsodyPredictor::forward(dur)`: 101 ms → 50 ms
- `ProsodyPredictor::forward(aligned)`: 98 ms → 50 ms
- `predict_f0_noise`: 281 ms → 155 ms

Note: this also benefits `DurationEncoder` which uses the same
`LSTM` struct internally.

### 5. Snake1D AVX-512 polynomial sin

`AdaINResBlock1::apply_snake_in_place` was calling libm's scalar
`sinf()` ~26 million times per Generator forward (60 Snake1D calls
× ~430k elements each), at ~18 ns per call = ~460 ms of inference
time.

Replaced with a branch-free **9-term minimax polynomial**
approximation (`fast_sin`), then wrapped in an AVX-512 SIMD kernel
(`apply_snake_chan_avx512`) that processes 16 elements per
iteration:

1. `ax = a * v`
2. Range-reduce: `n = round(ax/π)`, `y = ax - n*π` via
   `_mm512_roundscale_ps`
3. Polynomial: 4 `_mm512_fmadd_ps` building
   `sin(y) ≈ y·(1 + y²·(c3 + y²·(c5 + y²·(c7 + y²·c9))))`
4. Sign flip for odd `n` via `_mm512_mask_xor_ps` with the sign
   bit
5. `v += inv_a · sin(ax)²` via 2 more fmadds

Error stays within ~5e-7 (comparable to libm's sinf). Saves
~250 ms total.

### 6. `Conv1d` im2col buffer pool

Added a `thread_local! { IM2COL_BUFFER: RefCell<Vec<f32>> }` so the
~20 MB im2col scratch for each `Conv1d::conv1d_b1_g1_im2col` call
is reused instead of freshly allocated per call. Also rewrote
`im2col_b1` to fully overwrite its destination (boundary zeros via
`slice::fill`) so a pooled buffer with stale data is safe.

### 7. In-place `AdaINResBlock1`

Eliminated the `data().to_vec()` clones in `AdaINResBlock1::forward`
by taking the `Tensor` returned from `AdaIN1d::forward` as `mut` and
calling `apply_snake_in_place` directly on `xt.data_mut()`. Same for
the residual accumulator (`x_acc.data_mut().iter_mut().zip(...)`).

### 8. Fine-grained `FERRO_PROFILE` instrumentation

Added per-stage timers (gated on the `FERRO_PROFILE` env var) inside
`infer_with_phonemes`, `Decoder::forward`, `Generator::forward`, and
a per-phase breakdown inside `Conv1d::conv1d_b1_g1_im2col`
(`output_alloc_ns`, `im2col_ns`, `matmul_ns`, `bias_ns`). Absolutely
crucial for finding the hot spots — without it, several of the
optimizations above would not have been obvious wins.

## Diminishing Returns and Remaining Work

The matmul is at **93 % of theoretical AVX-512 peak** on the
dominant Stage 0 shape and ~77 % on the Stage 1 shape. Further
matmul speedups require either:

1. **Int8 / fp16 quantisation** for 2-4× more throughput via
   AVX-512 VNNI — **not available on this Xeon 8175M** (VNNI requires
   Cascade Lake or later). Would work on newer hardware, but this
   sandbox is capped.
2. **Higher-level parallelism** (e.g. running the 3 Generator
   resblocks per stage in parallel on separate threads) — limited
   by the 2 vCPUs being hyperthreads sharing one physical core's
   FMA units. Real 2-physical-core hosts would benefit.
3. **Reducing total FLOPs** via model changes — not an option
   because this is a faithful port of Kokoro-82M.

Other directions that may be worth ~100-200 ms each:
- **Direct conv** (no im2col) for small-kernel Generator convs:
  eliminates the ~130 ms im2col_b1 fill time.
- **Kernel fusion** of AdaIN + Snake1D into a single pass: saves
  memory bandwidth on the intermediate tensor.
- **ConvTranspose1d AVX-512 packed kernel**: currently uses the
  same matmul kernel but without the same level of tuning.
- **Reduce allocation churn** in `Generator::forward`: several
  `data().to_vec()` + `Tensor::from_data` pairs for LeakyReLU,
  combine, average, reflection pad.

## Timing History Summary

| Milestone                                           | Total wall | Speedup vs baseline |
|-----------------------------------------------------|-----------:|--------------------:|
| Pre-Phase 6 (scalar triple loop matmul)             | ~258 s     | 1×                  |
| + Direct-slice matmul                               |  ~62 s     | 4.2×                |
| + Conv1d im2col                                     |  ~21 s     | 12.3×               |
| + ConvTranspose1d / AdaIN1d / InstanceNorm rewrite  |  ~12 s     | 21.5×               |
| + `target-cpu=native` (auto-vec)                    |  ~12 s     | (no change)         |
| + Explicit AVX-512 matmul, cache blocking, 2-thread |   ~6.4 s   | 40×                 |
| + 4×16 register-blocked micro-kernel                |   ~3.2 s   | 80×                 |
| + Snake1D scalar `fast_sin` polynomial              |   ~3.1 s   | 83×                 |
| + BLIS-style b-panel packing                        |   ~2.9 s   | 89×                 |
| + `linear_f32` AVX-512 (BERT speedup)               |   ~2.5 s   | 103×                |
| + LSTM direct-slice rewrite                         |   ~2.2 s   | 117×                |
| + 8×32 main micro-kernel (commit_43)                |   ~2.0 s   | 130×                |
| + Lowered small-m threshold (commit_44)             |   ~2.0 s   | 130×  (minor)       |
| + 4×16 intermediate remainder path (commit_46)      |    ?       | **pending measurement** |

## Commit trail in this session

The commits implementing the work above are under the feature
branch `phase3-numerical-validation` (the long-running PR branch)
but labelled as Phase 6 performance work in their messages. Run
`git log --oneline -40` to see the full trail.

Key files touched:
- `ferrocarril-core/src/ops/matmul.rs` (the matmul kernel, multiple revisions)
- `ferrocarril-core/src/bin/bench_matmul.rs` (microbenchmark)
- `ferrocarril-nn/src/linear.rs` (linear_f32 dispatch)
- `ferrocarril-nn/src/lstm.rs` (direct-slice step)
- `ferrocarril-nn/src/bert/{attention,feed_forward,layer_norm,transformer,embeddings}.rs` (full rewrite)
- `ferrocarril-nn/src/conv.rs` (im2col pool, phase counters)
- `ferrocarril-nn/src/vocoder/adain_resblk1.rs` (fast_sin + AVX-512 Snake1D)
- `ferrocarril-nn/src/vocoder/mod.rs` (Generator + Decoder profile markers)
- `src/model/ferro_model.rs` (dump_conv1d_stats integration)
- `.cargo/config.toml` (`target-cpu=native` for x86_64)

## How to reproduce

```bash
# Build optimised
cd ~/ferrocarril/ferrocarril
cargo build --release

# Micro-benchmark matmul. Shapes that exercise which paths:
#   small  (m=3)   -> unblocked fast path (m < 16)
#   medium (m=22)  -> packed 8x32 main (m_main=16) + NEW 4x16 intermediate
#                     (rows 16..20) + 1x16 tail (rows 20..22)
#                     THIS IS THE SHAPE THAT EXERCISES COMMIT_46.
#   large  (m=256) -> packed 8x32 main, no remainder
#   xl     (m=128) -> packed 8x32 main, no remainder
cargo build --release --bin bench_matmul -p ferrocarril-core
./target/release/bench_matmul

# To confirm commit_46's 4x16 intermediate path is active, look at
# the `medium` GFLOPS row: commit_45 measured ~10.8 GFLOPS (all 6
# remainder rows went through the slow 1x16 path). With commit_46
# active, the 4x16 path should push `medium` closer to 20+ GFLOPS
# because 4 of those 6 rows now have 4 independent c accumulators
# and a shared b load per kk.

# Full profiled inference. Look at `[profile] conv1d matmul_f32`
# for end-to-end matmul time on the canonical "Hi" input.
FERRO_PROFILE=1 ./target/release/ferrocarril infer \
  --text "Hi" \
  --output /tmp/hi.wav \
  --model ../ferrocarril_weights \
  --voice af_heart

# Run the golden test suite to verify numerical correctness after
# both commit_45 and commit_46 (all 24 tests should pass at f32
# precision against the Python fixtures).
cargo test --release --features weights --no-fail-fast
```