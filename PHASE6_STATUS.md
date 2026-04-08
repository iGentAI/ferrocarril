# Phase 6 — Performance Optimization Progress

> **Evening session update — April 7, 2026**
>
> This session validated the multi-core parallelism arm on a
> **privileged 16-vCPU / 8-physical-core Sapphire Rapids-class
> sandbox** and measured whether `commit_46`'s 4×16 intermediate
> remainder kernel actually delivered its claimed improvement.
> All findings from this session are documented immediately below;
> the full afternoon-session log (commits 1–46) starts further
> down under **Afternoon session log**.
>
> **Hardware context corrections.** The default sandbox is no longer
> the Xeon Platinum 8175M (Skylake-X, 2.5 GHz) referenced in the
> afternoon-session block: it is now an **Intel Xeon Platinum 8375C
> (Ice Lake-SP) @ 2.9 GHz**. Theoretical AVX-512 2-FMA-port peak on
> the new default is therefore ~92.8 GFLOPS per core, not the 80
> GFLOPS quoted in the afternoon block. Both the default and the
> privileged sandboxes expose `avx512_vnni` in `/proc/cpuinfo`,
> contradicting the afternoon block's "VNNI not available on this
> Xeon 8175M" diminishing-returns note — on the _current_ hosts,
> int8 VNNI quantisation becomes a viable future step.
>
> The privileged sandbox created this session has 16 vCPUs / 8
> physical cores / 2 threads per core; CPU flags show `amx_tile`,
> `amx_bf16`, `amx_int8`, `avx512_fp16`, `avx512_bf16`, and
> `avx512_vnni`, identifying it as **Sapphire Rapids-class** (or
> newer — Emerald Rapids flags are identical). This is the first
> session in which a real ≥2-physical-core host has been available
> for Phase 6 measurement.
>
> **Test status at end of evening session:**
> `cargo test --workspace --release --no-fail-fast` on the
> privileged sandbox → **173 passed / 0 failed / 0 ignored / 0
> warnings** across 34 test binaries (24 ferrocarril integration
> tests + 20 ferrocarril-nn lib unit tests + 129 phonesis lib +
> integration tests) in **~18 s wall** (down from the ~10 min the
> afternoon block would project). The full suite runs this much
> faster because the golden tests are dominated by the decoder
> forward pass, and commit_1 + commit_4 + the ~3× per-core
> Sapphire Rapids advantage combine to make each golden test's
> forward run in ~1 s instead of ~5-8 minutes.

## Commits shipped in this session

1. **commit_1** `matmul.rs` — Generalise `matmul_f32_avx512_dispatch`
   and `matmul_f32_avx2_dispatch` from a hardcoded 2-way m-dimension
   split to an **N-way scoped-worker split**. Adds a
   process-lifetime-cached `matmul_workers()` helper
   (`min(available_parallelism, 8)`) with an
   `FERRO_MATMUL_THREADS` env-var override for ablation and
   benchmarking.
2. **commit_2** `matmul.rs` — Stale-docstring fix on `matmul_f32`
   ("2-way row-parallelism" → the new N-way policy). Overseer
   follow-up from commit_1.
3. **commit_3** `.gitignore` — Exclude `core.*` Linux core dumps
   from filestore syncing after a series of SIGILL incidents
   during debugging (see the mutagen section below) kept uploading
   2.8 MB dumps into the FileStore.
4. **commit_4** `matmul.rs` — Introduce a `MIN_WORK_PER_WORKER =
   32 Mops` threshold via a new `worker_count_for_shape(m, n, k)`
   helper. Caps workers by the minimum of host parallelism,
   available rows, and total work. **Fixes a latent ~4× regression
   on small Generator conv shapes** that had been hidden in the
   pre-existing hardcoded 2-way dispatcher — the biggest discovery
   of this session. Details below.
5. **commit_5** `matmul.rs` — Refresh the top-of-file module and
   `matmul_f32` doc comments to describe the new
   `worker_count_for_shape` policy (overseer follow-up from
   commit_4).

## The latent small-shape regression (and its fix)

The afternoon session's `commit_46` claimed that adding a 4×16
intermediate remainder kernel would push the `medium` bench shape
(m=22, the Generator's `conv_post` convolution) "closer to 20+
GFLOPS" from the commit_45 baseline of ~10.8 GFLOPS. This session
measured that prediction for the first time, and uncovered why
it wasn't quite right.

The outcome depends entirely on **how many workers m=22 gets split
across**:

| Dispatcher config                    | medium GFLOPS (privileged) | medium GFLOPS (default Ice Lake) |
|--------------------------------------|---------------------------:|---------------------------------:|
| hardcoded 2-way (commit_45 / commit_46) | ~32 GFLOPS               | ~11 GFLOPS                       |
| **`FERRO_MATMUL_THREADS=2` (same code path)** | **~32 GFLOPS (measured)**  | **~11 GFLOPS (measured)**        |
| **`FERRO_MATMUL_THREADS=1` (serial)**         | **~68 GFLOPS (measured)**  | **~42 GFLOPS (measured)**        |

At 2-way, each worker receives 11 rows; the main 8×32 kernel
consumes 8 of them and the remaining 3 rows all fall through to
the slow 1×16 tail. The 4×16 intermediate kernel never activates
in that case because its gate is `i + 4 <= m_worker`, and
`8 + 4 > 11`.

In the serial path (22 rows, 1 worker), the math is different:
- `m_main = 16 rows via 8×32`
- `remainder = 6 rows: 4 via the new 4×16 intermediate + 2 via 1×16 tail`

That's where commit_46's 4×16 intermediate actually lands.
Combined with eliminating the `std::thread::scope` spawn
overhead + per-worker `PACKED_B` re-allocation on a shape whose
actual compute finishes in under a millisecond, the serial path
delivers **~68 GFLOPS on multicore (6.3× the 2-way number)** and
**~42 GFLOPS on default (3.8× the 2-way number)**.

This is a **latent regression in the pre-Phase 6 hardcoded 2-way
dispatcher** that had never been surfaced because the old
`bench_matmul` suite only compared "blocked+SIMD" vs "scalar
baseline" — it never measured "serial vs 2-way" for small shapes.
commit_1's new `FERRO_MATMUL_THREADS=1` ablation exposed it
immediately, and **commit_4** fixed it by gating the parallel
path behind `total_work >= MIN_WORK_PER_WORKER`. The Generator's
`medium` shape is ~33 Mops — narrowly above the 32 Mops gate —
but the helper's integer division `33/32 = 1` caps the worker
count at 1 so it routes to serial anyway, picking up the full
6.3× improvement on the privileged sandbox.

## N-way scaling on the privileged sandbox

With commit_5 in place, the `bench_matmul` thread-sweep on the
privileged sandbox (isolated `CARGO_TARGET_DIR`, clean rebuild):

| N  | small (m=3) | medium (m=22) | large (m=256)    | xl (m=128)       |
|----|-------------|---------------|------------------|------------------|
| 1  | 21.8 GFLOPS | 67.8 GFLOPS   | 144.3 GFLOPS     | 135.2 GFLOPS     |
| 2  | 22.9        | 67.8 (pinned) | 277.9 (1.93×)    | 248.4 (1.84×)    |
| 3  | 21.9        | 68.9          | 293.3 (2.03×)    | 278.9 (2.06×)    |
| 4  | 20.4        | 66.1          | 426.0 (2.95×)    | 343.0 (2.54×)    |
| 5  | 21.8        | 67.6          | **534.8 (3.70×)**| 353.6 (2.62×)    |
| 6  | 22.3        | 68.2          | 401.4 (noise)    | 341.4            |
| 7  | 19.5        | 67.4          | **586.4 (4.06×)**| 136.7 (noise)    |
| 8  | 20.6        | 68.7          | 431.1 (noise)    | **415.0 (3.07×)**|

Observations:
- **Medium is pinned to ~68 GFLOPS** regardless of the env-var
  setting, exactly because the `MIN_WORK_PER_WORKER` gate fires
  before the host-parallelism cap. commit_4 working as designed.
- **Large peaks at ~4.1× (N=7)**, plateaus around N=5-8. The
  irregular dip at N=6 and the ~4× win at N=7 are reproducible
  across runs — it's thread-scheduling quantisation, not noise.
- **XL peaks at ~3.1× (N=8)**, with a suspicious regression at
  N=7 that IS noise (re-runs oscillate between 136 and ~400).
- Above N=8 (not shown, tested in prior runs) the SMT sibling
  contention dominates and the curve regresses — two logical
  threads sharing one physical core's FMA ports is strictly worse
  than one thread alone.

The cap of `8` workers hard-coded into `matmul_workers()` is the
right default for this class of hardware: the marginal win from
8 → 16 is noise-level negative, and the added per-call scope-spawn
overhead eats whatever was left. That's why the default policy
caps at 8 regardless of what `available_parallelism()` reports.

## End-to-end inference measurement on the privileged sandbox

Full `ferrocarril infer --text "Hi" --voice af_heart` with
`FERRO_PROFILE=1`, producing 1.275 s of audio (30600 samples @
24 kHz, RMS ~0.042, peak ~0.24):

| N  | wall     | `TOTAL infer_with_phonemes` | `matmul_f32` | `Generator::forward` | inference RTF |
|----|---------:|----------------------------:|-------------:|---------------------:|--------------:|
| 1  | 1424 ms | 889.2 ms                    | 605.5 ms     | 660.5 ms             | 1.43×         |
| 2  | 1210 ms | 672.3 ms                    | 378.6 ms     | 463.6 ms             | 1.90×         |
| 8  | **1122 ms** | **596.2 ms**            | **246.7 ms** | **402.1 ms**         | **2.14×**     |

(Inference RTF = `audio_duration / infer_with_phonemes`. >1 means
faster than real-time. Wall includes the ~506 ms
`load_from_loader` step, which the RTF column does not.)

**At N=8 on the privileged sandbox, the canonical "Hi" inference
runs at 2.14× real-time** — the 1275 ms audio takes 596 ms to
generate. Wall-clock including model load is 1122 ms.

The matmul-only speedup from N=1 → N=8 (2.45×) is larger than
the full-inference speedup (1.49×) because of Amdahl's law:
~250 ms of serial sections (BERT forward, LSTM inside
ProsodyPredictor, non-conv Generator work, iSTFT) don't
parallelise and cap the realisable win.

**Unexpected observation — `im2col_b1` cache contention.** The
im2col phase times **grow** with N:

| N  | `im2col_b1` | `matmul_f32` | `Conv1d TOTAL phases` |
|----|-------------|--------------|-----------------------|
| 1  | 34.7 ms     | 605.5 ms     | 648.6 ms              |
| 2  | 41.1 ms     | 378.6 ms     | 431.4 ms              |
| 8  | 94.4 ms     | 246.7 ms     | 352.1 ms              |

im2col itself is serial (it populates a thread-local buffer on
the calling thread), so it shouldn't be affected by worker count.
The ~2.7× slowdown at N=8 is almost certainly **L2 cache
eviction from the per-worker packed b panels**: 8 workers × 256
KB packed buffer = 2 MB of L2 pressure, which matches the
Sapphire Rapids 2 MB-per-physical-core L2 size. The main
thread's im2col input tensor keeps getting evicted between conv
calls. A persistent thread pool that kept those packed buffers
resident on specific worker threads (instead of re-allocating
them on every scope spawn) would likely recover most of this
loss — deferred as future work.

## mutagen cross-sandbox `target/` sync gotcha

While debugging a series of SIGILL crashes on the default Ice
Lake sandbox, this session discovered that **mutagen actively
syncs `target/` build directories between sandboxes that share
the same FileStore mount**, despite `target/` being in
`.gitignore`. The symptom reproduced as follows:

1. Default sandbox builds `bench_matmul` with
   `RUSTFLAGS="-C target-cpu=native"`. On Ice Lake that resolves
   to `icelake-server` — the binary is perfectly legal for the
   host.
2. Privileged sandbox builds the same binary in parallel. On
   Sapphire Rapids, `target-cpu=native` resolves to
   `sapphirerapids` and LLVM emits AMX + AVX-512-BF16 +
   AVX-512-FP16 instructions in the LTO-linked std formatter code.
3. mutagen propagates the privileged build's
   `target/release/bench_matmul` through the filestore back down
   to the default sandbox, overwriting the icelake-server build.
4. The next run on default hits SIGILL (`core::fmt::num::imp::<impl
   u64>::_fmt_inner`) because Ice Lake doesn't decode Sapphire
   Rapids AMX opcodes.

**Evidence**: both sandboxes reported identical 462448-byte
binaries (no Rust-version / target-CPU divergence visible in
file size) despite running `rustc 1.89` on default and
`rustc 1.93` on the privileged host; md5sums after the
workaround were distinct as expected.

**Fix**: set `CARGO_TARGET_DIR=/tmp/ferro_{default,multicore}_target`
so each sandbox's build lands outside the mutagen-synced tree.
`/tmp` is sandbox-local. All measurements in this session were
taken with the isolated build layout.

This is a general-purpose trap for anyone running Rust builds on
multiple sandboxes sharing a FileStore, and should probably be
called out in `HANDOFF.md` for future sessions.

## Deferred work

1. **Persistent thread pool**. Would recover the `im2col_b1`
   cache-contention loss (~50 ms per inference at N=8) and the
   per-call `std::thread::scope` spawn cost (~10-30 ms per
   inference). Marginal compared to the changes already landed,
   but worth revisiting once other wins are exhausted.
2. **AVX-512 VNNI int8 quantisation**. Both current sandboxes
   (default Ice Lake 8375C and privileged Sapphire Rapids) expose
   `avx512_vnni` in their flag sets, contradicting the afternoon
   block's "not available on 8175M" note. This is now a viable
   next step: int8 quantisation at load time, an int8 matmul
   kernel using VNNI's `vpdpbusd` dot-product instruction, and
   calibration against the golden tests. Potential ~2-4× on the
   matmul.
3. **Re-measure default Ice Lake with commit_5**. Per user
   direction, this session used only the privileged sandbox for
   benchmarks. The expected effect of commit_4 on the default
   sandbox (from the medium-shape serial number measured earlier
   in the session) is that the Generator's `medium` conv_post
   shape should run at ~42 GFLOPS serial instead of ~11 GFLOPS
   hardcoded-2-way, saving on the order of ~100 ms per inference
   on Ice Lake. This still needs to be confirmed by running
   `ferrocarril infer` on the default sandbox with commit_5.
4. **Kernel fusion / allocation churn cleanup / direct conv /
   ConvTranspose1d packed kernel**. Listed in the afternoon
   block's "Diminishing returns" section. Still valid, still
   deferred.

---

## Afternoon session log

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
cd ~/ferrocarril
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
  --model ferrocarril_weights \
  --voice af_heart

# Run the golden test suite to verify numerical correctness after
# both commit_45 and commit_46 (all 24 ferrocarril integration
# tests + the rest of the 173 workspace tests should pass at f32
# precision against the Python fixtures).
cargo test --workspace --release --no-fail-fast
```