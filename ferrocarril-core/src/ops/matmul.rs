//! Matrix multiplication operations
//!
//! Two flavours of f32 GEMM-family kernel are exposed:
//!
//! - [`matmul`] / [`matmul_f32`] — standard `c = a @ b` where both
//!   matrices are stored row-major. On x86_64 the inner kernel picks
//!   the widest SIMD path supported at runtime: **AVX-512** (16-wide
//!   f32 FMA) → **AVX2+FMA** (8-wide) → scalar fallback. Both SIMD
//!   paths use **3-level cache blocking** to keep the b panel
//!   resident in L2 across the m row iterations. For matrices with
//!   `m >= PARALLEL_THRESHOLD` they additionally split the m
//!   dimension across **N scoped worker threads**, where `N` is
//!   computed per-call by [`worker_count_for_shape`] —
//!   `min(matmul_workers(), m / MR_MIN, total_work / MIN_WORK_PER_WORKER)`.
//!   The `total_work` cap prevents small shapes like the Generator
//!   `conv_post` (m=22) from paying scoped-thread spawn overhead
//!   larger than the serial kernel's entire runtime. The host
//!   parallelism cap defaults to `min(available_parallelism, 8)`
//!   and is overridable via the `FERRO_MATMUL_THREADS` environment
//!   variable. On a 2-vCPU/1-physical-core hyperthreaded host the
//!   parallel path gives ~1.3-1.5× over serial on large shapes and
//!   the work-threshold keeps small shapes on the serial fast
//!   path. On an 8-physical-core host (e.g. a privileged sandbox
//!   with 16 vCPUs) the parallel path scales much closer to N×
//!   because each worker lands on a distinct physical core's FMA
//!   ports.
//!
//! - [`linear_f32`] — fused `y = x @ w^T + bias` kernel where `w` is
//!   stored in its **untransposed** PyTorch layout `[out_features,
//!   in_features]`. On x86_64 with `avx512f`, dispatches to a
//!   **4-way unrolled ZMM dot-product** kernel that breaks the fma
//!   dependency chain across 4 independent accumulators so both
//!   AVX-512 FMA ports can dispatch at full rate. Falls back to a
//!   scalar auto-vectorised triple loop on other hosts.
//!
//! **Output buffer contract.** `matmul_f32` **accumulates** into the
//! caller's `c` buffer — callers are responsible for zeroing it
//! first. `linear_f32` **overwrites** `y[i, o]`.
//!
//! **Cache blocking rationale.** Without blocking, each of `m` outer
//! `i` iterations re-touches the entire `(k, n)` b matrix. For the
//! Stage 1 Generator convs, that means the L2 cache thrashes
//! repeatedly. Blocking on `(j_block, k_block)` keeps a `KC × NC` b
//! sub-panel in L2 across all `m` iterations of `i`, dramatically
//! improving cache reuse. Tuned for the L2 cache sizes seen on
//! modern Intel server parts (Skylake-X / Ice Lake / Sapphire
//! Rapids — 1-2 MB per physical core) with `KC = NC = 256`
//! (256 KB packed b panel).
//!
//! **Parallelism rationale.** After cache blocking the matmul is
//! still compute-bound at the available SIMD throughput, so adding
//! more workers helps directly when those workers run on distinct
//! physical cores. Splitting the m dimension is row-disjoint in
//! `c`, so workers don't interfere and the borrow-checker is happy.
//! The `MIN_WORK_PER_WORKER` gate (32 Mops per worker) was
//! calibrated empirically from the Generator's `conv_post` shape
//! and prevents small-work shapes from paying spawn overhead
//! larger than their serial runtime. The cap of `8` workers
//! reflects the diminishing returns observed once the per-call
//! `scope.spawn` overhead (~30-100 us × N workers × ~88 matmul
//! calls per Generator forward) and the cumulative L2 footprint of
//! the per-worker BLIS-packed b panels (256 KB each) start eating
//! the win.

use crate::tensor::Tensor;
#[cfg(target_arch = "x86_64")]
use std::cell::RefCell;
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

/// Matrices with `m` strictly less than this fall through to the
/// single-threaded path. Avoids paying ~30-100 μs of thread spawn
/// overhead on tiny matmuls (BERT Q/K/V projections at m=3-7, etc).
#[cfg(target_arch = "x86_64")]
const PARALLEL_THRESHOLD: usize = 16;

/// Minimum rows per parallel worker. Equal to MR (the row block of
/// the main 8×32 micro-kernel) so that every worker has at least
/// one full main-kernel iteration. Workers with fewer than `MR_MIN`
/// rows would fall through to the small-m path, which is slower per
/// row and defeats the parallelism gain.
#[cfg(target_arch = "x86_64")]
const MR_MIN: usize = 8;

/// Minimum total work (f32 multiply-add ops) a parallel worker must
/// have before we spawn it. Below this, the scoped thread spawn
/// overhead plus per-worker `PACKED_B` re-allocation (each spawned
/// worker sees a fresh thread-local and pays the 256 KB alloc on
/// first use) dominates any parallel speedup and the dispatcher
/// loses to the serial path.
///
/// Calibrated empirically from the Generator's "medium" conv_post
/// shape (m=22 n=3361 k=448, ~33 Mops total) on both the default
/// Ice Lake sandbox and the 16-vCPU Sapphire Rapids privileged
/// sandbox:
///
/// - Default Ice Lake: N=1 = 42 GFLOPS, N=2 = 11 GFLOPS (2-way loses 3.8×)
/// - Multicore Sapphire Rapids: N=1 = 68 GFLOPS, N=2 = 32 GFLOPS (2-way loses 2.1×)
///
/// Setting the threshold at 32 Mops/worker keeps `medium` (33 Mops
/// total → 1 worker) on the serial path while allowing `large`
/// (~400 Mops) and `xl` (~600 Mops) to use the full worker count.
/// This fixes a latent regression in the pre-existing hardcoded
/// 2-way dispatcher that was never visible through the old bench
/// suite (which only compared "blocked+SIMD" vs "scalar baseline"
/// and never measured "serial vs 2-way" for these shapes).
#[cfg(target_arch = "x86_64")]
const MIN_WORK_PER_WORKER: usize = 32_000_000;

/// Determine how many worker threads to use for the m-dimension
/// split in the parallel matmul dispatchers. The result is cached
/// on first call so the env-var lookup and `available_parallelism()`
/// syscall only happen once per process.
///
/// **Override**: set the `FERRO_MATMUL_THREADS` environment variable
/// to a positive integer to force a specific worker count. Useful
/// for ablation studies (`FERRO_MATMUL_THREADS=1` disables the m
/// split entirely) and for benchmarking thread-scaling curves.
///
/// **Default**: `min(available_parallelism, 8)`. The cap reflects
/// the diminishing returns observed once the per-call `scope.spawn`
/// overhead (~30-100 us × N workers × ~88 matmul calls per
/// inference) and the cumulative L2 footprint of the per-worker
/// packed b panels (256 KB each) start eating the win.
#[cfg(target_arch = "x86_64")]
fn matmul_workers() -> usize {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    *WORKERS.get_or_init(|| {
        if let Ok(s) = std::env::var("FERRO_MATMUL_THREADS") {
            if let Ok(n) = s.parse::<usize>() {
                if n >= 1 {
                    return n;
                }
            }
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .min(8)
    })
}

/// Compute the number of worker threads for a matmul of the given
/// shape. Applies three independent caps, then takes the minimum:
///
/// 1. `matmul_workers()` — the host's effective parallelism (at
///    most 8, overridable via `FERRO_MATMUL_THREADS`).
/// 2. `m / MR_MIN` — workers with fewer rows than the main 8×32
///    micro-kernel's row block wouldn't even run the main kernel.
/// 3. `total_work / MIN_WORK_PER_WORKER` — ensures each worker has
///    enough flops to amortize its scoped-thread spawn + packed-b
///    allocation overhead.
///
/// The final clamp with `.max(1)` guarantees at least one worker
/// (the caller's own thread via the serial fallback if the result
/// is 1).
#[cfg(target_arch = "x86_64")]
fn worker_count_for_shape(m: usize, n: usize, k: usize) -> usize {
    let max_by_rows = (m / MR_MIN).max(1);
    // saturating_mul to defend against `m * n * k` overflowing usize
    // on 32-bit targets with very large (though unusual) shapes.
    let total_work = m.saturating_mul(n).saturating_mul(k);
    let max_by_work = (total_work / MIN_WORK_PER_WORKER).max(1);
    matmul_workers()
        .min(max_by_rows)
        .min(max_by_work)
        .max(1)
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Per-thread packed b panel for the BLIS-style AVX-512 matmul
    /// inner kernel. For each `(j_block, k_block)` outer iteration,
    /// we copy the `(KC, NC)` slice of b into this buffer in a
    /// layout that makes the inner kk reduction a sequential read.
    ///
    /// Layout: 16-element cache-line groups, with consecutive `kk`
    /// values stored consecutively in memory:
    ///
    /// ```text
    /// packed[jj_chunk * KC * 16 + kk * 16 + lane]
    ///   = b[k_block + kk, j_block + jj_chunk*16 + lane]
    /// ```
    ///
    /// Buffer grows monotonically to `KC * NC = 64K` floats (256 KB)
    /// on first use, then is reused across calls **on the same
    /// thread**. Note: scoped worker threads spawned by
    /// `std::thread::scope` are fresh per call, so each spawned
    /// worker re-allocates its 256 KB panel; only the calling
    /// thread benefits from the persistent reuse. A persistent
    /// thread pool would close that gap (see PHASE6_STATUS.md).
    static PACKED_B: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// `c = a @ b` for two row-major 2D `Tensor<f32>` operands.
pub fn matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    assert_eq!(a.shape().len(), 2, "First tensor must be 2D");
    assert_eq!(b.shape().len(), 2, "Second tensor must be 2D");
    assert_eq!(
        a.shape()[1],
        b.shape()[0],
        "Incompatible dimensions for matrix multiplication"
    );

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    let mut c_data = vec![0.0f32; m * n];
    matmul_f32(a.data(), b.data(), &mut c_data, m, n, k);
    Tensor::from_data(c_data, vec![m, n])
}

/// Low-level row-major f32 GEMM: `c[m, n] += a[m, k] @ b[k, n]`.
///
/// **Accumulating**: caller must pre-zero `c`.
///
/// On x86_64 dispatches to AVX-512, AVX2+FMA, or scalar based on
/// runtime CPU feature detection. The AVX-512 path processes 16 f32
/// lanes per fma; AVX2 processes 8. Both SIMD paths use cache
/// blocking and split the `m` dimension across `N` scoped worker
/// threads for `m >= PARALLEL_THRESHOLD`, where `N` is computed by
/// [`worker_count_for_shape`] as the minimum of the host
/// parallelism (at most 8, overridable via `FERRO_MATMUL_THREADS`),
/// the available row count (`m / MR_MIN`), and the total work
/// available (`total_work / MIN_WORK_PER_WORKER`). The work cap
/// keeps small shapes on the serial fast path where scoped-thread
/// spawn overhead would otherwise dominate.
#[inline]
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k, "matmul_f32: a length mismatch");
    debug_assert_eq!(b.len(), k * n, "matmul_f32: b length mismatch");
    debug_assert_eq!(c.len(), m * n, "matmul_f32: c length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            matmul_f32_avx512_dispatch(a, b, c, m, n, k);
            return;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            matmul_f32_avx2_dispatch(a, b, c, m, n, k);
            return;
        }
    }

    // Wasm SIMD128 hot path: enabled when the build was invoked with
    // `RUSTFLAGS="-C target-feature=+simd128"`. Mirrors the AVX2
    // 4×LANES blocked micro-kernel structure but uses 128-bit
    // `core::arch::wasm32` intrinsics (4-wide f32 instead of 8).
    // The auto-vectorised scalar fallback below covers wasm builds
    // that opt out of `simd128`, and any other unsupported target.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            matmul_f32_wasm_simd128(a, b, c, m, n, k);
        }
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    matmul_f32_scalar(a, b, c, m, n, k);
}

/// Portable scalar fallback. Uses the ikj loop order, which LLVM
/// auto-vectorises up to the widest SIMD instruction set the compiler
/// was told to target.
#[allow(dead_code)]
fn matmul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for kk in 0..k {
            let a_ik = a_row[kk];
            let b_row = &b[kk * n..(kk + 1) * n];
            for j in 0..n {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// ============================================================================
// AVX-512 path
// ============================================================================

/// Public dispatcher for the AVX-512 path. Decides whether to run
/// the serial blocked kernel or split `m` across `N` scoped worker
/// threads, where `N` is computed by [`worker_count_for_shape`] —
/// the minimum of the host parallelism, the available rows, and
/// the total work available. Workers write into disjoint row
/// ranges of `c` so the parallel split is borrow-checker safe.
#[cfg(target_arch = "x86_64")]
fn matmul_f32_avx512_dispatch(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    if m < PARALLEL_THRESHOLD {
        unsafe {
            matmul_f32_avx512_serial(a, b, c, m, n, k);
        }
        return;
    }

    let workers = worker_count_for_shape(m, n, k);

    if workers == 1 {
        unsafe {
            matmul_f32_avx512_serial(a, b, c, m, n, k);
        }
        return;
    }

    let rows_per_worker = (m + workers - 1) / workers;

    std::thread::scope(|s| {
        let a_chunks = a.chunks(rows_per_worker * k);
        let c_chunks = c.chunks_mut(rows_per_worker * n);
        for (a_chunk, c_chunk) in a_chunks.zip(c_chunks) {
            let m_local = a_chunk.len() / k;
            s.spawn(move || {
                unsafe {
                    matmul_f32_avx512_serial(a_chunk, b, c_chunk, m_local, n, k);
                }
            });
        }
    });
}

/// Serial AVX-512 cache-blocked GEMM with **8×32 register-blocked
/// micro-kernel** and **BLIS-style b-panel packing**. Block sizes
/// `NC = KC = 256` — b panel is 256 KB, fits in the 1 MB/core L2.
///
/// For each `(j_block, k_block)` outer iteration, the relevant
/// `(KC, NC)` slice of b is first packed into a contiguous
/// thread-local buffer. The inner kernel then reads from the packed
/// buffer sequentially, which eliminates the stride-n gather pattern
/// the prefetcher couldn't keep up with in the unpacked version.
///
/// The main 8×32 micro-kernel processes 8 rows × 2 adjacent
/// jj_chunks (32 cols) per iteration, holding 16 c accumulators
/// live across the KC reduction. Per inner iteration: 2 b-loads +
/// 8 a-broadcasts + 16 vfmadd231ps. The 2 b loads are amortized
/// over 16 fmadds (mem/fma = 0.625, vs 1.125 for the old 8×16
/// kernel), and the 16 independent accumulators break the fma
/// dependency chain so both SKX FMA ports can dispatch at full rate.
///
/// For **small m (m < 16)**, packing overhead dominates the actual
/// compute, so we dispatch to a simpler unblocked kernel
/// [`matmul_f32_avx512_small`] that avoids the per-outer-block
/// packing cost.
///
/// Register pressure for the 8×32 main path: 16 c + 2 b + 1 reused
/// a = 19 ZMM registers in flight (LLVM serialises the 8 a
/// broadcasts so only 1 is live at a time). For the odd-chunk
/// fallback 8×16 path: 8 c + 1 b + 8 a = 17. For the 1×16 remainder
/// path: 1 c + 1 b + 1 a = 3. All fit comfortably in the 32 AVX-512
/// registers.
///
/// The j%16 scalar tail (`jj >= j_width_vec`) still reads from the
/// unpacked `b` directly since that path is element-at-a-time anyway
/// and isn't worth packing overhead for the few edge columns.
///
/// # Safety
/// Requires the host to have the `avx512f` feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn matmul_f32_avx512_serial(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Small m: packing overhead dominates. Use the simple unblocked
    // kernel. Threshold is 16 because the main 8×32 loop needs at
    // least MR=8 rows to run meaningfully, and 16 = 2 × MR gives
    // enough amortization to pay back the packing cost.
    if m < 16 {
        matmul_f32_avx512_small(a, b, c, m, n, k);
        return;
    }

    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps,
    };

    const LANES: usize = 16;
    const NC: usize = 256;
    const KC: usize = 256;
    const MR: usize = 8;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    PACKED_B.with(|buf_cell| {
        let mut packed_buf = buf_cell.borrow_mut();
        let packed_size = KC * NC; // 256 * 256 = 64K floats = 256 KB
        if packed_buf.len() < packed_size {
            packed_buf.resize(packed_size, 0.0);
        }
        let packed_ptr = packed_buf.as_mut_ptr();

        let mut j_block = 0;
        while j_block < n {
            let j_end = (j_block + NC).min(n);
            let j_width = j_end - j_block;
            let j_width_vec = (j_width / LANES) * LANES;
            let nc_chunks = j_width_vec / LANES;

            let mut k_block = 0;
            while k_block < k {
                let k_end = (k_block + KC).min(k);
                let kc = k_end - k_block;

                // Pack b[k_block..k_end, j_block..j_block+j_width_vec]
                // into the thread-local buffer in `(jj_chunk, kk)`
                // contiguous order.
                for jj_chunk in 0..nc_chunks {
                    let jj = jj_chunk * LANES;
                    let dst_base = jj_chunk * kc * LANES;
                    for kk in 0..kc {
                        let src = b_ptr.add((k_block + kk) * n + j_block + jj);
                        let dst = packed_ptr.add(dst_base + kk * LANES);
                        let v = _mm512_loadu_ps(src);
                        _mm512_storeu_ps(dst, v);
                    }
                }

                // Main 8×32 micro-kernel loop over the packed buffer.
                // Each iteration of the jj loop processes TWO consecutive
                // jj_chunks (32 b columns), holding 16 c accumulators
                // live across the KC reduction.
                let m_main = (m / MR) * MR;
                let pair_end = (nc_chunks / 2) * 2;
                let mut i = 0;
                while i < m_main {
                    let mut jj_chunk = 0;
                    while jj_chunk < pair_end {
                        let jj0 = jj_chunk * LANES;
                        let jj1 = (jj_chunk + 1) * LANES;

                        let c00 = c_ptr.add(i * n + j_block + jj0);
                        let c01 = c_ptr.add(i * n + j_block + jj1);
                        let c10 = c_ptr.add((i + 1) * n + j_block + jj0);
                        let c11 = c_ptr.add((i + 1) * n + j_block + jj1);
                        let c20 = c_ptr.add((i + 2) * n + j_block + jj0);
                        let c21 = c_ptr.add((i + 2) * n + j_block + jj1);
                        let c30 = c_ptr.add((i + 3) * n + j_block + jj0);
                        let c31 = c_ptr.add((i + 3) * n + j_block + jj1);
                        let c40 = c_ptr.add((i + 4) * n + j_block + jj0);
                        let c41 = c_ptr.add((i + 4) * n + j_block + jj1);
                        let c50 = c_ptr.add((i + 5) * n + j_block + jj0);
                        let c51 = c_ptr.add((i + 5) * n + j_block + jj1);
                        let c60 = c_ptr.add((i + 6) * n + j_block + jj0);
                        let c61 = c_ptr.add((i + 6) * n + j_block + jj1);
                        let c70 = c_ptr.add((i + 7) * n + j_block + jj0);
                        let c71 = c_ptr.add((i + 7) * n + j_block + jj1);

                        let mut vc00 = _mm512_loadu_ps(c00);
                        let mut vc01 = _mm512_loadu_ps(c01);
                        let mut vc10 = _mm512_loadu_ps(c10);
                        let mut vc11 = _mm512_loadu_ps(c11);
                        let mut vc20 = _mm512_loadu_ps(c20);
                        let mut vc21 = _mm512_loadu_ps(c21);
                        let mut vc30 = _mm512_loadu_ps(c30);
                        let mut vc31 = _mm512_loadu_ps(c31);
                        let mut vc40 = _mm512_loadu_ps(c40);
                        let mut vc41 = _mm512_loadu_ps(c41);
                        let mut vc50 = _mm512_loadu_ps(c50);
                        let mut vc51 = _mm512_loadu_ps(c51);
                        let mut vc60 = _mm512_loadu_ps(c60);
                        let mut vc61 = _mm512_loadu_ps(c61);
                        let mut vc70 = _mm512_loadu_ps(c70);
                        let mut vc71 = _mm512_loadu_ps(c71);

                        let packed_chunk0_base =
                            packed_ptr.add(jj_chunk * kc * LANES);
                        let packed_chunk1_base =
                            packed_ptr.add((jj_chunk + 1) * kc * LANES);

                        for kk in 0..kc {
                            let vb0 = _mm512_loadu_ps(packed_chunk0_base.add(kk * LANES));
                            let vb1 = _mm512_loadu_ps(packed_chunk1_base.add(kk * LANES));

                            let kk_abs = k_block + kk;

                            let a0 = *a_ptr.add(i * k + kk_abs);
                            let va0 = _mm512_set1_ps(a0);
                            vc00 = _mm512_fmadd_ps(va0, vb0, vc00);
                            vc01 = _mm512_fmadd_ps(va0, vb1, vc01);

                            let a1 = *a_ptr.add((i + 1) * k + kk_abs);
                            let va1 = _mm512_set1_ps(a1);
                            vc10 = _mm512_fmadd_ps(va1, vb0, vc10);
                            vc11 = _mm512_fmadd_ps(va1, vb1, vc11);

                            let a2 = *a_ptr.add((i + 2) * k + kk_abs);
                            let va2 = _mm512_set1_ps(a2);
                            vc20 = _mm512_fmadd_ps(va2, vb0, vc20);
                            vc21 = _mm512_fmadd_ps(va2, vb1, vc21);

                            let a3 = *a_ptr.add((i + 3) * k + kk_abs);
                            let va3 = _mm512_set1_ps(a3);
                            vc30 = _mm512_fmadd_ps(va3, vb0, vc30);
                            vc31 = _mm512_fmadd_ps(va3, vb1, vc31);

                            let a4 = *a_ptr.add((i + 4) * k + kk_abs);
                            let va4 = _mm512_set1_ps(a4);
                            vc40 = _mm512_fmadd_ps(va4, vb0, vc40);
                            vc41 = _mm512_fmadd_ps(va4, vb1, vc41);

                            let a5 = *a_ptr.add((i + 5) * k + kk_abs);
                            let va5 = _mm512_set1_ps(a5);
                            vc50 = _mm512_fmadd_ps(va5, vb0, vc50);
                            vc51 = _mm512_fmadd_ps(va5, vb1, vc51);

                            let a6 = *a_ptr.add((i + 6) * k + kk_abs);
                            let va6 = _mm512_set1_ps(a6);
                            vc60 = _mm512_fmadd_ps(va6, vb0, vc60);
                            vc61 = _mm512_fmadd_ps(va6, vb1, vc61);

                            let a7 = *a_ptr.add((i + 7) * k + kk_abs);
                            let va7 = _mm512_set1_ps(a7);
                            vc70 = _mm512_fmadd_ps(va7, vb0, vc70);
                            vc71 = _mm512_fmadd_ps(va7, vb1, vc71);
                        }

                        _mm512_storeu_ps(c00, vc00);
                        _mm512_storeu_ps(c01, vc01);
                        _mm512_storeu_ps(c10, vc10);
                        _mm512_storeu_ps(c11, vc11);
                        _mm512_storeu_ps(c20, vc20);
                        _mm512_storeu_ps(c21, vc21);
                        _mm512_storeu_ps(c30, vc30);
                        _mm512_storeu_ps(c31, vc31);
                        _mm512_storeu_ps(c40, vc40);
                        _mm512_storeu_ps(c41, vc41);
                        _mm512_storeu_ps(c50, vc50);
                        _mm512_storeu_ps(c51, vc51);
                        _mm512_storeu_ps(c60, vc60);
                        _mm512_storeu_ps(c61, vc61);
                        _mm512_storeu_ps(c70, vc70);
                        _mm512_storeu_ps(c71, vc71);

                        jj_chunk += 2;
                    }

                    // Fallback 8×16 micro-kernel for an odd last jj_chunk.
                    while jj_chunk < nc_chunks {
                        let jj = jj_chunk * LANES;
                        let c_slot0 = c_ptr.add(i * n + j_block + jj);
                        let c_slot1 = c_ptr.add((i + 1) * n + j_block + jj);
                        let c_slot2 = c_ptr.add((i + 2) * n + j_block + jj);
                        let c_slot3 = c_ptr.add((i + 3) * n + j_block + jj);
                        let c_slot4 = c_ptr.add((i + 4) * n + j_block + jj);
                        let c_slot5 = c_ptr.add((i + 5) * n + j_block + jj);
                        let c_slot6 = c_ptr.add((i + 6) * n + j_block + jj);
                        let c_slot7 = c_ptr.add((i + 7) * n + j_block + jj);

                        let mut vc0 = _mm512_loadu_ps(c_slot0);
                        let mut vc1 = _mm512_loadu_ps(c_slot1);
                        let mut vc2 = _mm512_loadu_ps(c_slot2);
                        let mut vc3 = _mm512_loadu_ps(c_slot3);
                        let mut vc4 = _mm512_loadu_ps(c_slot4);
                        let mut vc5 = _mm512_loadu_ps(c_slot5);
                        let mut vc6 = _mm512_loadu_ps(c_slot6);
                        let mut vc7 = _mm512_loadu_ps(c_slot7);

                        let packed_chunk_base =
                            packed_ptr.add(jj_chunk * kc * LANES);

                        for kk in 0..kc {
                            let kk_abs = k_block + kk;
                            let a0 = *a_ptr.add(i * k + kk_abs);
                            let a1 = *a_ptr.add((i + 1) * k + kk_abs);
                            let a2 = *a_ptr.add((i + 2) * k + kk_abs);
                            let a3 = *a_ptr.add((i + 3) * k + kk_abs);
                            let a4 = *a_ptr.add((i + 4) * k + kk_abs);
                            let a5 = *a_ptr.add((i + 5) * k + kk_abs);
                            let a6 = *a_ptr.add((i + 6) * k + kk_abs);
                            let a7 = *a_ptr.add((i + 7) * k + kk_abs);

                            let vb = _mm512_loadu_ps(packed_chunk_base.add(kk * LANES));
                            let va0 = _mm512_set1_ps(a0);
                            let va1 = _mm512_set1_ps(a1);
                            let va2 = _mm512_set1_ps(a2);
                            let va3 = _mm512_set1_ps(a3);
                            let va4 = _mm512_set1_ps(a4);
                            let va5 = _mm512_set1_ps(a5);
                            let va6 = _mm512_set1_ps(a6);
                            let va7 = _mm512_set1_ps(a7);

                            vc0 = _mm512_fmadd_ps(va0, vb, vc0);
                            vc1 = _mm512_fmadd_ps(va1, vb, vc1);
                            vc2 = _mm512_fmadd_ps(va2, vb, vc2);
                            vc3 = _mm512_fmadd_ps(va3, vb, vc3);
                            vc4 = _mm512_fmadd_ps(va4, vb, vc4);
                            vc5 = _mm512_fmadd_ps(va5, vb, vc5);
                            vc6 = _mm512_fmadd_ps(va6, vb, vc6);
                            vc7 = _mm512_fmadd_ps(va7, vb, vc7);
                        }

                        _mm512_storeu_ps(c_slot0, vc0);
                        _mm512_storeu_ps(c_slot1, vc1);
                        _mm512_storeu_ps(c_slot2, vc2);
                        _mm512_storeu_ps(c_slot3, vc3);
                        _mm512_storeu_ps(c_slot4, vc4);
                        _mm512_storeu_ps(c_slot5, vc5);
                        _mm512_storeu_ps(c_slot6, vc6);
                        _mm512_storeu_ps(c_slot7, vc7);

                        jj_chunk += 1;
                    }

                    // Scalar tail for the 8-row micro-tile.
                    let mut jj_tail = j_width_vec;
                    while jj_tail < j_width {
                        let j_idx = j_block + jj_tail;
                        let mut c0 = *c_ptr.add(i * n + j_idx);
                        let mut c1 = *c_ptr.add((i + 1) * n + j_idx);
                        let mut c2 = *c_ptr.add((i + 2) * n + j_idx);
                        let mut c3 = *c_ptr.add((i + 3) * n + j_idx);
                        let mut c4 = *c_ptr.add((i + 4) * n + j_idx);
                        let mut c5 = *c_ptr.add((i + 5) * n + j_idx);
                        let mut c6 = *c_ptr.add((i + 6) * n + j_idx);
                        let mut c7 = *c_ptr.add((i + 7) * n + j_idx);
                        for kk in k_block..k_end {
                            let bv = *b_ptr.add(kk * n + j_idx);
                            c0 += *a_ptr.add(i * k + kk) * bv;
                            c1 += *a_ptr.add((i + 1) * k + kk) * bv;
                            c2 += *a_ptr.add((i + 2) * k + kk) * bv;
                            c3 += *a_ptr.add((i + 3) * k + kk) * bv;
                            c4 += *a_ptr.add((i + 4) * k + kk) * bv;
                            c5 += *a_ptr.add((i + 5) * k + kk) * bv;
                            c6 += *a_ptr.add((i + 6) * k + kk) * bv;
                            c7 += *a_ptr.add((i + 7) * k + kk) * bv;
                        }
                        *c_ptr.add(i * n + j_idx) = c0;
                        *c_ptr.add((i + 1) * n + j_idx) = c1;
                        *c_ptr.add((i + 2) * n + j_idx) = c2;
                        *c_ptr.add((i + 3) * n + j_idx) = c3;
                        *c_ptr.add((i + 4) * n + j_idx) = c4;
                        *c_ptr.add((i + 5) * n + j_idx) = c5;
                        *c_ptr.add((i + 6) * n + j_idx) = c6;
                        *c_ptr.add((i + 7) * n + j_idx) = c7;
                        jj_tail += 1;
                    }

                    i += MR;
                }

                // Intermediate 4×16 micro-kernel for m%8 >= 4 case.
                // Handles up to 4 rows at a time with 4 independent
                // c accumulators, which breaks the 1-row path's fma
                // dependency chain and amortizes the b load across 4
                // fmadds (down from 3 mem ops per fma to 1.5).
                if i + 4 <= m {
                    for jj_chunk in 0..nc_chunks {
                        let jj = jj_chunk * LANES;
                        let c_slot0 = c_ptr.add(i * n + j_block + jj);
                        let c_slot1 = c_ptr.add((i + 1) * n + j_block + jj);
                        let c_slot2 = c_ptr.add((i + 2) * n + j_block + jj);
                        let c_slot3 = c_ptr.add((i + 3) * n + j_block + jj);

                        let mut vc0 = _mm512_loadu_ps(c_slot0);
                        let mut vc1 = _mm512_loadu_ps(c_slot1);
                        let mut vc2 = _mm512_loadu_ps(c_slot2);
                        let mut vc3 = _mm512_loadu_ps(c_slot3);

                        let packed_chunk_base =
                            packed_ptr.add(jj_chunk * kc * LANES);

                        for kk in 0..kc {
                            let kk_abs = k_block + kk;
                            let a0 = *a_ptr.add(i * k + kk_abs);
                            let a1 = *a_ptr.add((i + 1) * k + kk_abs);
                            let a2 = *a_ptr.add((i + 2) * k + kk_abs);
                            let a3 = *a_ptr.add((i + 3) * k + kk_abs);

                            let vb = _mm512_loadu_ps(packed_chunk_base.add(kk * LANES));
                            let va0 = _mm512_set1_ps(a0);
                            let va1 = _mm512_set1_ps(a1);
                            let va2 = _mm512_set1_ps(a2);
                            let va3 = _mm512_set1_ps(a3);

                            vc0 = _mm512_fmadd_ps(va0, vb, vc0);
                            vc1 = _mm512_fmadd_ps(va1, vb, vc1);
                            vc2 = _mm512_fmadd_ps(va2, vb, vc2);
                            vc3 = _mm512_fmadd_ps(va3, vb, vc3);
                        }

                        _mm512_storeu_ps(c_slot0, vc0);
                        _mm512_storeu_ps(c_slot1, vc1);
                        _mm512_storeu_ps(c_slot2, vc2);
                        _mm512_storeu_ps(c_slot3, vc3);
                    }

                    // Scalar tail for the 4-row intermediate tile.
                    let mut jj_tail = j_width_vec;
                    while jj_tail < j_width {
                        let j_idx = j_block + jj_tail;
                        let mut c0 = *c_ptr.add(i * n + j_idx);
                        let mut c1 = *c_ptr.add((i + 1) * n + j_idx);
                        let mut c2 = *c_ptr.add((i + 2) * n + j_idx);
                        let mut c3 = *c_ptr.add((i + 3) * n + j_idx);
                        for kk in k_block..k_end {
                            let bv = *b_ptr.add(kk * n + j_idx);
                            c0 += *a_ptr.add(i * k + kk) * bv;
                            c1 += *a_ptr.add((i + 1) * k + kk) * bv;
                            c2 += *a_ptr.add((i + 2) * k + kk) * bv;
                            c3 += *a_ptr.add((i + 3) * k + kk) * bv;
                        }
                        *c_ptr.add(i * n + j_idx) = c0;
                        *c_ptr.add((i + 1) * n + j_idx) = c1;
                        *c_ptr.add((i + 2) * n + j_idx) = c2;
                        *c_ptr.add((i + 3) * n + j_idx) = c3;
                        jj_tail += 1;
                    }

                    i += 4;
                }

                // Final 1-row remainder (0-3 rows left after the 4×16
                // intermediate path).
                while i < m {
                    let a_row_ptr = a_ptr.add(i * k);

                    for jj_chunk in 0..nc_chunks {
                        let jj = jj_chunk * LANES;
                        let c_slot = c_ptr.add(i * n + j_block + jj);
                        let mut vc = _mm512_loadu_ps(c_slot);

                        let packed_chunk_base =
                            packed_ptr.add(jj_chunk * kc * LANES);

                        for kk in 0..kc {
                            let a_ik = *a_row_ptr.add(k_block + kk);
                            let va = _mm512_set1_ps(a_ik);
                            let vb = _mm512_loadu_ps(packed_chunk_base.add(kk * LANES));
                            vc = _mm512_fmadd_ps(va, vb, vc);
                        }

                        _mm512_storeu_ps(c_slot, vc);
                    }

                    // Scalar tail for remainder rows.
                    let mut jj_tail = j_width_vec;
                    while jj_tail < j_width {
                        let j_idx = j_block + jj_tail;
                        let mut c_val = *c_ptr.add(i * n + j_idx);
                        for kk in k_block..k_end {
                            c_val += *a_row_ptr.add(kk) * *b_ptr.add(kk * n + j_idx);
                        }
                        *c_ptr.add(i * n + j_idx) = c_val;
                        jj_tail += 1;
                    }

                    i += 1;
                }

                k_block += KC;
            }

            j_block += NC;
        }
    });
}

/// Simple unblocked 1×16 AVX-512 kernel for small matmuls where the
/// packing overhead of the full blocked kernel would dominate the
/// actual compute. Used by `matmul_f32_avx512_serial` as a fast path
/// for `m < 16`.
///
/// For small shapes the whole a + b + c working set typically fits
/// in L1/L2, so the hardware prefetcher handles the access pattern
/// well without explicit packing. Loop order is `for i { for kk {
/// for jj } }`, which gives the inner jj loop contiguous access to b
/// and c rows — the SIMD fmadd pattern the ZMM registers handle
/// natively.
///
/// # Safety
/// Requires the host to have the `avx512f` feature. Called only from
/// `matmul_f32_avx512_serial` which is itself `avx512f`-gated.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn matmul_f32_avx512_small(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps,
    };

    const LANES: usize = 16;
    let n_vec = (n / LANES) * LANES;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for i in 0..m {
        let c_row = c_ptr.add(i * n);
        let a_row = a_ptr.add(i * k);

        for kk in 0..k {
            let a_ik = *a_row.add(kk);
            let b_row = b_ptr.add(kk * n);
            let va = _mm512_set1_ps(a_ik);

            let mut jj = 0;
            while jj < n_vec {
                let vb = _mm512_loadu_ps(b_row.add(jj));
                let vc = _mm512_loadu_ps(c_row.add(jj));
                let vres = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(c_row.add(jj), vres);
                jj += LANES;
            }

            while jj < n {
                *c_row.add(jj) += a_ik * *b_row.add(jj);
                jj += 1;
            }
        }
    }
}

// ============================================================================
// AVX2 + FMA path
// ============================================================================

/// Public dispatcher for the AVX2 path. Decides whether to run the
/// serial blocked kernel or split `m` across `N` scoped worker
/// threads, where `N` is computed by [`worker_count_for_shape`].
/// Same parallelism strategy as the AVX-512 dispatcher above; only
/// the underlying kernel differs.
#[cfg(target_arch = "x86_64")]
fn matmul_f32_avx2_dispatch(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    if m < PARALLEL_THRESHOLD {
        unsafe {
            matmul_f32_avx2_serial(a, b, c, m, n, k);
        }
        return;
    }

    let workers = worker_count_for_shape(m, n, k);

    if workers == 1 {
        unsafe {
            matmul_f32_avx2_serial(a, b, c, m, n, k);
        }
        return;
    }

    let rows_per_worker = (m + workers - 1) / workers;

    std::thread::scope(|s| {
        let a_chunks = a.chunks(rows_per_worker * k);
        let c_chunks = c.chunks_mut(rows_per_worker * n);
        for (a_chunk, c_chunk) in a_chunks.zip(c_chunks) {
            let m_local = a_chunk.len() / k;
            s.spawn(move || {
                unsafe {
                    matmul_f32_avx2_serial(a_chunk, b, c_chunk, m_local, n, k);
                }
            });
        }
    });
}

/// Serial AVX2 + FMA cache-blocked GEMM with 4×8 register-blocked
/// micro-kernel. 8-wide f32 lanes instead of 16.
///
/// Same loop order and rationale as the AVX-512 kernel. Register
/// pressure: 4 c + 1 b + 4 a = 9 YMM registers. AVX2 has 16 YMM
/// registers, so still fits without spilling.
///
/// # Safety
/// Requires the host to have the `avx2` and `fma` features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_f32_avx2_serial(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    const NC: usize = 256;
    const KC: usize = 256;
    const MR: usize = 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let mut j_block = 0;
    while j_block < n {
        let j_end = (j_block + NC).min(n);
        let j_width = j_end - j_block;
        let j_width_vec = (j_width / LANES) * LANES;

        let mut k_block = 0;
        while k_block < k {
            let k_end = (k_block + KC).min(k);

            let m_main = (m / MR) * MR;
            let mut i = 0;
            while i < m_main {
                let mut jj = 0;
                while jj < j_width_vec {
                    let c_slot0 = c_ptr.add(i * n + j_block + jj);
                    let c_slot1 = c_ptr.add((i + 1) * n + j_block + jj);
                    let c_slot2 = c_ptr.add((i + 2) * n + j_block + jj);
                    let c_slot3 = c_ptr.add((i + 3) * n + j_block + jj);

                    let mut vc0 = _mm256_loadu_ps(c_slot0);
                    let mut vc1 = _mm256_loadu_ps(c_slot1);
                    let mut vc2 = _mm256_loadu_ps(c_slot2);
                    let mut vc3 = _mm256_loadu_ps(c_slot3);

                    for kk in k_block..k_end {
                        let a0 = *a_ptr.add(i * k + kk);
                        let a1 = *a_ptr.add((i + 1) * k + kk);
                        let a2 = *a_ptr.add((i + 2) * k + kk);
                        let a3 = *a_ptr.add((i + 3) * k + kk);

                        let vb = _mm256_loadu_ps(b_ptr.add(kk * n + j_block + jj));
                        let va0 = _mm256_set1_ps(a0);
                        let va1 = _mm256_set1_ps(a1);
                        let va2 = _mm256_set1_ps(a2);
                        let va3 = _mm256_set1_ps(a3);

                        vc0 = _mm256_fmadd_ps(va0, vb, vc0);
                        vc1 = _mm256_fmadd_ps(va1, vb, vc1);
                        vc2 = _mm256_fmadd_ps(va2, vb, vc2);
                        vc3 = _mm256_fmadd_ps(va3, vb, vc3);
                    }

                    _mm256_storeu_ps(c_slot0, vc0);
                    _mm256_storeu_ps(c_slot1, vc1);
                    _mm256_storeu_ps(c_slot2, vc2);
                    _mm256_storeu_ps(c_slot3, vc3);

                    jj += LANES;
                }

                while jj < j_width {
                    let j_idx = j_block + jj;
                    let mut c0 = *c_ptr.add(i * n + j_idx);
                    let mut c1 = *c_ptr.add((i + 1) * n + j_idx);
                    let mut c2 = *c_ptr.add((i + 2) * n + j_idx);
                    let mut c3 = *c_ptr.add((i + 3) * n + j_idx);
                    for kk in k_block..k_end {
                        let bv = *b_ptr.add(kk * n + j_idx);
                        c0 += *a_ptr.add(i * k + kk) * bv;
                        c1 += *a_ptr.add((i + 1) * k + kk) * bv;
                        c2 += *a_ptr.add((i + 2) * k + kk) * bv;
                        c3 += *a_ptr.add((i + 3) * k + kk) * bv;
                    }
                    *c_ptr.add(i * n + j_idx) = c0;
                    *c_ptr.add((i + 1) * n + j_idx) = c1;
                    *c_ptr.add((i + 2) * n + j_idx) = c2;
                    *c_ptr.add((i + 3) * n + j_idx) = c3;
                    jj += 1;
                }

                i += MR;
            }

            while i < m {
                let c_row_ptr = c_ptr.add(i * n + j_block);
                let a_row_ptr = a_ptr.add(i * k);

                for kk in k_block..k_end {
                    let a_ik = *a_row_ptr.add(kk);
                    let b_row_ptr = b_ptr.add(kk * n + j_block);
                    let va = _mm256_set1_ps(a_ik);

                    let mut jj = 0;
                    while jj < j_width_vec {
                        let vb = _mm256_loadu_ps(b_row_ptr.add(jj));
                        let vc = _mm256_loadu_ps(c_row_ptr.add(jj));
                        let vres = _mm256_fmadd_ps(va, vb, vc);
                        _mm256_storeu_ps(c_row_ptr.add(jj), vres);
                        jj += LANES;
                    }

                    while jj < j_width {
                        *c_row_ptr.add(jj) += a_ik * *b_row_ptr.add(jj);
                        jj += 1;
                    }
                }

                i += 1;
            }

            k_block += KC;
        }

        j_block += NC;
    }
}

// ============================================================================
// wasm32 SIMD128 path
// ============================================================================

/// Cache-blocked f32 GEMM for `wasm32-unknown-unknown` built with
/// `-C target-feature=+simd128`. Mirrors the structure of
/// [`matmul_f32_avx2_serial`] above: same `(KC, NC)` outer cache
/// blocking, same 4-row register-blocked main micro-kernel, same
/// scalar tail strategy. The only differences are:
///
///   - **Lane width**: 4 f32 lanes per `v128` register instead of 8
///     per `__m256`. The main micro-kernel processes a 4×4 c tile
///     per inner iteration (4 c accumulators, 1 shared b vector,
///     4 a broadcasts).
///   - **No native fma**: wasm32's relaxed-simd `f32x4_relaxed_madd`
///     would require an additional `relaxed-simd` target feature
///     that isn't yet ubiquitous in browsers, so we use the strict
///     `add(mul(a, b), c)` form instead. The performance hit vs a
///     true fma is small in practice because the load/broadcast
///     pressure dominates.
///
/// Cache blocking is the biggest single win over the auto-vectorised
/// scalar path: without it, every outer `i` row re-touches the entire
/// `(k, n)` b matrix, which thrashes wasm's linear-memory cache for
/// any non-trivial shape. With `(KC, NC) = (256, 256)`, a `b` panel
/// occupies 256 KB and stays resident across all `m` row iterations
/// of the inner kernel.
///
/// # Safety
/// Requires the build to enable the `simd128` target feature.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn matmul_f32_wasm_simd128(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    use std::arch::wasm32::{
        f32x4_add, f32x4_mul, f32x4_splat, v128, v128_load, v128_store,
    };

    #[inline(always)]
    unsafe fn fma(a: v128, b: v128, c: v128) -> v128 {
        f32x4_add(f32x4_mul(a, b), c)
    }

    const LANES: usize = 4;
    const NC: usize = 256;
    const KC: usize = 256;
    const MR: usize = 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let mut j_block = 0;
    while j_block < n {
        let j_end = (j_block + NC).min(n);
        let j_width = j_end - j_block;
        let j_width_vec = (j_width / LANES) * LANES;

        let mut k_block = 0;
        while k_block < k {
            let k_end = (k_block + KC).min(k);

            let m_main = (m / MR) * MR;
            let pair_end = (j_width_vec / (2 * LANES)) * (2 * LANES);
            let mut i = 0;
            while i < m_main {
                let mut jj = 0;
                while jj < pair_end {
                    let jj1 = jj + LANES;

                    let c00 = c_ptr.add(i * n + j_block + jj);
                    let c01 = c_ptr.add(i * n + j_block + jj1);
                    let c10 = c_ptr.add((i + 1) * n + j_block + jj);
                    let c11 = c_ptr.add((i + 1) * n + j_block + jj1);
                    let c20 = c_ptr.add((i + 2) * n + j_block + jj);
                    let c21 = c_ptr.add((i + 2) * n + j_block + jj1);
                    let c30 = c_ptr.add((i + 3) * n + j_block + jj);
                    let c31 = c_ptr.add((i + 3) * n + j_block + jj1);

                    let mut vc00 = v128_load(c00 as *const v128);
                    let mut vc01 = v128_load(c01 as *const v128);
                    let mut vc10 = v128_load(c10 as *const v128);
                    let mut vc11 = v128_load(c11 as *const v128);
                    let mut vc20 = v128_load(c20 as *const v128);
                    let mut vc21 = v128_load(c21 as *const v128);
                    let mut vc30 = v128_load(c30 as *const v128);
                    let mut vc31 = v128_load(c31 as *const v128);

                    for kk in k_block..k_end {
                        let vb0 = v128_load(b_ptr.add(kk * n + j_block + jj) as *const v128);
                        let vb1 = v128_load(b_ptr.add(kk * n + j_block + jj1) as *const v128);

                        let a0 = *a_ptr.add(i * k + kk);
                        let a1 = *a_ptr.add((i + 1) * k + kk);
                        let a2 = *a_ptr.add((i + 2) * k + kk);
                        let a3 = *a_ptr.add((i + 3) * k + kk);

                        let va0 = f32x4_splat(a0);
                        let va1 = f32x4_splat(a1);
                        let va2 = f32x4_splat(a2);
                        let va3 = f32x4_splat(a3);

                        vc00 = fma(va0, vb0, vc00);
                        vc01 = fma(va0, vb1, vc01);
                        vc10 = fma(va1, vb0, vc10);
                        vc11 = fma(va1, vb1, vc11);
                        vc20 = fma(va2, vb0, vc20);
                        vc21 = fma(va2, vb1, vc21);
                        vc30 = fma(va3, vb0, vc30);
                        vc31 = fma(va3, vb1, vc31);
                    }

                    v128_store(c00 as *mut v128, vc00);
                    v128_store(c01 as *mut v128, vc01);
                    v128_store(c10 as *mut v128, vc10);
                    v128_store(c11 as *mut v128, vc11);
                    v128_store(c20 as *mut v128, vc20);
                    v128_store(c21 as *mut v128, vc21);
                    v128_store(c30 as *mut v128, vc30);
                    v128_store(c31 as *mut v128, vc31);

                    jj += 2 * LANES;
                }

                while jj < j_width_vec {
                    let c_slot0 = c_ptr.add(i * n + j_block + jj);
                    let c_slot1 = c_ptr.add((i + 1) * n + j_block + jj);
                    let c_slot2 = c_ptr.add((i + 2) * n + j_block + jj);
                    let c_slot3 = c_ptr.add((i + 3) * n + j_block + jj);

                    let mut vc0 = v128_load(c_slot0 as *const v128);
                    let mut vc1 = v128_load(c_slot1 as *const v128);
                    let mut vc2 = v128_load(c_slot2 as *const v128);
                    let mut vc3 = v128_load(c_slot3 as *const v128);

                    for kk in k_block..k_end {
                        let vb = v128_load(b_ptr.add(kk * n + j_block + jj) as *const v128);

                        let a0 = *a_ptr.add(i * k + kk);
                        let a1 = *a_ptr.add((i + 1) * k + kk);
                        let a2 = *a_ptr.add((i + 2) * k + kk);
                        let a3 = *a_ptr.add((i + 3) * k + kk);

                        let va0 = f32x4_splat(a0);
                        let va1 = f32x4_splat(a1);
                        let va2 = f32x4_splat(a2);
                        let va3 = f32x4_splat(a3);

                        vc0 = fma(va0, vb, vc0);
                        vc1 = fma(va1, vb, vc1);
                        vc2 = fma(va2, vb, vc2);
                        vc3 = fma(va3, vb, vc3);
                    }

                    v128_store(c_slot0 as *mut v128, vc0);
                    v128_store(c_slot1 as *mut v128, vc1);
                    v128_store(c_slot2 as *mut v128, vc2);
                    v128_store(c_slot3 as *mut v128, vc3);

                    jj += LANES;
                }

                while jj < j_width {
                    let j_idx = j_block + jj;
                    let mut c0 = *c_ptr.add(i * n + j_idx);
                    let mut c1 = *c_ptr.add((i + 1) * n + j_idx);
                    let mut c2 = *c_ptr.add((i + 2) * n + j_idx);
                    let mut c3 = *c_ptr.add((i + 3) * n + j_idx);
                    for kk in k_block..k_end {
                        let bv = *b_ptr.add(kk * n + j_idx);
                        c0 += *a_ptr.add(i * k + kk) * bv;
                        c1 += *a_ptr.add((i + 1) * k + kk) * bv;
                        c2 += *a_ptr.add((i + 2) * k + kk) * bv;
                        c3 += *a_ptr.add((i + 3) * k + kk) * bv;
                    }
                    *c_ptr.add(i * n + j_idx) = c0;
                    *c_ptr.add((i + 1) * n + j_idx) = c1;
                    *c_ptr.add((i + 2) * n + j_idx) = c2;
                    *c_ptr.add((i + 3) * n + j_idx) = c3;
                    jj += 1;
                }

                i += MR;
            }

            // 1-row remainder for `m % MR` rows.
            while i < m {
                let c_row_ptr = c_ptr.add(i * n + j_block);
                let a_row_ptr = a_ptr.add(i * k);

                for kk in k_block..k_end {
                    let a_ik = *a_row_ptr.add(kk);
                    let b_row_ptr = b_ptr.add(kk * n + j_block);
                    let va = f32x4_splat(a_ik);

                    let mut jj = 0;
                    while jj < j_width_vec {
                        let vb = v128_load(b_row_ptr.add(jj) as *const v128);
                        let vc = v128_load(c_row_ptr.add(jj) as *const v128);
                        let vres = fma(va, vb, vc);
                        v128_store(c_row_ptr.add(jj) as *mut v128, vres);
                        jj += LANES;
                    }

                    while jj < j_width {
                        *c_row_ptr.add(jj) += a_ik * *b_row_ptr.add(jj);
                        jj += 1;
                    }
                }

                i += 1;
            }

            k_block += KC;
        }

        j_block += NC;
    }
}

// ============================================================================
// Linear forward kernel
// ============================================================================

/// Fused linear forward kernel: `y[m, n] = x[m, k] @ w^T[k, n] + bias[n]`.
///
/// `w` is stored in its untransposed `[out_features, in_features] =
/// [n, k]` PyTorch layout. This kernel computes each `y[i, o]` as a
/// dot product of two contiguous rows `x[i, ..]` and `w[o, ..]`, then
/// adds bias.
///
/// Dispatches to an AVX-512 kernel with 4 parallel accumulators on
/// x86_64 hosts that support `avx512f`; otherwise uses a scalar
/// auto-vectorised triple loop. The AVX-512 path breaks the fma
/// dependency chain into 4 parallel streams so both SKX FMA ports
/// can dispatch at full rate.
#[inline]
pub fn linear_f32(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(x.len(), m * k, "linear_f32: x length mismatch");
    debug_assert_eq!(w.len(), n * k, "linear_f32: w length mismatch");
    debug_assert_eq!(y.len(), m * n, "linear_f32: y length mismatch");
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), n, "linear_f32: bias length mismatch");
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            unsafe {
                linear_f32_avx512(x, w, bias, y, m, k, n);
            }
            return;
        }
    }

    linear_f32_scalar(x, w, bias, y, m, k, n);
}

/// Scalar fallback for `linear_f32`. Auto-vectorises to 4-wide SSE2
/// by default; wider with `target-feature` flags.
fn linear_f32_scalar(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];
        let y_row = &mut y[i * n..(i + 1) * n];

        for o in 0..n {
            let w_row = &w[o * k..(o + 1) * k];

            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += x_row[kk] * w_row[kk];
            }
            y_row[o] = sum;
        }

        if let Some(b) = bias {
            for o in 0..n {
                y_row[o] += b[o];
            }
        }
    }
}

/// Explicit AVX-512 linear kernel. 4-way unrolled ZMM accumulation
/// for the dot product breaks the fma dependency chain so both SKX
/// FMA ports can dispatch at full rate.
///
/// Inner loop processes 64 f32 elements per iteration (4 vectors ×
/// 16 lanes each). The 4 accumulators are reduced to 1 via element-
/// wise add at the end, then `_mm512_reduce_add_ps` horizontal-sums
/// the final 16 lanes to a single f32 per output column.
///
/// # Safety
/// Requires the host to have the `avx512f` feature.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn linear_f32_avx512(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps,
        _mm512_setzero_ps,
    };

    const LANES: usize = 16;
    const UNROLL: usize = 4;
    const STRIDE: usize = LANES * UNROLL; // 64

    let x_ptr = x.as_ptr();
    let w_ptr = w.as_ptr();
    let y_ptr = y.as_mut_ptr();

    let main_end = (k / STRIDE) * STRIDE;
    let vec_end = (k / LANES) * LANES;

    for i in 0..m {
        let x_row = x_ptr.add(i * k);
        let y_row = y_ptr.add(i * n);

        for o in 0..n {
            let w_row = w_ptr.add(o * k);

            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            let mut kk = 0;
            while kk < main_end {
                let vx0 = _mm512_loadu_ps(x_row.add(kk));
                let vw0 = _mm512_loadu_ps(w_row.add(kk));
                acc0 = _mm512_fmadd_ps(vx0, vw0, acc0);

                let vx1 = _mm512_loadu_ps(x_row.add(kk + LANES));
                let vw1 = _mm512_loadu_ps(w_row.add(kk + LANES));
                acc1 = _mm512_fmadd_ps(vx1, vw1, acc1);

                let vx2 = _mm512_loadu_ps(x_row.add(kk + 2 * LANES));
                let vw2 = _mm512_loadu_ps(w_row.add(kk + 2 * LANES));
                acc2 = _mm512_fmadd_ps(vx2, vw2, acc2);

                let vx3 = _mm512_loadu_ps(x_row.add(kk + 3 * LANES));
                let vw3 = _mm512_loadu_ps(w_row.add(kk + 3 * LANES));
                acc3 = _mm512_fmadd_ps(vx3, vw3, acc3);

                kk += STRIDE;
            }

            while kk < vec_end {
                let vx = _mm512_loadu_ps(x_row.add(kk));
                let vw = _mm512_loadu_ps(w_row.add(kk));
                acc0 = _mm512_fmadd_ps(vx, vw, acc0);
                kk += LANES;
            }

            let acc01 = _mm512_add_ps(acc0, acc1);
            let acc23 = _mm512_add_ps(acc2, acc3);
            let acc_final = _mm512_add_ps(acc01, acc23);
            let mut sum = _mm512_reduce_add_ps(acc_final);

            while kk < k {
                sum += *x_row.add(kk) * *w_row.add(kk);
                kk += 1;
            }

            *y_row.add(o) = sum;
        }

        if let Some(b) = bias {
            let b_ptr = b.as_ptr();
            for o in 0..n {
                *y_row.add(o) += *b_ptr.add(o);
            }
        }
    }
}

/// Alias for compatibility. `matmul` already auto-dispatches.
#[cfg(target_arch = "x86_64")]
pub fn matmul_simd(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
    matmul(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_matmul() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = matmul(&a, &b);

        assert_eq!(c[&[0, 0]], 19.0);
        assert_eq!(c[&[0, 1]], 22.0);
        assert_eq!(c[&[1, 0]], 43.0);
        assert_eq!(c[&[1, 1]], 50.0);
    }

    #[test]
    fn test_matmul_f32_kernel() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_f32_scalar_kernel() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0f32; 4];
        matmul_f32_scalar(&a, &b, &mut c, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_f32_large_consistency() {
        // Verify the blocked + parallel dispatch matches the scalar
        // reference. m=20 is above PARALLEL_THRESHOLD so the scoped
        // threads are exercised; n=257 and k=257 each spill into a
        // second outer block; the last 17 columns hit the SIMD tail.
        let m = 20;
        let k = 257;
        let n = 257;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos()).collect();

        let mut c_dispatch = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c_dispatch, m, n, k);

        let mut c_ref = vec![0.0f32; m * n];
        matmul_f32_scalar(&a, &b, &mut c_ref, m, n, k);

        for i in 0..m * n {
            let diff = (c_dispatch[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-3,
                "mismatch at index {}: dispatch={} scalar={} diff={}",
                i,
                c_dispatch[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn test_matmul_f32_small_no_parallel() {
        // Below PARALLEL_THRESHOLD: should use the serial path.
        let m = 5;
        let k = 7;
        let n = 17;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos()).collect();

        let mut c_dispatch = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c_dispatch, m, n, k);

        let mut c_ref = vec![0.0f32; m * n];
        matmul_f32_scalar(&a, &b, &mut c_ref, m, n, k);

        for i in 0..m * n {
            let diff = (c_dispatch[i] - c_ref[i]).abs();
            assert!(
                diff < 1e-4,
                "mismatch at index {}: dispatch={} scalar={} diff={}",
                i,
                c_dispatch[i],
                c_ref[i],
                diff
            );
        }
    }

    #[test]
    fn test_linear_f32_kernel() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let bias = vec![10.0, 20.0, 30.0, 40.0];
        let mut y = vec![0.0f32; 8];
        linear_f32(&x, &w, Some(&bias), &mut y, 2, 3, 4);
        assert_eq!(y, vec![11.0, 22.0, 33.0, 46.0, 14.0, 25.0, 36.0, 55.0]);
    }
}