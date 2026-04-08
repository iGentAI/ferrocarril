//! Standalone microbenchmark for `ferrocarril_core::ops::matmul::matmul_f32`.
//!
//! Measures raw f32 GEMM throughput at the shapes the Kokoro
//! Generator hits on its hot path. Run with:
//!
//! ```sh
//! cargo build --release --bin bench_matmul -p ferrocarril-core
//! ./target/release/bench_matmul
//! ```
//!
//! A GFLOPS number near the theoretical peak (~40 GFLOPS for AVX-512
//! scalar-width on this Xeon Platinum 8175M) means LLVM has vectorised
//! the kernel. ~3 GFLOPS means the inner loop is running scalar.

use ferrocarril_core::ops::matmul::matmul_f32;
use std::time::Instant;

fn bench_shape(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    // Fill with something other than zero so the compiler can't
    // constant-fold the whole thing away.
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32).cos()).collect();
    let mut c: Vec<f32> = vec![0.0f32; m * n];

    // Warm-up: prime the caches and force any JIT / first-call
    // overheads out of the measurement window.
    for _ in 0..3 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        matmul_f32(&a, &b, &mut c, m, n, k);
    }

    let start = Instant::now();
    for _ in 0..iters {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        matmul_f32(&a, &b, &mut c, m, n, k);
    }
    let elapsed = start.elapsed();

    // Prevent dead-code elimination by using c.
    let checksum: f32 = c.iter().take(16).sum();
    // Each matmul does `m * n * k` multiply-adds; count each as 2 ops
    // (one mul + one add), which is the convention in the literature.
    let ops_per_call = 2.0 * (m * n * k) as f64;
    let total_ops = ops_per_call * iters as f64;
    let secs = elapsed.as_secs_f64();
    let gflops = total_ops / secs / 1e9;
    let ms_per_call = secs * 1000.0 / iters as f64;

    println!(
        "{:30} m={:5} n={:5} k={:5}  iters={:4}  {:6.1} ms/call  {:6.2} GFLOPS  (checksum={:.3})",
        label, m, n, k, iters, ms_per_call, gflops, checksum,
    );
}

fn main() {
    println!("ferrocarril matmul_f32 microbenchmark");
    println!("=====================================");

    // Small matmul: typical BERT attention projection (1 batch × 3
    // tokens × 768 hidden → 768 out_features).
    bench_shape("small (BERT Q proj)      ", 3, 768, 768, 1000);

    // Medium: Generator conv_post (im2col -> matmul (22, 3361, 64*7=448))
    bench_shape("medium (Generator post)  ", 22, 3361, 448, 50);

    // Large: Generator Stage 1 resblock conv
    //   C_out=256, L_out=560, C_in*K = 256*11 = 2816
    bench_shape("large  (Gen stage 1 conv)", 256, 560, 2816, 20);

    // Extra large: Generator Stage 2 resblock conv
    //   C_out=128, L_out=3360, C_in*K = 128*11 = 1408
    bench_shape("xl     (Gen stage 2 conv)", 128, 3360, 1408, 20);

    println!();
    println!("Theoretical peak on this host:");
    println!("  Scalar FMA  @ 2.5 GHz           ≈  2.5 GFLOPS");
    println!("  SSE2  4-wide FMA @ 2.5 GHz      ≈ 10.0 GFLOPS");
    println!("  AVX2  8-wide FMA @ 2.5 GHz      ≈ 20.0 GFLOPS");
    println!("  AVX-512 16-wide FMA @ 2.5 GHz   ≈ 40.0 GFLOPS");
    println!("  AVX-512 + 2 FMA ports           ≈ 80.0 GFLOPS");
}