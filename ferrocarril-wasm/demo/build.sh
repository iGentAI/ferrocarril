#!/usr/bin/env bash
# Build the ferrocarril-wasm crate and emit a browser-ready `pkg/`
# directory next to this script.
#
# Prerequisites:
#   - rustup + the wasm32-unknown-unknown target installed
#   - wasm-bindgen-cli 0.2.100 on PATH (must match the crate dep)
#   - (optional) wasm-opt for size optimisation
#
# Usage: ./demo/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${CRATE_DIR}/.." && pwd)"
PKG_DIR="${SCRIPT_DIR}/pkg"

echo ">> Workspace root: ${WORKSPACE_ROOT}"
echo ">> Target pkg dir: ${PKG_DIR}"

cd "${WORKSPACE_ROOT}"

echo ">> Stage 1: cargo build --release --target wasm32-unknown-unknown --lib -p ferrocarril-wasm (+simd128)"
# `+simd128` enables the hand-written wasm SIMD matmul hot path in
# ferrocarril-core/src/ops/matmul.rs. Without this flag the build
# falls through to a scalar path that is ~2× slower on inference.
RUSTFLAGS="-C target-feature=+simd128" \
    cargo build \
    --release \
    --target wasm32-unknown-unknown \
    --lib \
    -p ferrocarril-wasm

WASM_INPUT="${WORKSPACE_ROOT}/target/wasm32-unknown-unknown/release/ferrocarril_wasm.wasm"
if [[ ! -f "${WASM_INPUT}" ]]; then
    echo "!! Expected ${WASM_INPUT} to exist after cargo build" >&2
    exit 1
fi

mkdir -p "${PKG_DIR}"

echo ">> Stage 2: wasm-bindgen --target web"
wasm-bindgen \
    --target web \
    --out-dir "${PKG_DIR}" \
    "${WASM_INPUT}"

if command -v wasm-opt >/dev/null 2>&1; then
    echo ">> Stage 3: wasm-opt -Oz (optional)"
    wasm-opt -Oz \
        --enable-simd \
        -o "${PKG_DIR}/ferrocarril_wasm_bg.wasm" \
        "${PKG_DIR}/ferrocarril_wasm_bg.wasm"
else
    echo ">> wasm-opt not on PATH, skipping size optimisation"
fi

ls -lh "${PKG_DIR}"
WASM_OUT="${PKG_DIR}/ferrocarril_wasm_bg.wasm"
if [[ -f "${WASM_OUT}" ]] && command -v gzip >/dev/null 2>&1; then
    RAW=$(stat -c%s "${WASM_OUT}" 2>/dev/null || stat -f%z "${WASM_OUT}")
    GZ=$(gzip -c "${WASM_OUT}" | wc -c)
    echo ">> Final wasm size: ${RAW} bytes raw, ${GZ} bytes gzipped"
fi
echo ">> Done. Serve this directory (or its parent) with a plain HTTP server"
echo "   and open demo/index.html in a modern browser."