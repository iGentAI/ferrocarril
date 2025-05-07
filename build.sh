#!/bin/bash
# Build and run the test_binary_loader.rs program

set -e

echo "Building test_binary_loader..."

# Get the repo root directory
REPO_ROOT=$(pwd)

# Build the ferrocarril-core library first if needed
if [[ ! -f "${REPO_ROOT}/ferrocarril-core/target/debug/libferrocarril_core.rlib" ]]; then
    echo "Building ferrocarril-core library..."
    (cd "${REPO_ROOT}/ferrocarril-core" && cargo build)
fi

# Create a helper module file to link the crates
echo "Creating helper module for linking..."
cat > lib.rs << EOF
//! Helper module for linking
#[path = "${REPO_ROOT}/ferrocarril-core/src/weights_binary.rs"]
pub mod weights_binary;

#[path = "${REPO_ROOT}/ferrocarril-core/src/tensor.rs"]
pub mod tensor;

#[path = "${REPO_ROOT}/ferrocarril-core/src/weights.rs"]
pub mod weights;

pub use ferrocarril_core::{Parameter, FerroError};
EOF

# Compile the test program with explicit module paths
rustc -o test_binary_loader test_binary_loader.rs \
  --extern ferrocarril_core="${REPO_ROOT}/ferrocarril-core/target/debug/libferrocarril_core.rlib" \
  --extern serde_json="${REPO_ROOT}/ferrocarril-core/target/debug/deps/libserde_json-*.rlib" \
  --extern serde="${REPO_ROOT}/ferrocarril-core/target/debug/deps/libserde-*.rlib" \
  -L "${REPO_ROOT}/ferrocarril-core/target/debug/deps" \
  -L "${REPO_ROOT}/ferrocarril-core/target/debug/"

echo "Running test_binary_loader..."
./test_binary_loader