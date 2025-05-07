# Simple Architecture Fix

## Current Problem

We have a circular dependency:
- `ferrocarril-core` imports types from `ferrocarril-nn` (TextEncoder, ProsodyPredictor, etc.)
- `ferrocarril-nn` depends on `ferrocarril-core` for basic types (Tensor, Parameter, etc.)

## Simple Solution

Keep the existing 3 crate structure, but fix the dependency direction:

1. **ferrocarril-core**: Basic types, traits, and utilities
   - Tensor, Parameter, Config
   - LoadWeights traits
   - NO model implementation details
   - NO imports from ferrocarril-nn

2. **ferrocarril-nn**: Neural network components
   - All neural network layers (Linear, LSTM, Conv, etc.)
   - Complete model implementations (TextEncoder, ProsodyPredictor, Decoder)
   - DSP components
   - Depends only on ferrocarril-core

3. **ferrocarril**: Main executable and pipeline
   - Integrates all components
   - Handles G2P, inference pipeline, CLI
   - Depends on both ferrocarril-core and ferrocarril-nn

## Implementation Steps

1. Remove all imports of `ferrocarril-nn` from `ferrocarril-core`
2. Move the `FerroModel` struct from core to the main ferrocarril crate
3. Keep Phonesis integration at the main crate level

That's it. No need for ferrocarril-models, ferrocarril-inference, or ferrocarril-pipeline.