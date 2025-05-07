# Ferrocarril Implementation Status (as of May 6, 2025)

This document tracks the current status of the Ferrocarril project, a pure Rust implementation of the kokoro TTS inference pipeline.

## Overall Architecture

The codebase is structured as a Rust workspace with multiple crates:
- `ferrocarril-core`: Basic tensor operations, weight loading, and G2P integration.
- `ferrocarril-nn`: Neural network components (Linear, LSTM, Conv, AdaIN, etc.).
- `ferrocarril-dsp`: Digital signal processing (STFT, windows, audio output).
- `ferrocarril` (main): Integration and executable for demos/testing.

The structure is modular and designed for zero external dependencies, with Phonesis integrated as an obligate dependency for G2P functionality.

## Recent Improvements for Functional Correctness

In May 2025, the project has shifted focus from merely matching tensor shapes to ensuring **functional correctness** - components that actually transform data meaningfully.

1. **Critical Implementation Principles**
   - ✅ **Zero Tolerance for Silent Error Masking**: Replaced fallbacks with explicit assertions
   - ✅ **No Fake Weights**: All components must load real weights for proper testing
   - ✅ **No Dimensional Fudging**: Components no longer silently reshape on dimension mismatch
   - ✅ **Honor the Reference**: All components match exact shape transformations of Kokoro
   - ✅ **Fail Fast**: Components now panic with clear error messages when deviations occur
   - ✅ **Functional Correctness**: All components validate non-zero outputs to verify meaningful data transformation

2. **High-Priority Fixes Implemented**
   - ✅ **Alignment Tensor Creation**: Fixed tensor creation to properly expand durations into a [T, sum(durations)] matrix
   - ✅ **AdaIN Configuration**: Set `affine=false` in InstanceNorm1d to match Kokoro exactly
   - ✅ **Voice Embedding Handling**: Fixed embedding processing from [510, 1, 256] to [1, 256] and splitting into reference/style
   - ✅ **Reflection Padding in Decoder**: Fixed padding direction to apply on left side only
   - ✅ **Verification Testing**: Added tests that load real weights and verify non-zero, statistically reasonable outputs

3. **Comprehensive Test Structure**
   - ✅ **Real Weight Testing**: All component tests now load actual weights from model files
   - ✅ **Statistical Validation**: Tests verify that outputs have expected statistical properties
   - ✅ **End-to-End Testing**: Added tests for the full pipeline from phonemes to audio
   - ✅ **Integration Testing**: Added tests for voice embedding handling and alignment creation

## Completed Components

1.  **Core Infrastructure** (`ferrocarril-core`)
    -   ✅ Tensor module (`tensor.rs`): Implemented with shape handling, indexing, and basic ops (reshape, transpose, cat). Enhanced with additional utilities (add, scalar_div, map, concat, slice). Basic tests pass.
    -   ✅ Operations module (`ops/`): Implemented element-wise add/mul, transpose, basic matrix multiplication. Basic tests pass.
    -   ✅ PyTorch Weight Loading (`weights.rs`): Implemented parser for legacy pickle and zip-based PyTorch formats. Supports memory-mapping. Includes `LoadWeights` trait. Config loading from `config.json` added. Basic tests pass for config and mock loading.
    -   ✅ Binary Weight Loading (`weights_binary.rs`): Implemented loader for converted binary weights with JSON metadata. Supports both model weights and voice files. Successfully tested with Kokoro weights.
    -   ✅ Python Weight Converter: Created `weight_converter.py` script to convert PyTorch weights and voice files to a simple binary format with JSON metadata.
    -   ✅ Build system: Workspace compiles successfully with tests passing.
    -   ✅ G2P Integration: **COMPLETE** - Direct integration with Phonesis G2P for text-to-phoneme conversion, with all tests passing. The G2P component is now a proper obligate dependency rather than an optional feature.

2.  **Neural Network Components** (`ferrocarril-nn`)
    -   ✅ Linear layer (`linear.rs`): Fully implemented with bias and `LoadWeights` trait.
    -   ✅ Activation functions (`activation.rs`): Implemented ReLU, LeakyReLU, Sigmoid, Tanh, Snake.
    -   ✅ Conv1d layer (`conv.rs`): Fully implemented with stride, padding, dilation, groups, and `LoadWeights` trait.
    -   ✅ ConvTranspose1d layer (`conv_transpose.rs`): Implemented for upsampling operations with stride, padding, and groups.
    -   ✅ LSTM layer (`lstm.rs`): **COMPLETE** - Fully implemented bidirectional LSTM with proper forward/reverse processing and concatenation of outputs.
    -   ✅ AdaIN module (`adain.rs`): **COMPLETE** - Implemented with `affine=false` for InstanceNorm1d to match Kokoro exactly.
    -   ✅ `text_encoder.rs`: Implemented TextEncoder including embedding, LayerNorm, ConvBlock, and bidirectional LSTM logic.
    -   ✅ `prosody/` module: **COMPLETE** - Fixed style dimension handling in energy pooling and corrected tensor transposition for LSTM input.
    -   ✅ `vocoder/` module:
        - ✅ `SineGen`: Implemented with support for multiple harmonics.
        - ✅ `SourceModuleHnNSF`: Implemented source module with voice conditioning.
        - ✅ `AdainResBlk1`: **COMPLETE** - Implemented upsampling in both main and residual paths with proper shortcut handling.
        - ✅ `Generator`: **COMPLETE** - Full implementation with upsampling stack and waveform generation.
        - ✅ `Decoder`: **COMPLETE** - Fixed reflection padding to apply on left side only.
        - ✅ `UpSample1d`: Implemented with nearest neighbor interpolation.

3.  **DSP Components** (`ferrocarril-dsp`)
    -   ✅ STFT/iSTFT module (`stft.rs`): Custom implementation using convolutions. Includes windowing and padding logic matching reference. Updated to handle various input shape formats. Basic tests pass.
    -   ✅ Window functions (`window.rs`): Implemented Hann, Hamming, Rectangular windows with correct periodic logic. Tests pass.
    -   ✅ WAV file output (`lib.rs`): Implemented `save_wav` function.

4.  **Main Executable** (`ferrocarril/src/main.rs`) and Model Implementation
    -   ✅ Basic CLI interface with infer and demo commands.
    -   ✅ G2P testing through direct Phonesis integration.
    -   ✅ **NEW** Alignment Tensor Creation: Properly implements expansion of durations into [T, sum(durations)] matrix.
    -   ✅ **NEW** Voice Embedding Handling: Correctly processes voice embedding from [510, 1, 256] to [1, 256].
    -   ✅ **NEW** Comprehensive Tests: End-to-end tests for the full pipeline.

## Areas that Need Attention

Though significant progress has been made, some areas still require attention:

1. **Variable-Length Sequence Support** (MEDIUM)
   - Implement proper pack/unpack for variable-length sequences in LSTM

2. **BERT Implementation** (MEDIUM)
   - Complete BERT implementation to match reference model

3. **Performance Optimization** (LOW)
   - Optimize matrix multiplication and tensor operations
   - Implement batched processing for improved throughput

## Updated Testing Strategy

The project now follows a strict testing strategy that ensures functional correctness, not just structural matching:

1. **Component Tests**
   - Load real weights from model files
   - Use realistic inputs
   - Verify non-zero, statistically reasonable outputs
   - Compare with Kokoro outputs when possible

2. **Integration Tests**
   - Test the full pipeline from phonemes to audio
   - Verify voice embedding handling
   - Validate alignment tensor creation

3. **End-to-End Tests**
   - Generate audio from phonemes with real weights
   - Verify audio has appropriate statistical properties
   - Compare audio from different voice embeddings

## Documentation Updates

- ✅ `FERROCARRIL_TTS_BURNDOWN.md`: Detailed burndown list of implementation issues and progress
- ✅ `FERROCARRIL_TENSOR_SHAPES.md`: Visual guide to tensor shapes throughout the TTS pipeline
- ✅ `IMPLEMENTATION_PROGRESS.md`: Overall implementation status
- ✅ `STATUS.md`: Updated with focus on functional correctness

## Code Quality Notes

-   **Structure**: Good modular structure using Rust workspaces and crates with clean dependency graph.
-   **Dependencies**: Successfully maintains zero external runtime dependencies, with the exception of Phonesis as an obligate dependency for G2P.
-   **Testing**: Comprehensive testing strategy with real weights validation.
-   **Documentation**: Code is well-documented with inline comments. Architecture docs updated regularly.
-   **Error Handling**: Components now fail fast with clear messages on dimension mismatches.

## Implementation Strategy Moving Forward

1. Focus on verifying functional correctness with real weights for all components
2. Continue improving the test suite to catch regressions
3. Complete the remaining items in the burndown list
4. Finally optimize for performance once functional correctness is ensured