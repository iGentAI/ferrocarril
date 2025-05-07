# Ferrocarril: Pure Rust TTS Implementation Design Document

## Overview

Ferrocarril is a pure Rust, dependency-free implementation of the kokoro TTS inference pipeline. This document outlines the architecture, components, and implementation strategy.

## Architecture

### Core Components

1. **Tensor Module** (`tensor.rs`)
   - Custom tensor implementation with SIMD support
   - Memory-efficient operations for CPU execution
   - Support for broadcasting and strided operations

2. **Matrix Operations** (`ops/`)
   - Matrix multiplication (optimized with SIMD)
   - Convolution operations
   - Transposition and reshaping

3. **Neural Network Layers** (`nn/`)
   - Linear layers
   - Conv1d implementation
   - LSTM (optimized for CPU)
   - AdaIN (Adaptive Instance Normalization)
   - Activation functions

4. **Signal Processing** (`dsp/`)
   - Custom STFT/iSTFT implementation
   - Sine wave generation
   - Signal manipulation utilities

5. **Model** (`model.rs`)
   - Main inference pipeline
   - Weight loading from PyTorch format
   - Component orchestration

6. **Weight Loading** (`weights.rs`)
   - Direct PyTorch file parsing
   - Memory-mapped loading for efficiency

## Implementation Strategy

### Phase 1: Core Infrastructure
- Basic tensor operations
- Memory management system
- Simple matrix multiplication
- Unit tests for core operations

### Phase 2: Neural Network Layers
- Linear layer implementation
- Conv1d with optimizations
- LSTM implementation
- Activation functions
- Normalization layers

### Phase 3: Signal Processing
- STFT/iSTFT implementation
- Audio processing utilities
- Validation against reference implementation

### Phase 4: Weight Loading
- PyTorch weight parser
- Memory-mapped weight loading

### Phase 5: Model Integration
- Component integration
- Full model assembly
- Test against reference outputs

### Phase 6: Optimization
- Profile and optimize bottlenecks
- SIMD optimizations
- Memory usage optimization
- Multithreading improvements

## Testing Strategy

### Integration Testing
- Generate audio from text inputs
- Compare audio characteristics (not binary equality)
- Use STT verification for quality assessment

### Performance Testing
- Inference speed benchmarks
- Memory usage profiling
- CPU utilization metrics

## Dependencies

This project has zero external dependencies. All functionality is implemented from scratch.