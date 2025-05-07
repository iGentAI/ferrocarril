# Ferrocarril TTS Tensor Shape Reference Guide

This document provides a visual guide to the tensor shapes throughout the TTS pipeline, showing the expected shapes in Kokoro vs. the current Ferrocarril implementation.

## Voice Embedding Shapes

```
┌─────────────────┐             ┌────────────────────┐
│ Voice File      │             │  Kokoro Usage      │
│ [510, 1, 256]   │─────────────►  [1, 256]          │
└─────────────────┘             └─┬──────────────────┘
                                  │
                                  │  Split at dimension 128
                                  ▼
┌─────────────────────┐        ┌─────────────────────┐
│ Reference Part      │        │ Style Part          │
│ ref_s[:, :128]      │        │ ref_s[:, 128:]      │
│ [1, 128]            │        │ [1, 128]            │
└─────────────────────┘        └─────────────────────┘
```

**Current Issue:** The voice is incorrectly flattened to shape `[1, 130560]` instead of `[1, 256]` and isn't properly split.

## LSTM Bidirectional Output

```
┌─────────────────────┐        ┌─────────────────────┐
│ Kokoro LSTM         │        │ Expected Output     │
│ bidirectional=True  │───────►│ [batch, seq, C]     │
└─────────────────────┘        └─────────────────────┘
                                │
                                │  Concatenated outputs
                                ▼
┌─────────────────────┐        ┌─────────────────────┐
│ Forward Direction   │        │ Backward Direction  │
│ [batch, seq, C/2]   │        │ [batch, seq, C/2]   │
└─────────────────────┘        └─────────────────────┘
```

**Current Issue:** The Rust LSTM claims to be bidirectional but only processes in one direction, producing output with half the expected channels.

## ProsodyPredictor Energy Pooling

```
┌───────────────────────┐      ┌─────────────────────┐
│ Kokoro                │      │ Energy Tensor       │
│ d_enc: [B, T, H+S]    │─────►│ en: [B, H+S, F]     │
└───────────────────────┘      └─────────────────────┘
    d_enc includes style
    
┌───────────────────────┐      ┌─────────────────────┐
│ Ferrocarril           │      │ Energy Tensor       │
│ d_enc: [B, T, H]      │─────►│ en: [B, H, F]       │
└───────────────────────┘      └─────────────────────┘
    style dimension missing
```

**Current Issue:** Ferrocarril's energy tensor is missing the style dimension, affecting prosody conditioning.

## Alignment Tensor Shape

```
┌───────────────────────┐      ┌───────────────────────┐
│ Kokoro                │      │                       │
│ Durations:            │─────►│ Alignment: [T, F]     │
│ [batch, seq_len]      │      │ where F = sum(durations)│
└───────────────────────┘      └───────────────────────┘
    
┌───────────────────────┐      ┌───────────────────────┐
│ Ferrocarril           │      │                       │
│ Tests using:          │─────►│ Alignment: [T, T]     │
│ [batch, seq_len]      │      │ (identity matrix)     │
└───────────────────────┘      └───────────────────────┘
```

**Current Issue:** Ferrocarril's alignment tensor doesn't expand according to durations, breaking the text-to-audio mapping.

## BERT Feed-Forward Network

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Kokoro FFN      │      │ Intermediate    │      │ Output          │
│ Input: [B,T,768]│─────►│ [B,T,3072]      │─────►│ [B,T,768]       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
    Two linear layers: hidden→intermediate→hidden
    
┌─────────────────┐      ┌─────────────────┐
│ Ferrocarril FFN │      │ Output          │
│ Input: [B,T,768]│─────►│ [B,T,3072]      │
└─────────────────┘      └─────────────────┘
    Missing second projection back to hidden_size
```

**Current Issue:** Ferrocarril's FFN is missing the second linear projection, producing output with the wrong dimensionality.

## AdainResBlk1d Upsampling

```
Kokoro:
┌─────────────┐  Upsample   ┌─────────────┐   Conv/Norm    ┌─────────────┐
│ Input       │────────────►│ Upsampled   │────────────────►│ Output      │
│ [B,C,T]     │             │ [B,C,2T]    │                 │ [B,2C,2T]   │
└─────────────┘             └─────────────┘                 └─────────────┘
                                   │
                                   │ Also upsamples
                                   ▼
                            ┌────────────────┐   Add   ┌─────────────┐
                            │ Upsampled Res  │─────────► Final Out   │
                            │ [B,2C,2T]      │         │ [B,2C,2T]   │
                            └────────────────┘         └─────────────┘

Ferrocarril:
┌─────────────┐  Upsample   ┌─────────────┐   Conv/Norm    ┌─────────────┐
│ Input       │────────────►│ Upsampled   │────────────────►│ Output      │
│ [B,C,T]     │             │ [B,C,2T]    │                 │ [B,2C,2T]   │
└─────────────┘             └─────────────┘                 └─────────────┘
                                                                   │
                                                                   │
┌─────────────┐  Upsample   ┌─────────────┐                       │
│ Input       │────────────►│ Upsampled   │                       │
│ [B,C,T]     │             │ [B,C,2T]    │                       │
└─────────────┘             └─────────────┘                       │
                                   │                              │
                                   │                      Channel mismatch
                                   │                              │
                                   ▼                              ▼
                            ┌────────────────┐   Add   ┌─────────────┐
                            │ Res path       │─────────► Error!      │
                            │ [B,C,2T]       │         │ [B,2C,2T]   │
                            └────────────────┘         └─────────────┘
```

**Current Issue:** Channel dimension mismatch in upsampling blocks' residual connections.

## Decoder Shape Issues

```
Kokoro:
┌─────────────┐      ┌─────────────┐
│ Reflection  │──────► Input padded │
│ Left only   │      │ on left only │
└─────────────┘      └─────────────┘
                         
Ferrocarril:
┌─────────────┐      ┌─────────────┐
│ Padding     │──────► Input padded │
│ Right side  │      │ on right side│
└─────────────┘      └─────────────┘
```

**Current Issue:** Reflection padding is applied on the wrong side, creating time dimension mismatches.

## Core Shape Mismatch Patterns

1. **Half Dimension**: LSTM producing [B, T, C/2] instead of [B, T, C]
2. **Missing Style**: Energy tensor [B, H, S] vs expected [B, H+S, S] 
3. **Dimension Swaps**: [B, C, T] ↔ [B, T, C] occurring at wrong points
4. **Channel Mismatches**: In residual connections after upsampling
5. **Time Mismatches**: Due to incorrect padding direction

## Shape Checking Checklist

For each tensor operation in the codebase:

1. Does it explicitly verify input tensor shapes match expectations?
2. Does it handle broadcasting correctly for each dimension?
3. Will it fail immediately on mismatch rather than attempting a workaround?
4. Does the resulting shape exactly match what would happen in PyTorch?

Replace any code matching this pattern:

```rust
if dimensions_dont_match {
    println!("Warning: Dimensions don't match, working around...");
    // Workaround code
}
```

With this pattern:

```rust
assert!(dimensions_match, 
    "FATAL ERROR: Dimensions must match. Expected {:?}, got {:?}", 
    expected_shape, actual_shape);
```

## Shape Issues per Model Component

| Component | Expected Shape | Current Shape | Issue |
|-----------|----------------|--------------|-------|
| Voice Embeddings | [1, 256] | [1, 130560] | Flattened instead of selecting position |
| LSTM Output | [B, T, C] | [B, T, C/2] | Missing reverse direction |
| ProsodyPredictor (en) | [B, H+S, F] | [B, H, F] | Missing style dimension |
| ProsodyPredictor (F0) | [B, F] | [B, F] | Intermediate dimensions swapped |
| BERT FFN | [B, T, 768] | [B, T, 3072] | Missing second projection |
| AdainResBlk1d | [B, 2C, 2T] | [B, C, 2T] + [B, 2C, 2T] | Channel count mismatch |
| Decoder padding | Left-padded | Right-padded | Wrong padding direction |