# Neural Network Component Validation Plan

**CRITICAL PRINCIPLE: All validation uses REAL Kokoro weights ONLY**
- NO synthetic weights
- NO random initialization  
- NO mock data
- ONLY the production 81.8M parameter Kokoro model

## Validation Methodology

### Phase 1: Foundation Components (PRIORITY 1)

**1.1 TextEncoder Validation**
- **Real Weights**: 24 tensors, 5.6M parameters from text_encoder component
- **Critical Tests**:
  - Load real embedding weights: `module.embedding.weight` [178, 512]
  - Test CNN blocks with real weight_g/weight_v normalization
  - Validate bidirectional LSTM with real forward/reverse weights
  - **Test Pattern**: Real phoneme input → Real CNN processing → Real LSTM output
- **Expected Output**: [Batch, 512, Time] with meaningful hidden representations
- **Failure Signals**: All-zero output, dimension mismatches, silent fallbacks

**1.2 LSTM Component Validation** 
- **Real Weights**: Bidirectional LSTM weights from multiple components
- **Critical Tests**:
  - Forward direction: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
  - Reverse direction: weight_ih_l0_reverse, weight_hh_l0_reverse, etc.
  - Test concatenation of forward + reverse outputs along feature dimension
  - **Test Pattern**: Real sequence input → Bidirectional processing → Concatenated output
- **Expected Output**: [Batch, Time, Hidden*2] with distinct forward/reverse contributions
- **Failure Signals**: Only uni-directional output, reverse weights ignored

### Phase 2: Text Understanding (PRIORITY 1) 

**2.1 BERT Implementation (CRITICAL GAP)**
- **Real Weights**: 25 tensors, 6.2M parameters from bert component  
- **Missing Components**:
  - CustomAlbert transformer (12 layers)
  - Word embeddings: [178, 128] 
  - Position embeddings: [512, 128]
  - Attention layers: Query/Key/Value matrices
  - Feed-forward networks: Intermediate projections
- **Implementation Required**: Complete CustomAlbert based on Python reference
- **Test Pattern**: Real token IDs → Real embeddings → Real transformer → Hidden states
- **Expected Output**: [Batch, Sequence, 768] contextual representations

**2.2 BERT Encoder Validation**
- **Real Weights**: 2 tensors, 393K parameters from bert_encoder component
- **Critical Tests**: 
  - Linear projection: [768] → [512] from BERT to Ferrocarril hidden size
  - Real BERT output → Real projection weights → Ferrocarril format
- **Expected Output**: [Batch, Sequence, 512] for downstream processing

### Phase 3: Prosody Prediction (PRIORITY 1)

**3.1 ProsodyPredictor Validation**
- **Real Weights**: 122 tensors, 16.1M parameters from predictor component
- **Critical Tests**:
  - Duration prediction with real duration encoder weights
  - F0 (pitch) prediction with real F0 network weights  
  - Noise prediction with real noise network weights
  - Style conditioning via real AdaIN weights
- **Test Pattern**: Real text features + Real style → Real prosody predictions
- **Expected Output**: Duration curves, F0 curves, noise curves for natural speech
- **Failure Signals**: Flat prosody, identical F0/noise outputs, style ignored

### Phase 4: Audio Generation (PRIORITY 2)

**4.1 Decoder Validation**
- **Real Weights**: 375 tensors, 53.2M parameters from decoder component
- **Critical Tests**:
  - Encoding blocks with real AdaIN conditioning weights
  - Upsampling layers with real transposed convolution weights
  - ResBlocks with real weight normalization and style conditioning
- **Test Pattern**: Real features → Real vocoder → Real audio spectrograms
- **Expected Output**: Meaningful spectrograms ready for waveform generation

**4.2 Generator Validation**
- **Real Weights**: Subset of decoder component (source module, upsampling, final projection)
- **Critical Tests**:
  - Harmonic source generation with real sine generation parameters
  - Custom STFT/iSTFT with real spectral processing
  - Final waveform generation  
- **Test Pattern**: Real spectrograms → Real source modeling → Real audio waveforms
- **Expected Output**: Audio waveforms with natural speech characteristics

## Real Weight Testing Framework

### Component Test Template

```rust
#[test]
fn test_real_[component]_weights() {
    // MANDATORY: Load real Kokoro weights
    let loader = BinaryWeightLoader::from_directory("real_kokoro_weights")?;
    
    // Create component
    let mut component = ComponentType::new(config);
    
    // MANDATORY: Load real weights
    component.load_weights_binary(&loader, "component_name", "prefix")?;
    
    // Prepare real input data (NOT synthetic)
    let real_input = create_realistic_input();
    
    // Forward pass with real weights
    let output = component.forward(&real_input);
    
    // MANDATORY: Validate functional correctness
    assert!(!output.data().iter().all(|&v| v.abs() < 1e-6),
            "Component produces zero output with real weights - functionally dead!");
            
    // Validate statistical properties
    let mean = calculate_mean(output.data());
    let variance = calculate_variance(output.data(), mean);
    assert!(variance > min_expected_variance, "Output lacks variation!");
    
    // Validate tensor shapes match Python reference exactly
    assert_eq!(output.shape(), expected_shape_from_kokoro);
}
```

### Validation Checklist (Per Component)

- [ ] **Real Weight Loading**: Component loads all expected weights without fallback
- [ ] **Tensor Shape Matching**: Exact shape correspondence with Python Kokoro
- [ ] **Functional Data Transformation**: Non-zero, statistically valid outputs
- [ ] **No Silent Fallbacks**: Assert on dimension mismatches, fail fast
- [ ] **Python Reference Alignment**: Behavior matches reference implementation

## Component Dependency Chain

```
Text Input
    ↓
┌──────────────┐
│ G2P (Phonesis) │ ✅ PRODUCTION READY
└───────┬──────┘
        ↓ Phonemes
┌──────────────┐
│ TextEncoder   │ ⚠️ NEEDS REAL WEIGHT TESTING
└───────┬──────┘
        ↓ Text Features [B, 512, T]
┌──────────────┐
│ BERT/Albert   │ ❌ MISSING IMPLEMENTATION
└───────┬──────┘
        ↓ Contextual Hidden States [B, T, 768]
┌──────────────┐
│ BERT Encoder  │ ⚠️ NEEDS REAL WEIGHT TESTING  
└───────┬──────┘
        ↓ TTS Hidden States [B, T, 512]
┌──────────────┐
│ProsodyPredctr │ ⚠️ NEEDS REAL WEIGHT TESTING
└───────┬──────┘
        ↓ Duration, F0, Noise
┌──────────────┐
│   Decoder     │ ⚠️ NEEDS REAL WEIGHT TESTING
└───────┬──────┘
        ↓ Audio Spectrograms
┌──────────────┐
│  Generator    │ ⚠️ NEEDS REAL WEIGHT TESTING
└───────┬──────┘
        ↓ 
   Final Audio
```

## Validation Schedule

### Week 1: Foundation Layer Real Weight Validation
- **Day 1-2**: TextEncoder with real weights 
- **Day 3-4**: LSTM component validation with real bidirectional weights
- **Day 5**: Integration testing of TextEncoder + real weights

### Week 2: Critical Gap Resolution  
- **Day 1-5**: BERT/CustomAlbert implementation using real transformer weights

### Week 3: Prosody and Alignment
- **Day 1-3**: ProsodyPredictor with real duration/F0/noise weights
- **Day 4-5**: Alignment matrix generation with real duration predictions

### Week 4: Audio Generation
- **Day 1-3**: Decoder validation with real vocoder weights  
- **Day 4-5**: Generator and final audio output validation

### Week 5: End-to-End Pipeline
- **Day 1-3**: Complete inference pipeline with real weights
- **Day 4-5**: Audio quality validation against Python reference

## Success Criteria

**Component-Level:**
- ✅ Loads all real weights without errors
- ✅ Produces non-zero, statistically reasonable outputs
- ✅ Matches Python tensor shapes exactly
- ✅ No silent fallbacks or dimension fudging

**System-Level:**
- ✅ Complete Text → Audio pipeline using only real weights
- ✅ Audio output quality comparable to Python Kokoro
- ✅ No component produces functionally dead outputs
- ✅ All tensor transformations match reference implementation

The validation succeeds only when we can generate high-quality speech audio using the real trained Kokoro model weights in the Rust implementation.