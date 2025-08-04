# Comprehensive Layer-by-Layer Validation Plan

## Foundation Status ✅

**Compilation Status**: ferrocarril-core compiles successfully (exit code 0)  
**Weight System**: 688 parameters converted from real Kokoro-82M model  
**G2P Integration**: 90.3% symbol compatibility between Phonesis and Kokoro  
**Binary Loading**: BinaryWeightLoader working with memmap2 integration  

## 9-Layer TTS Pipeline Validation Strategy

### **Layer 1: G2P Conversion** ✅ READY
- **Input**: Text string ("Hello world")
- **Process**: Text → Phonesis IPA → Kokoro tokens
- **Output**: Token sequence [1, seq_len]
- **Validation**: Verify token mapping accuracy (90.3% direct compatibility confirmed)
- **Test**: Compare Phonesis output against expected token sequences

### **Layer 2: CustomBERT** 🔧 NEEDS TESTING
- **Input**: Tokens [1, seq_len]
- **Process**: Token contexts → BERT embeddings
- **Output**: Embeddings [1, seq_len, 768]
- **Validation**: Load real BERT weights (25 parameters), test forward pass
- **Test**: Compare output statistics (mean, std) against PyTorch reference

### **Layer 3: BERT→Hidden Projection** 🔧 NEEDS TESTING
- **Input**: BERT embeddings [1, seq_len, 768]
- **Process**: Linear projection + transpose
- **Output**: Hidden features [1, 512, seq_len]
- **Validation**: Test projection matrix (768→512) with real weights
- **Test**: Verify transpose operation matches PyTorch pattern

### **Layer 4: Duration Prediction** 🔧 NEEDS TESTING
- **Input**: Hidden features [1, 512, seq_len]
- **Process**: ProsodyPredictor duration estimation
- **Output**: Duration logits [1, seq_len, 50], predicted durations
- **Validation**: Test ProsodyLSTM (146 parameters) with real weights
- **Test**: Compare duration patterns against PyTorch reference

### **Layer 5: Energy Pooling** 🔧 NEEDS TESTING
- **Input**: Features + alignment matrix
- **Process**: Apply duration-based alignment
- **Output**: Frame-aligned features [1, hidden_dim+style, total_frames]
- **Validation**: Test alignment matrix construction and application
- **Test**: Verify energy pooling matches PyTorch tensor operations

### **Layer 6: F0/Noise Prediction** 🔧 NEEDS TESTING
- **Input**: Aligned features [1, 640, frames]
- **Process**: Prosody feature extraction
- **Output**: F0 curve [1, frames], Noise [1, frames]
- **Validation**: Test AdaIN blocks and prosody prediction
- **Test**: Compare F0 statistics (mean ~100Hz, reasonable variation)

### **Layer 7: TextEncoder** 🔧 NEEDS TESTING
- **Input**: Original tokens [1, seq_len]
- **Process**: TextEncoderLSTM + Conv1d blocks
- **Output**: Phoneme features [1, 512, seq_len]
- **Validation**: Test specialized TextEncoderLSTM (512→512)
- **Test**: Compare against PyTorch TextEncoder output

### **Layer 8: ASR Alignment** 🔧 NEEDS TESTING
- **Input**: TextEncoder features + alignment matrix
- **Process**: Apply same alignment to TextEncoder output
- **Output**: Frame-aligned ASR [1, 512, total_frames]
- **Validation**: Test matrix multiplication consistency
- **Test**: Verify aligned features match PyTorch computation

### **Layer 9: Decoder (Audio Generation)** 🔧 NEEDS TESTING
- **Input**: ASR features + F0 + Noise + Voice embedding
- **Process**: Vocoder neural processing
- **Output**: Audio waveform [1, audio_samples]
- **Validation**: Test massive Decoder (491 parameters, 65% of model)
- **Test**: Verify audio generation produces realistic waveforms

## Validation Methodology

### **Reference Extraction**
1. Load PyTorch Kokoro with real weights
2. Process "Hello world" through each layer
3. Extract tensor shapes, statistics (mean, std, min, max)
4. Save reference data for Rust comparison

### **Rust Testing**
1. Load same real weights using BinaryWeightLoader
2. Process same input through each specialized implementation
3. Compare outputs against PyTorch reference
4. Validate numerical accuracy within tolerance (1e-4 to 1e-2)

### **Success Criteria**
- **Shape Compatibility**: All layers produce expected tensor shapes
- **Numerical Accuracy**: Mean/std within tolerance of PyTorch reference  
- **Weight Loading**: All 688 parameters load correctly with proper precision
- **Audio Quality**: Generated audio has realistic characteristics

## Next Steps for Implementation

1. **Immediate**: Test basic weight loading in working ferrocarril-core
2. **Short-term**: Implement Layer 2 (CustomBERT) validation with real weights
3. **Medium-term**: Complete all 9 layers with PyTorch reference comparison
4. **Long-term**: End-to-end audio generation quality validation

**Status**: Foundation ready, comprehensive testing framework prepared
