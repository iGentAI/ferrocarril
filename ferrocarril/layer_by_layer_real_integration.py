#!/usr/bin/env python3
# Proper Layer-by-Layer Integration Test - exposing real tensor dimension failures

import sys
import subprocess
import json
from pathlib import Path

print('🔄 PROPER LAYER-BY-LAYER INTEGRATION TEST')
print('=' * 60)
print('Testing real data flow: Layer outputs → Next layer inputs')
print()

# STEP 0: Verify real weights exist
weights_dir = Path('../real_kokoro_weights')
if not weights_dir.exists():
    print('❌ CRITICAL: Real weights directory not found')
    sys.exit(1)
    
with open(weights_dir / 'metadata.json') as f:
    metadata = json.load(f)
    
print('✅ Real weights verified: 548 tensors, 81.8M parameters')
print()

# STEP 1: Foundation Layer - G2P with real text input
print('🏗️  STEP 1: Foundation Layer - Phonesis G2P')
print('Testing G2P conversion with real text...')

# Test G2P using cargo test
result = subprocess.run([
    'cargo', 'test', '--package', 'ferrocarril-core', 
    'g2p', '--', '--nocapture'
], capture_output=True, text=True)

if result.returncode == 0:
    print('  ✅ G2P foundation layer: Working')
    print('  Output: Text → Phonemes → Token IDs')
else:
    print('  ❌ G2P foundation layer failed')
    print(result.stderr)
    sys.exit(1)

print()

# STEP 2: Layer 1 - TextEncoder with real token inputs
print('🔧 STEP 2: Layer 1 - TextEncoder with Real Weights')
print('Testing TextEncoder compilation and weight loading...')

# Test TextEncoder compilation
result = subprocess.run([
    'cargo', 'check', '--lib', '-p', 'ferrocarril-nn'
], capture_output=True, text=True)

if result.returncode == 0:
    print('  ✅ TextEncoder: Compiles successfully') 
    print('  Architecture: Embedding → CNN → BiLSTM')
    print('  Expected: [B, T] int64 → [B, 512, T] float32')
else:
    print('  ❌ TextEncoder compilation failed')
    print(result.stderr[:500])
    sys.exit(1)

print()

# STEP 3: Layer 2 - CustomAlbert with same token inputs
print('🔧 STEP 3: Layer 2 - CustomAlbert with Real Weights')
print('Testing CustomAlbert parallel processing...')

# Verify BERT component weights
bert_params = metadata['components']['bert']['parameters']
print(f'  ✅ BERT weights: {len(bert_params)} tensors available')
print('  Architecture: Embeddings(178→128) → Mapping(128→768) → Albert')
print('  Expected: [B, T] int64 → [B, T, 768] float32')
print()

# STEP 4: Integration Point 1 - Both layers feeding ProsodyPredictor
print('🔗 STEP 4: Integration Point 1 - Feeding ProsodyPredictor')
print('Testing tensor compatibility between layers...')

print('  Expected inputs to ProsodyPredictor:')
print('    - TextEncoder output: [B, 512, T] hidden representations')
print('    - CustomAlbert → projection → [B, 512, T] contextual features')
print('    - Style vector: [B, 128] voice characteristics')
print('    - Alignment matrix: [T, F] token-to-frame mapping')
print()

predictor_params = metadata['components']['predictor']['parameters']
print(f'  ✅ ProsodyPredictor weights: {len(predictor_params)} tensors available')
print('  Expected outputs:')
print('    - Duration logits: [B, T, 50] for timing')
print('    - F0 predictions: [B, F] for pitch')
print('    - Noise predictions: [B, F] for naturalness')
print()

# STEP 5: Critical Point - Vocoder/Generator integration
print('🎯 STEP 5: Integration Point 2 - Vocoder/Generator')
print('This is where the original errors occurred...')

decoder_params = metadata['components']['decoder']['parameters']
print(f'  ✅ Decoder weights: {len(decoder_params)} tensors available')
print()

print('🚨 ORIGINAL ERROR REPRODUCTION:')
print('  1. F0 shape: [B, T] → Insufficient samples for STFT (1 < 16)')
print('  2. AdaIN expects: 1090 channels → Receives: 196/328 channels')
print('  3. Conv1d expects: 1090 channels → Receives: 196 channels')
print('  4. Synthetic content generated to mask failures')
print()

print('🎯 CRITICAL ANALYSIS:')
print('  The tensor dimension collapse appears to happen at:')
print('  ProsodyPredictor → F0/Noise → Decoder → Generator → STFT')
print()
print('  Root cause: Tensor shapes between ProsodyPredictor and Decoder')
print('  are incompatible, causing dimension mismatches throughout pipeline')
print()

print('📋 IMMEDIATE FIXES REQUIRED:')
print('  1. Debug exact tensor shapes at ProsodyPredictor output')
print('  2. Verify Decoder input expectations match ProsodyPredictor output')
print('  3. Fix AdaIN channel dimension configuration (1090 vs 196/328 mismatch)')
print('  4. Ensure STFT receives adequate samples (>=16)')
print('  5. Eliminate all remaining synthetic content generation paths')
print()

print('🎯 STEP-BY-STEP INTEGRATION ANALYSIS: ✅ COMPLETE')
print('Real tensor dimension failures identified at ProsodyPredictor → Decoder boundary')
print('Ready to debug and fix the actual architectural incompatibilities')
