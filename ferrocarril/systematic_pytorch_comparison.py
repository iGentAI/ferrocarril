#!/usr/bin/env python3
"""
Systematic PyTorch vs Rust Comparison
Find exact divergence points in Ferrocarril TTS pipeline
"""

import torch
import numpy as np
import sys
import json
sys.path.append('kokoro')

from kokoro import KModel, KPipeline
from kokoro.model import KModel

def analysis_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def compare_tensors(pytorch_tensor, rust_output, name, threshold=1e-6):
    """Compare PyTorch tensor with expected Rust output characteristics"""
    print(f"\n🔍 {name} Comparison:")
    
    if pytorch_tensor is not None:
        pt_shape = list(pytorch_tensor.shape)
        pt_mean = pytorch_tensor.mean().item()
        pt_std = pytorch_tensor.std().item()
        pt_min = pytorch_tensor.min().item()
        pt_max = pytorch_tensor.max().item()
        pt_nonzero = (pytorch_tensor.abs() > threshold).sum().item()
        
        print(f"   PyTorch - Shape: {pt_shape}, Mean: {pt_mean:.6f}, Std: {pt_std:.6f}")
        print(f"             Range: [{pt_min:.6f}, {pt_max:.6f}], NonZero: {pt_nonzero}")
        
        if rust_output:
            print(f"   Expected Rust - {rust_output}")
            
            # Check for critical divergence patterns
            if abs(pt_mean) > 10.0:
                print(f"   ❌ WARNING: Mean value {pt_mean:.6f} is unusually large!")
            if pt_std < 1e-8:
                print(f"   ❌ WARNING: Std dev {pt_std:.8f} suggests all zeros or constant!")
            if pt_min == pt_max:
                print(f"   ❌ CRITICAL: Min == Max suggests dead/constant output!")
            if pt_nonzero == 0:
                print(f"   ❌ CRITICAL: All zero output detected!")
            
        return {
            'shape': pt_shape,
            'mean': pt_mean,
            'std': pt_std,
            'min': pt_min,
            'max': pt_max,
            'nonzero_count': pt_nonzero
        }
    
    return None

def test_pytorch_internals():
    """Test PyTorch internals step by step to understand exact processing"""
    analysis_header("PYTORCH INTERNAL STEP-BY-STEP ANALYSIS")
    
    # Initialize PyTorch system
    print("Initializing PyTorch KModel and KPipeline...")
    model = KModel('hexgrad/Kokoro-82M').eval()
    pipeline = KPipeline(lang_code='a', model=model)
    
    text = "Hello world."
    print(f"Input text: '{text}'")
    
    # Step 1: G2P Processing
    print("\n--- STEP 1: G2P Processing ---")
    _, tokens = pipeline.g2p(text)
    phonemes = pipeline.tokens_to_ps(tokens)
    print(f"Phonemes: '{phonemes}'")
    print(f"Phoneme length: {len(phonemes)}")
    
    # Step 2: Token processing
    print("\n--- STEP 2: Token Processing ---")
    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(model.device)
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Token sequence length: {input_ids.shape[1]}")
    
    # Load voice
    pack = pipeline.load_voice('af_heart').to(model.device)
    ref_s = pack[len(phonemes)-1]
    print(f"Voice embedding shape: {ref_s.shape}")
    
    # Step 3: Model internals
    print("\n--- STEP 3: Model Forward Pass Internals ---")
    
    with torch.no_grad():
        # Text processing
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], 
                                  device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(model.device)
        
        print(f"Input lengths: {input_lengths}")
        print(f"Text mask shape: {text_mask.shape}")
        
        # BERT processing
        print("\n🧠 BERT Processing:")
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        compare_tensors(bert_dur, "Expected: [1, seq_len, 768]", "BERT Output")
        
        # BERT encoder projection
        print("\n🔄 BERT Encoder Projection:")
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        compare_tensors(d_en, "Expected: [1, 512, seq_len]", "BERT Encoded")
        
        # Style processing  
        s = ref_s[:, 128:]
        compare_tensors(s, "Expected: [1, 128]", "Style Embedding")
        
        # Duration encoder
        print("\n⏱️ Duration Encoder:")
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        compare_tensors(d, "Expected: [1, 512, seq_len]", "Duration Encoder Output")
        
        # Duration LSTM
        print("\n🔄 Duration LSTM:")
        x, _ = model.predictor.lstm(d)
        compare_tensors(x, "Expected: [1, seq_len, 512]", "Duration LSTM Output")
        
        # Duration prediction
        print("\n📏 Duration Prediction:")
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / 1.0  # speed=1
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        compare_tensors(duration, "Expected: [1, seq_len]", "Duration Logits")
        compare_tensors(pred_dur, "Expected: [seq_len] integers", "Predicted Durations")
        
        print(f"Predicted durations: {pred_dur.tolist()}")
        total_duration_frames = pred_dur.sum().item()
        print(f"Total duration frames: {total_duration_frames}")
        
        # Alignment matrix
        print("\n🎯 Alignment Matrix:")
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=model.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(model.device)
        
        print(f"Alignment matrix shape: {pred_aln_trg.shape}")
        print(f"Audio frames from alignment: {pred_aln_trg.shape[2]}")
        
        # Energy pooling
        print("\n⚡ Energy Pooling:")
        en = d.transpose(-1, -2) @ pred_aln_trg
        compare_tensors(en, "Expected: [1, 512, total_frames]", "Energy Pooled")
        
        # F0 and noise prediction
        print("\n🎵 F0 and Noise Prediction:")
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        compare_tensors(F0_pred, f"Expected: [1, upsampled_frames]", "F0 Prediction")
        compare_tensors(N_pred, f"Expected: [1, upsampled_frames]", "Noise Prediction")
        
        print(f"F0 range: [{F0_pred.min().item():.3f}, {F0_pred.max().item():.3f}]")
        print(f"Noise range: [{N_pred.min().item():.3f}, {N_pred.max().item():.3f}]")
        
        # Text encoder
        print("\n📝 Text Encoder:")
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        compare_tensors(t_en, "Expected: [1, 512, seq_len]", "Text Encoder Output")
        
        # ASR alignment
        print("\n🎤 ASR Alignment:")
        asr = t_en @ pred_aln_trg
        compare_tensors(asr, f"Expected: [1, 512, {pred_aln_trg.shape[2]}]", "ASR Aligned")
        
        # Final decoder
        print("\n🔊 Decoder Processing:")
        print(f"Decoder inputs:")
        print(f"  ASR: {asr.shape}")
        print(f"  F0: {F0_pred.shape}")
        print(f"  Noise: {N_pred.shape}")
        print(f"  Reference: {ref_s[:, :128].shape}")
        
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        compare_tensors(audio, f"Expected: [audio_samples] normalized float", "Final Audio")
        
        # CRITICAL ANALYSIS
        analysis_header("CRITICAL DIVERGENCE ANALYSIS")
        print(f"Final PyTorch audio characteristics:")
        print(f"  Length: {len(audio)} samples ({len(audio)/24000:.3f}s)")
        print(f"  Mean: {audio.mean().item():.8f}")
        print(f"  Std: {audio.std().item():.6f}")
        print(f"  Range: [{audio.min().item():.6f}, {audio.max().item():.6f}]")
        print(f"  NonZero: {(audio.abs() > 1e-6).sum().item()}/{len(audio)}")
        
        # Duration analysis
        print(f"\nDuration chain analysis:")
        print(f"  Duration predictions: {pred_dur.tolist()}")
        print(f"  Total frames: {total_duration_frames}")
        print(f"  Audio samples: {len(audio)}")
        print(f"  Expansion ratio: {len(audio) / total_duration_frames:.2f}")
        
        # Check for processing issues
        if audio.std().item() < 1e-6:
            print("❌ CRITICAL: Audio has no variation - decoder is dead!")
        if audio.mean().item() > 0.1:
            print("❌ CRITICAL: Audio mean is too high - normalization issue!")
        if len(audio) < 30000:
            print("❌ WARNING: Audio is shorter than expected for 'Hello world.'")
            
    return {
        'audio_length': len(audio),
        'duration_frames': total_duration_frames,
        'audio_stats': {
            'mean': audio.mean().item(),
            'std': audio.std().item(),
            'min': audio.min().item(),
            'max': audio.max().item()
        }
    }

if __name__ == "__main__":
    print("🔬 SYSTEMATIC PYTORCH vs RUST DIVERGENCE ANALYSIS")
    print("   Identifying exact numerical processing differences...")
    
    results = test_pytorch_internals()
    
    analysis_header("COMPARISON SUMMARY")
    print(f"PyTorch Reference Results:")
    print(f"  Audio: {results['audio_length']} samples")
    print(f"  Duration frames: {results['duration_frames']}")
    print(f"  Audio stats: {results['audio_stats']}")
    
    print(f"\nExpected Rust Results (from terminal analysis):")
    print(f"  Audio: 21000 samples ❌ (44% missing)")
    print(f"  Mean: 35.34 ❌ (wrong scaling)")
    print(f"  Range: [-158, 357] ❌ (integer vs float)")
    
    print(f"\n🎯 KEY DIVERGENCE POINTS TO INVESTIGATE:")
    print(f"1. Duration prediction: Does Rust predict same duration values?")
    print(f"2. Alignment matrix: Does Rust create same frame count?") 
    print(f"3. Audio generation: Does Rust decoder produce same sample count?")
    print(f"4. Normalization: Does Rust maintain float precision throughout?")