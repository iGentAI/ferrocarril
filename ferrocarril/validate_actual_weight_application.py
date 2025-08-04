#!/usr/bin/env python3
"""
Systematic Weight Application Validation
Verify that weights are actually affecting neural computation, not just loaded structurally
"""

import sys
sys.path.append('kokoro')
import torch
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
import json

def validate_pytorch_reference():
    """Get PyTorch reference values for comparison"""
    print("🔬 SYSTEMATIC WEIGHT APPLICATION VALIDATION")
    print("=" * 60)
    
    # Initialize PyTorch system
    model = KModel('hexgrad/Kokoro-82M').eval()
    pipeline = KPipeline(lang_code='a', model=model)
    
    text = "Hello world."
    _, tokens = pipeline.g2p(text)
    phonemes = pipeline.tokens_to_ps(tokens)
    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(model.device)
    
    pack = pipeline.load_voice('af_heart').to(model.device)
    ref_s = pack[len(phonemes)-1]
    
    print(f"Input tokens: {input_ids.shape} -> {input_ids.tolist()}")
    print(f"Voice embedding: {ref_s.shape}")
    print()
    
    # Run layer-by-layer validation
    with torch.no_grad():
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(model.device)
        
        validation_results = {}
        
        # Layer 1: BERT Processing
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        validation_results['bert'] = {
            'shape': list(bert_dur.shape),
            'mean': bert_dur.mean().item(),
            'std': bert_dur.std().item(),
            'min': bert_dur.min().item(),
            'max': bert_dur.max().item(),
            'nonzero_count': (bert_dur.abs() > 1e-8).sum().item(),
            'total_elements': bert_dur.numel()
        }
        
        # Layer 2: BERT Encoder Projection
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        validation_results['bert_encoder'] = {
            'shape': list(d_en.shape),
            'mean': d_en.mean().item(),
            'std': d_en.std().item(),
            'min': d_en.min().item(),
            'max': d_en.max().item(),
            'nonzero_count': (d_en.abs() > 1e-8).sum().item(),
            'total_elements': d_en.numel()
        }
        
        # Layer 3: Style Processing
        s = ref_s[:, 128:]
        validation_results['style'] = {
            'shape': list(s.shape),
            'mean': s.mean().item(),
            'std': s.std().item(),
            'min': s.min().item(),
            'max': s.max().item(),
            'nonzero_count': (s.abs() > 1e-8).sum().item(),
            'total_elements': s.numel()
        }
        
        # Layer 4: Duration Encoder
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        validation_results['duration_encoder'] = {
            'shape': list(d.shape),
            'mean': d.mean().item(),
            'std': d.std().item(),
            'min': d.min().item(),
            'max': d.max().item(),
            'nonzero_count': (d.abs() > 1e-8).sum().item(),
            'total_elements': d.numel()
        }
        
        # Layer 5: Duration LSTM
        x, _ = model.predictor.lstm(d)
        validation_results['duration_lstm'] = {
            'shape': list(x.shape),
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item(),
            'nonzero_count': (x.abs() > 1e-8).sum().item(),
            'total_elements': x.numel()
        }
        
        # Layer 6: Duration Prediction & Alignment
        duration = model.predictor.duration_proj(x)
        duration_sigmoid = torch.sigmoid(duration).sum(axis=-1) / 1.0
        pred_dur = torch.round(duration_sigmoid).clamp(min=1).long().squeeze()
        
        validation_results['duration_prediction'] = {
            'shape': list(duration.shape),
            'mean': duration.mean().item(),
            'std': duration.std().item(),
            'durations': pred_dur.tolist(),
            'total_frames': pred_dur.sum().item()
        }
        
        # Create alignment for energy pooling
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=model.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(model.device)
        
        # Layer 7: Energy Pooling
        en = d.transpose(-1, -2) @ pred_aln_trg
        validation_results['energy_pooling'] = {
            'shape': list(en.shape),
            'mean': en.mean().item(),
            'std': en.std().item(),
            'min': en.min().item(),
            'max': en.max().item(),
            'nonzero_count': (en.abs() > 1e-8).sum().item(),
            'total_elements': en.numel()
        }
        
        # Layer 8: F0 and Noise Prediction - CRITICAL FOR SPEECH
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        validation_results['f0_prediction'] = {
            'shape': list(F0_pred.shape),
            'mean': F0_pred.mean().item(),
            'std': F0_pred.std().item(),
            'min': F0_pred.min().item(),
            'max': F0_pred.max().item(),
            'has_variation': F0_pred.std().item() > 1.0,
            'speech_like': 50 <= F0_pred.mean().item() <= 350
        }
        
        validation_results['noise_prediction'] = {
            'shape': list(N_pred.shape),
            'mean': N_pred.mean().item(),
            'std': N_pred.std().item(),
            'min': N_pred.min().item(),
            'max': N_pred.max().item(),
            'has_variation': N_pred.std().item() > 0.1
        }
        
        # Layer 9: Text Encoder
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        validation_results['text_encoder'] = {
            'shape': list(t_en.shape),
            'mean': t_en.mean().item(),
            'std': t_en.std().item(),
            'min': t_en.min().item(),
            'max': t_en.max().item(),
            'nonzero_count': (t_en.abs() > 1e-8).sum().item(),
            'total_elements': t_en.numel()
        }
        
        # Layer 10: ASR Alignment
        asr = t_en @ pred_aln_trg
        validation_results['asr_aligned'] = {
            'shape': list(asr.shape),
            'mean': asr.mean().item(),
            'std': asr.std().item(),
            'min': asr.min().item(),
            'max': asr.max().item(),
            'nonzero_count': (asr.abs() > 1e-8).sum().item(),
            'total_elements': asr.numel()
        }
    
    return validation_results

def analyze_weight_effectiveness(results):
    """Analyze if PyTorch components show meaningful neural activity"""
    print("\n📊 PYTORCH NEURAL ACTIVITY ANALYSIS")
    print("=" * 40)
    
    for layer_name, stats in results.items():
        print(f"\n🔍 {layer_name.upper()}:")
        print(f"   Shape: {stats['shape']}")
        print(f"   Mean: {stats['mean']:.8f}")
        print(f"   Std: {stats['std']:.6f}")
        
        if 'nonzero_count' in stats:
            nonzero_pct = (stats['nonzero_count'] / stats['total_elements']) * 100
            print(f"   NonZero: {stats['nonzero_count']}/{stats['total_elements']} ({nonzero_pct:.1f}%)")
        
        # Check for meaningful neural activity
        if stats['std'] > 0.01:
            print("   ✅ Shows meaningful variation (likely functional)")
        else:
            print(f"   ❌ Low variation (std={stats['std']:.6f}) - possibly dead")
            
        if abs(stats['mean']) < 1.0:
            print("   ✅ Reasonable mean value")
        else:
            print(f"   ❌ Unusual mean: {stats['mean']:.6f}")
    
    # Special checks for critical speech components
    if 'f0_prediction' in results:
        f0 = results['f0_prediction']
        print(f"\n🎵 F0 SPEECH ANALYSIS:")
        print(f"   Mean F0: {f0['mean']:.1f} Hz")
        print(f"   F0 variation: {f0['std']:.3f}")
        
        if f0['has_variation'] and f0['speech_like']:
            print("   ✅ F0 shows realistic speech characteristics")
        else:
            print("   ❌ F0 lacks speech-like patterns")

if __name__ == "__main__":
    print("🧪 VALIDATING ACTUAL WEIGHT APPLICATION vs STRUCTURAL LOADING")
    
    try:
        # Get PyTorch reference validation
        pytorch_results = validate_pytorch_reference()
        
        # Analyze effectiveness
        analyze_weight_effectiveness(pytorch_results)
        
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"PyTorch reference shows proper neural activity patterns")
        print(f"Our Rust system should match these statistics if weights work correctly")
        print(f"Any significant divergence indicates weight application issues")
        
        # Save results for comparison
        with open('pytorch_reference_validation.json', 'w') as f:
            json.dump(pytorch_results, f, indent=2)
        print(f"\n💾 Reference validation saved to: pytorch_reference_validation.json")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)