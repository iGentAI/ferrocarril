#!/usr/bin/env python3
"""
PyTorch Layer Statistics Debug Script
Outputs statistical fingerprints from each layer for comparison with Rust implementation.
"""

import sys
sys.path.append('kokoro')

import torch
import numpy as np
from kokoro.model import KModel

def print_tensor_stats(name, tensor, indent=""):
    """Print statistical fingerprints of a tensor."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        
        # Handle different data types
        if tensor.dtype in [torch.long, torch.int64, torch.int32]:
            unique_vals = len(torch.unique(tensor))
            min_val = int(torch.min(tensor))
            max_val = int(torch.max(tensor))
            print(f"{indent}🔍 {name}: shape={list(tensor.shape)} dtype={tensor.dtype} unique={unique_vals} range=[{min_val}, {max_val}]")
            return
        
        # Convert to float for statistics
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        flat = tensor.flatten()
        
        # Statistical fingerprints
        mean_val = float(torch.mean(flat))
        std_val = float(torch.std(flat))
        min_val = float(torch.min(flat))
        max_val = float(torch.max(flat))
        non_zero_count = int(torch.count_nonzero(torch.abs(flat) > 1e-8))
        total_elements = int(flat.numel())
        
        print(f"{indent}🔍 {name}: shape={list(tensor.shape)} mean={mean_val:.6f} std={std_val:.6f} range=[{min_val:.6f}, {max_val:.6f}] nonzero={non_zero_count}/{total_elements}")
        
        # Check for problematic patterns
        if std_val < 1e-6:
            print(f"{indent}  ⚠️  Very low std ({std_val:.8f}) - possible zero/constant output")
        if abs(mean_val) < 1e-6 and std_val < 1e-6:
            print(f"{indent}  🔴 Near-zero output detected!")
    else:
        print(f"{indent}🔍 {name}: Not a tensor")

def debug_pytorch_inference(text="Test"):
    """Run PyTorch inference with layer-by-layer statistical output."""
    print(f"\n🔍 PYTORCH REFERENCE DEBUG: '{text}' → Audio")
    print("=" * 60)
    
    try:
        # Load model - use default hexgrad/Kokoro-82M
        model = KModel()
        print("✅ PyTorch model loaded")
        
        # Use simple character split like Rust debug
        phonemes = list(text.lower().replace(" ", ""))
        input_ids = [0]  # BOS
        for char in phonemes:
            token_id = model.vocab.get(char, 1)  # 1 for unknown
            input_ids.append(token_id)
        input_ids.append(0)  # EOS
        
        input_ids_tensor = torch.LongTensor([input_ids]).to(model.device)
        print_tensor_stats("Input IDs", input_ids_tensor)
        
        # Create voice embedding (use something reasonable)
        ref_s = torch.randn(1, 256).to(model.device) * 0.1  # Small random values
        print_tensor_stats("Voice Embedding", ref_s)
        
        print("\n🧠 LAYER-BY-LAYER PYTORCH PROCESSING:")
        
        # Use the model's forward_with_tokens for complete processing
        audio, pred_dur = model.forward_with_tokens(input_ids_tensor, ref_s, 1.0)
        print_tensor_stats("Final PyTorch audio", audio)
        
        print(f"\n✅ PYTORCH REFERENCE COMPLETE: {audio.numel()} audio samples")
        
        # Save for comparison
        audio_np = audio.cpu().numpy()
        print(f"PyTorch audio stats: mean={np.mean(audio_np):.6f}, std={np.std(audio_np):.6f}")
        
        return audio_np
        
    except Exception as e:
        print(f"❌ PyTorch debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Test"
    debug_pytorch_inference(text)