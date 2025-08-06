#!/usr/bin/env python3
"""
Official PyTorch Kokoro Reference with Proper G2P
Uses the official KPipeline and misaki G2P for fair comparison with Rust implementation.
"""

import sys
sys.path.append('kokoro')
import torch
import numpy as np
from kokoro.pipeline import KPipeline
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

def debug_official_pytorch_inference(text="Test", voice="af_heart"):
    """Run official PyTorch inference with layer-by-layer statistical output."""
    print(f"\n🔍 OFFICIAL PYTORCH REFERENCE DEBUG: '{text}' → Audio")
    print("=" * 60)
    
    try:
        # Use official KPipeline for proper G2P
        pipeline = KPipeline(lang_code='en-us', repo_id='hexgrad/Kokoro-82M')
        print("✅ Official PyTorch pipeline loaded")
        
        print("\n🔤 OFFICIAL TOKENIZATION:")
        print(f"Input text: '{text}'")
        
        # Use official pipeline to process text
        results = list(pipeline(text, voice=voice, speed=1.0))
        
        for i, result in enumerate(results):
            print(f"  Chunk {i}: '{result.graphemes}'")
            print(f"  Phonemes: '{result.phonemes}'")
            
            # Extract input_ids from the phonemes like the model does
            input_ids = list(filter(lambda x: x is not None, map(lambda p: pipeline.model.vocab.get(p), result.phonemes)))
            input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
            print_tensor_stats("Official Input IDs", input_ids, "  ")
            
            # Get the voice pack
            ref_s = pipeline.load_voice(voice).to(pipeline.model.device)
            print_tensor_stats("Official Voice Embedding", ref_s, "  ")
            
            if result.audio is not None:
                print_tensor_stats("Official Final Audio", result.audio, "  ")
                
                # Save official PyTorch output for comparison
                audio_np = result.audio.cpu().numpy()
                np.save(f'official_pytorch_audio_{i}.npy', audio_np)
                print(f"  💾 Official PyTorch chunk {i} saved to: official_pytorch_audio_{i}.npy")
                
                return audio_np
        
    except Exception as e:
        print(f"❌ Official PyTorch debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "Test"
    voice = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
    debug_official_pytorch_inference(text, voice)