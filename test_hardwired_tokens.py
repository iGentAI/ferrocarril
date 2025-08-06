#!/usr/bin/env python3
"""
Extract exact token sequences from misaki for hardwiring into Rust test.
"""

import sys
sys.path.append('kokoro')
from kokoro.pipeline import KPipeline

def get_misaki_tokens(text, voice="af_heart"):
    """Get the exact token sequence that misaki produces."""
    print(f"🔍 MISAKI TOKEN EXTRACTION: '{text}'")
    print("=" * 50)
    
    # Use official KPipeline
    pipeline = KPipeline(lang_code='en-us', repo_id='hexgrad/Kokoro-82M')
    
    # Get phonemes from misaki
    results = list(pipeline(text, voice=voice, speed=1.0))
    
    for i, result in enumerate(results):
        print(f"Text: '{result.graphemes}'")
        print(f"Phonemes: '{result.phonemes}'")
        
        # Extract token IDs
        input_ids = list(filter(lambda x: x is not None, map(lambda p: pipeline.model.vocab.get(p), result.phonemes)))
        input_ids_with_bos_eos = [0] + input_ids + [0]
        
        print(f"Token IDs: {input_ids_with_bos_eos}")
        print(f"Length: {len(input_ids_with_bos_eos)}")
        
        # Print individual phoneme → token mappings
        print(f"Individual mappings:")
        print(f"  <BOS> → 0")
        for j, phoneme in enumerate(result.phonemes):
            token_id = pipeline.model.vocab.get(phoneme)
            print(f"  '{phoneme}' → {token_id}")
        print(f"  <EOS> → 0")
        
        return input_ids_with_bos_eos, result.phonemes

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "hello world"
    get_misaki_tokens(text)