#!/usr/bin/env python3
"""
Analyze Kokoro vocabulary structure and debug token mapping issues
"""

import json

def main():
    config = json.load(open('ferrocarril_weights/config.json'))
    vocab = config['vocab']
    
    print("🔍 COMPLETE KOKORO VOCABULARY ANALYSIS")
    print(f"Total vocabulary size: {len(vocab)}")
    print()
    
    # Analyze character types
    letters = [k for k in vocab.keys() if k.isalpha()]
    print(f"Letters: {len(letters)} entries")
    print("First 20 letters:")
    for k in sorted(letters)[:20]:
        print(f"  {repr(k)}: {vocab[k]}")
    print()
    
    # Test specific phoneme characters
    phonemes = "HH EH0 L OW0 UH0 W ER0 R L D"
    print(f"Testing G2P output: {phonemes}")
    print("Character mapping from phonemes:")
    
    mapped_chars = []
    total_chars = 0
    
    for phoneme in phonemes.split():
        print(f"  Phoneme '{phoneme}':")
        for char in phoneme:
            total_chars += 1
            if char.isdigit():
                print(f"    '{char}': STRESS MARKER (skipped)")
                continue
                
            if char in vocab:
                print(f"    '{char}': token_id={vocab[char]}")
                mapped_chars.append((char, vocab[char]))
            else:
                print(f"    '{char}': NOT IN VOCAB")
    
    print()
    print(f"Summary: {len(mapped_chars)} characters mapped out of {total_chars} total")
    print(f"Mapped characters: {[c for c, _ in mapped_chars]}")
    print(f"Token IDs: {[tid for _, tid in mapped_chars]}")
    
    # Check if this explains the short sequence issue
    print()
    print("DIAGNOSIS:")
    if len(mapped_chars) < 5:
        print("❌ CRITICAL: Very few characters mapping to tokens")
        print("This explains the extremely short sequences (5 tokens) and Layer 3 failures")
        print("Solution: Need proper phoneme tokenization strategy")
    else:
        print("✅ Character mapping looks reasonable")

if __name__ == "__main__":
    main()