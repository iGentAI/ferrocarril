#!/usr/bin/env python3
# Test Phonesis G2P functionality for IPA output

import sys
sys.path.append('phonesis')

print("🧪 TESTING PHONESIS G2P - IPA OUTPUT VERIFICATION")
print("=" * 50)

# Test if we can import and use Phonesis
try:
    # This would be the Python interface if available
    print("Testing Phonesis G2P functionality:")
    
    test_texts = [
        "hello world",
        "the quick brown fox",
        "text to speech",
        "Kokoro TTS"
    ]
    
    # Since we don't have Python bindings, let's test the Rust implementation via a simple test
    print("\nNote: Need to test via Rust implementation")
    print("Creating Rust test to verify Phonesis G2P output...")
    
except Exception as e:
    print(f"Error testing Phonesis: {e}")
    print("Will create Rust test instead")

print("\n📝 Creating Rust test for Phonesis G2P verification...")
