#!/usr/bin/env python3
# Direct test of Phonesis G2P output

print("🧪 TESTING PHONESIS G2P DIRECT OUTPUT")
print("=" * 40)

# Since Phonesis is a Rust library, let's examine its output via the Rust test files
# that just ran successfully

print("\nChecking for G2P test outputs in Phonesis...")

# Let's create a simple Rust program that links to the existing Phonesis tests
print("\nCreating simple G2P test to verify IPA output...")

test_program = '''
use phonesis::{G2PEngine, PhonemeStandard};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let engine = G2PEngine::new(PhonemeStandard::IPA)?;
    
    let test_words = ["hello", "world", "TTS", "Kokoro"];
    
    for word in &test_words {
        let phonemes = engine.convert(word)?;
        println!("{}: {}", word, phonemes.to_string());
    }
    
    Ok(())
}
'''

with open('test_phonesis_simple.rs', 'w') as f:
    f.write(test_program)

print("Created test_phonesis_simple.rs for direct G2P testing")
