#!/usr/bin/env python3
# Test what IPA symbols Phonesis can output vs Kokoro vocabulary

# Kokoro IPA symbols from the analysis
kokoro_ipa = [
    "̃", "ʣ", "ʥ", "ʦ", "ʨ", "ᵝ", "ꭧ", "ᵊ", "ɑ", "ɐ", "ɒ", "æ", "β", "ɔ",
    "ɕ", "ç", "ɖ", "ð", "ʤ", "ə", "ɚ", "ɛ", "ɜ", "ɟ", "ɡ", "ɥ", "ɨ", "ɪ",
    "ʝ", "ɯ", "ŋ", "ɳ", "ɲ", "ɴ", "ø", "ɸ", "θ", "œ", "ɹ", "ɾ", "ɻ",
    "ʁ", "ɽ", "ʂ", "ʃ", "ʈ", "ʧ", "ʊ", "ʋ", "ʌ", "ɣ", "ɤ", "χ", "ʎ",
    "ʒ", "ʔ", "ˈ", "ˌ", "ː", "ʰ", "ʲ", "↓", "→", "↗", "↘", "ᵻ"
]

# Also basic letters: a-z
kokoro_basic = list("abcdefghijklmnopqrstuvwxyz")

all_kokoro_phonemes = kokoro_ipa + kokoro_basic

print(f"🔍 PHONESIS vs KOKORO COMPREHENSIVE COMPATIBILITY")
print(f"Kokoro has {len(all_kokoro_phonemes)} phoneme tokens total")
print(f"Let\'s test some complex words to see Phonesis IPA range...")

# We\'ll analyze this after running the Rust test
print("\n📋 Need to test Phonesis with complex words to see its IPA output range")
print("This will show if Phonesis outputs the full IPA range Kokoro expects")
