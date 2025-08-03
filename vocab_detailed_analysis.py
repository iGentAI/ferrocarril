#!/usr/bin/env python3
import json

config = json.load(open('ferrocarril_weights/config.json'))
vocab = config['vocab']

print('🔍 COMPLETE KOKORO VOCABULARY ANALYSIS')
print(f'Total vocabulary size: {len(vocab)}')
print()

# Categorize all vocabulary entries
letters = {}
symbols = {}
special_chars = {}

for char, token_id in vocab.items():
    if char.isalpha():
        letters[char] = token_id
    elif char in '.,!?;:()"—…':
        symbols[char] = token_id
    else:
        special_chars[char] = token_id

print(f'Alphabetic characters: {len(letters)} entries')
print('All letters in vocabulary:')
for i, (char, token_id) in enumerate(sorted(letters.items())):
    if i % 10 == 0:
        print()
    print(f'  {repr(char)}:{token_id}', end='')
print('\n')

print(f'Special IPA/phoneme characters: {len(special_chars)} entries')
print('All special phoneme chars:')
for char, token_id in sorted(special_chars.items()):
    print(f'  {repr(char)}: {token_id}')
print()

print(f'Punctuation: {len(symbols)} entries')
print('Punctuation chars:')
for char, token_id in sorted(symbols.items()):
    print(f'  {repr(char)}: {token_id}')
print()

# Check what would work for common English phonemes
print('Common English phoneme characters in Kokoro vocab:')
common_phonemes = {
    'h': 'voiceless glottal fricative', 'l': 'lateral approximant', 
    'r': 'rhotic consonant', 'd': 'voiced alveolar stop',
    't': 'voiceless alveolar stop', 'n': 'alveolar nasal',
    's': 'voiceless alveolar fricative', 'w': 'labial-velar approximant',
    'e': 'mid front vowel', 'i': 'close front vowel', 
    'o': 'mid back vowel', 'u': 'close back vowel', 
    'a': 'open central vowel'
}

for phone, description in common_phonemes.items():
    if phone in vocab:
        print(f'  ✅ {repr(phone)}: {vocab[phone]} ({description})')
    else:
        print(f'  ❌ {repr(phone)}: NOT IN VOCAB ({description})')

