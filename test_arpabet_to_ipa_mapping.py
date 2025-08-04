#!/usr/bin/env python3
# Create ARPABET to Kokoro IPA mapping

import json

with open('ferrocarril_weights/config.json') as f:
    config = json.load(f)

vocab = config['vocab']
print('🔧 ARPABET → KOKORO IPA MAPPING SOLUTION')
print('=' * 50)

# Extract IPA symbols from Kokoro vocab
ipa_symbols = {k: v for k, v in vocab.items() if any(c in k for c in ['ʌ','ə','ɛ','ɪ','ʊ','ɑ','ɔ','æ','ɝ','ɚ','ŋ','θ','ð','ʃ','ʒ'])}
print(f'Kokoro IPA symbols: {len(ipa_symbols)}')
for symbol, token_id in sorted(ipa_symbols.items(), key=lambda x: x[1]):
    print(f'  "{symbol}": {token_id}')

print('\n📝 ARPABET → IPA MAPPING NEEDED:')
# Based on Phonesis ARPABET output: HH EH0 L OW0 UH0 W ER0 R L D
arpabet_to_ipa = {
    'HH': 'h',
    'EH0': 'ɛ',   # Strip stress marker 0
    'EH1': 'ɛ',   # Strip stress marker 1
    'EH2': 'ɛ',   # Strip stress marker 2
    'L': 'l',
    'OW0': 'o',   # Map diphthong to simple vowel
    'OW1': 'o',
    'OW2': 'o', 
    'UH0': 'ʊ',   # Strip stress
    'UH1': 'ʊ',
    'UH2': 'ʊ',
    'W': 'w',
    'ER0': 'ɚ',   # R-colored vowel
    'ER1': 'ɚ',
    'ER2': 'ɚ',
    'R': 'r',
    'D': 'd',
    'T': 't',
    'K': 'k',
    'AH0': 'ʌ',   # Map to schwa variants
    'S': 's',
    'IY': 'i'
}

print('\nSUGGESTED ARPABET → KOKORO IPA MAPPING:')
for arpabet, ipa in arpabet_to_ipa.items():
    if ipa in vocab:
        print(f'  {arpabet:4} → {ipa} (token {vocab[ipa]})')
    else:
        print(f'  {arpabet:4} → {ipa} (❌ NOT IN KOKORO VOCAB)')

print(f'\n🎯 SOLUTION FOR INFERENCE PIPELINE:')
print('1. Use Phonesis ARPABET output (native format)')
print('2. Strip stress markers from ARPABET phonemes')
print('3. Map simplified ARPABET to Kokoro IPA vocabulary')
print('4. Handle missing mappings with phoneme approximations')
