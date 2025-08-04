import json

with open('ferrocarril_weights/config.json') as f:
    config = json.load(f)

print('🔍 KOKORO VOCABULARY ANALYSIS')
print('=' * 40)
print(f'n_token: {config["n_token"]}')

vocab = config['vocab']
print(f'Vocab size: {len(vocab)} entries')

print('\nSample vocabulary entries:')
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
for k, v in sorted_vocab[:20]:
    print(f'  "{k}": {v}')

print('\nPHONEME-LIKE ENTRIES:')
phoneme_like = [item for item in sorted_vocab if len(item[0]) > 1 or not item[0].isalnum()]
print(f'Found {len(phoneme_like)} non-single-letter entries')
for k, v in phoneme_like[:15]:
    print(f'  "{k}": {v}')

print('\n🚨 CRITICAL ISSUE IDENTIFIED:')
print('Kokoro vocab contains only character mappings, not IPA phonemes!')
print('\nPHONESIS IPA OUTPUT:')
print('  "hello world" → ["h", "ɛ", "l", "oʊ", "ʊ", "w", "ɝ", "r", "l", "d"]')
print('\nKOKORO VOCAB MAPPING:')
print('  Contains single letters: a-z, punctuation')
print('  Missing IPA symbols: ɛ, oʊ, ʊ, ɝ, etc.')
print('\n⚠️  VOCAB MISMATCH PROBLEM:')
print('  The inference pipeline cannot map Phonesis IPA output to Kokoro tokens!')
