import json

with open('ferrocarril_weights/config.json') as f:
    config = json.load(f)

vocab = config['vocab']
phonesis_ipa = ['h', 'ɛ', 'l', 'oʊ', 'ʊ', 'w', 'ɝ', 'r', 'l', 'd']

print('🔍 DIRECT IPA MATCHING CHECK')
print('Phonesis IPA for "hello world": h ɛ l oʊ ʊ w ɝ r l d')
print()
print('MATCHING ANALYSIS:')
for phoneme in phonesis_ipa:
    print(f'  {phoneme}: ', end='')
    if phoneme in vocab:
        print(f'✅ token {vocab[phoneme]}')
    else:
        print('❌ NOT IN VOCAB')

print()
missing = [p for p in phonesis_ipa if p not in vocab]
present = [p for p in phonesis_ipa if p in vocab]
print(f'SUMMARY: {len(present)}/{len(phonesis_ipa)} phonemes directly mappable')
print(f'Missing: {missing}')
print(f'Present: {present}')
if len(missing) <= 2:
    print('✅ MOSTLY COMPATIBLE - just handle the few missing ones!')
