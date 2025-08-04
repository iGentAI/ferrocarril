# COMPREHENSIVE SYMBOL SPACE COMPARISON
import json

with open('ferrocarril_weights/config.json') as f:
    config = json.load(f)

vocab = config['vocab']

# Phonesis IPA symbols from comprehensive test
phonesis_ipa = [
    "b", "d", "f", "g", "h", "i", "k", "l", "m", "n",
    "oʊ", "p", "r", "s", "t", "v", "w", "z", "æ", "ð",
    "ŋ", "ɑ", "ɔ", "ɛ", "ɝ", "ɪ", "ʃ", "ʊ", "ʌ", "ʒ", "θ"
]

# ALL IPA symbols in Kokoro (from previous analysis)
ipa_chars = 'ɑɐɒæɔəɚɛɜɝɨɪɯøœɤʊʌβɕçɖðʤɟɡɥʝŋɳɲɴɸθɹɾɻʁɽʂʃʈʧʦʨʋɣχʎʒʔˈˌːʰʲ̃ᵊᵝᵻ↓→↗↘ꭧʣʥ'
kokoro_ipa = [k for k in vocab.keys() if any(c in ipa_chars for c in k)]
kokoro_basic = [k for k in vocab.keys() if k in 'abcdefghijklmnopqrstuvwxyz']

print('🔍 PHONESIS ↔ KOKORO SYMBOL SPACE COMPARISON')
print('=' * 60)
print(f'Phonesis can produce: {len(phonesis_ipa)} IPA symbols')
print(f'Kokoro vocabulary has: {len(kokoro_ipa)} IPA symbols + {len(kokoro_basic)} basic letters')
print(f'Total Kokoro phonemes: {len(kokoro_ipa) + len(kokoro_basic)}')

print('\n🎯 CRITICAL ALIGNMENT ANALYSIS:')

# Check which Phonesis symbols are in Kokoro
phonesis_in_kokoro = []
phonesis_missing_from_kokoro = []

for symbol in phonesis_ipa:
    if symbol in vocab:
        phonesis_in_kokoro.append((symbol, vocab[symbol]))
    else:
        phonesis_missing_from_kokoro.append(symbol)

# Check which Kokoro IPA symbols Phonesis can't produce
kokoro_missing_from_phonesis = []
for symbol in kokoro_ipa:
    if symbol not in phonesis_ipa and symbol not in 'abcdefghijklmnopqrstuvwxyz':
        kokoro_missing_from_phonesis.append(symbol)

print(f'\n✅ PHONESIS SYMBOLS MATCHED IN KOKORO: {len(phonesis_in_kokoro)}/{len(phonesis_ipa)}')
for symbol, token_id in sorted(phonesis_in_kokoro, key=lambda x: x[1]):
    print(f'  {symbol} → token {token_id}')

print(f'\n❌ PHONESIS SYMBOLS MISSING FROM KOKORO: {len(phonesis_missing_from_kokoro)}')
for symbol in phonesis_missing_from_kokoro:
    print(f'  {symbol} (not in Kokoro vocabulary)')

print(f'\n⚠️ KOKORO IPA SYMBOLS NOT PRODUCED BY PHONESIS: {len(kokoro_missing_from_phonesis)}')
for symbol in sorted(kokoro_missing_from_phonesis):
    token_id = vocab[symbol]
    print(f'  {symbol} (token {token_id}) - Kokoro has this but Phonesis cannot produce')

# Calculate coverage
phonesis_coverage = len(phonesis_in_kokoro) / len(phonesis_ipa) * 100
kokoro_coverage = len(phonesis_in_kokoro) / len(kokoro_ipa) * 100

print(f'\n📊 SYMBOL SPACE COVERAGE ANALYSIS:')
print(f'  Phonesis→Kokoro coverage: {phonesis_coverage:.1f}% ({len(phonesis_in_kokoro)}/{len(phonesis_ipa)} mapped)')
print(f'  Kokoro IPA coverage by Phonesis: {kokoro_coverage:.1f}% ({len(phonesis_in_kokoro)}/{len(kokoro_ipa)} available)')

if len(phonesis_missing_from_kokoro) <= 3:
    print('\n✅ EXCELLENT ALIGNMENT: Phonesis outputs almost all symbols Kokoro needs')
else:
    print('\n⚠️ SIGNIFICANT GAP: Many Phonesis symbols not in Kokoro')
