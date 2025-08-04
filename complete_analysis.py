import json

with open('ferrocarril_weights/config.json') as f:
    config = json.load(f)
print('🔍 COMPLETE KOKORO-PHONESIS IPA COMPATIBILITY ANALYSIS')
print('=' * 60)
vocab = config['vocab']
print(f'Total Kokoro tokens: {len(vocab)}')

# Get ALL IPA symbols from Kokoro
ipa_chars = 'ɑɐɒæɔəɚɛɜɝɨɪɯøœɤʊʌβɕçɖðʤɟɡɥʝŋɳɲɴɸθɹɾɻʁɽʂʃʈʧʦʨʋɣχʎʒʔˈˌːʰʲ̃ᵊᵝᵻ↓→↗↘ꭧʣʥ'
all_ipa_in_kokoro = [(k, v) for k, v in vocab.items() if any(c in ipa_chars for c in k)]

print(f'\n🎯 ALL IPA SYMBOLS IN KOKORO: {len(all_ipa_in_kokoro)}')
for token, token_id in sorted(all_ipa_in_kokoro, key=lambda x: x[1]):
    print(f'  {token_id:3}: "{token}"')

# Basic letters that are also IPA
basic_ipa_letters = [k for k in vocab.keys() if k in 'abcdefghijklmnopqrstuvwxyz']
print(f'\n📝 BASIC LETTERS (also IPA): {len(basic_ipa_letters)}')
print('  ' + ' '.join(sorted(basic_ipa_letters)))

total_phoneme_tokens = len(all_ipa_in_kokoro) + len(basic_ipa_letters)
print(f'\n📊 PHONEME TOKEN SUMMARY:')
print(f'  IPA-specific symbols: {len(all_ipa_in_kokoro)}')
print(f'  Basic letter phonemes: {len(basic_ipa_letters)}')
print(f'  Total phoneme vocabulary: {total_phoneme_tokens} out of {len(vocab)} tokens')
print(f'  Phoneme coverage: {total_phoneme_tokens/len(vocab)*100:.1f}%')
