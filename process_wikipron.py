#!/usr/bin/env python3
"""
Process WikiPron data for integration into Phonesis G2P library.
"""

import sys
import re
from collections import defaultdict, Counter
import argparse

# IPA to ARPABET mapping (Enhanced)
IPA_TO_ARPABET = {
    # Vowels
    'i': 'IY', 'iː': 'IY', 'ɪ': 'IH',
    'e': 'EH', 'ɛ': 'EH', 'æ': 'AE',
    'a': 'AA', 'ɑ': 'AA', 'ɒ': 'AA',
    'ɔ': 'AO', 'o': 'OW', 'oʊ': 'OW',
    'u': 'UW', 'uː': 'UW', 'ʊ': 'UH',
    'ʌ': 'AH', 'ə': 'AH',
    'ɝ': 'ER', 'ɚ': 'ER', 'ɜ': 'ER', 'ɜ:': 'ER',
    'ɐ': 'AH',
    
    # Diphthongs
    'aɪ': 'AY', 'aʊ': 'AW', 'ɔɪ': 'OY',
    'eɪ': 'EY', 'ai': 'AY',
    'au': 'AW', 'oi': 'OY', 
    
    # Consonants
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D',
    'k': 'K', 'g': 'G', 'ɡ': 'G',  # Different unicode g
    'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH',
    's': 'S', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH',
    'h': 'HH', 'x': 'HH',  # x is a voiceless velar fricative, close to h
    'tʃ': 'CH', 'dʒ': 'JH',
    'm': 'M', 'n': 'N', 'ŋ': 'NG',
    'l': 'L', 'ɫ': 'L', 
    'ɹ': 'R', 'r': 'R', 'ɾ': 'R',
    'w': 'W', 'j': 'Y', 'y': 'Y',
    'ʔ': '',  # Glottal stop (not in ARPABET)
    
    # Special symbols
    'ˈ': '',  # Primary stress (handle separately)
    'ˌ': '',  # Secondary stress (handle separately)
    'ː': '',  # Length mark (already handled in long vowels)
    '.': '',  # Syllable boundary
    'ʰ': '',  # Aspiration (not in ARPABET)
    'ʷ': 'W',  # Labialization
    'ʲ': 'Y',  # Palatalization
    '̩': '',   # Syllabic consonant (handled separately)
    '̃': '',   # Nasalization (not in ARPABET)
    '͡': '',   # Tie bar for affricates (already handled in compounds)
}

# Regular expressions for compound sounds
COMPOUND_PATTERNS = {
    r't͡ʃ': 'CH',
    r'd͡ʒ': 'JH',
    r't͡s': 'TS',  # Can be mapped to 'T S'
    r't͡ʂ': 'CH',  # Approximation
    r'aɪ': 'AY',
    r'aʊ': 'AW',
    r'ɔɪ': 'OY',
    r'eɪ': 'EY',
    r'oʊ': 'OW',
    r'iː': 'IY',
    r'uː': 'UW',
}


def normalize_ipa(ipa_text):
    """Normalize common IPA variants."""
    # Normalize g
    ipa_text = ipa_text.replace('ɡ', 'g')
    # Normalize single quote-like characters to standard apostrophe
    ipa_text = ipa_text.replace('\u2018', "'")  # Left single quote
    ipa_text = ipa_text.replace('\u2019', "'")  # Right single quote
    ipa_text = ipa_text.replace('\u201B', "'")  # Single high-reversed-9 quote
    ipa_text = ipa_text.replace('\u02BC', "'")  # Modifier letter apostrophe
    # Handle combined/decomposed Unicode
    ipa_text = ipa_text.replace('ẽ', 'e')  # Remove nasalization
    ipa_text = ipa_text.replace('ɹ̩', 'ɝ')  # Syllabic r to r-colored vowel
    ipa_text = ipa_text.replace('n̩', 'ən')  # Syllabic n
    ipa_text = ipa_text.replace('m̩', 'əm')  # Syllabic m
    ipa_text = ipa_text.replace('l̩', 'əl')  # Syllabic l
    return ipa_text

def convert_ipa_to_arpabet(ipa_text, debug=False):
    """Convert IPA transcription to ARPABET."""
    # Normalize
    ipa_text = normalize_ipa(ipa_text)
    
    # Handle compound patterns first
    for pattern, replacement in COMPOUND_PATTERNS.items():
        ipa_text = re.sub(pattern, replacement, ipa_text)
    
    # Now process remaining characters
    result = []
    i = 0
    stress = 0  # 0 = no stress, 1 = primary, 2 = secondary
    
    while i < len(ipa_text):
        # Handle stress markers
        if i < len(ipa_text) and ipa_text[i] == 'ˈ':
            stress = 1
            i += 1
            continue
        elif i < len(ipa_text) and ipa_text[i] == 'ˌ':
            stress = 2
            i += 1
            continue
        
        # Check for syllabic consonants
        if i + 1 < len(ipa_text) and ipa_text[i + 1] == '̩':
            consonant = ipa_text[i]
            if consonant in IPA_TO_ARPABET:
                result.append(IPA_TO_ARPABET[consonant])
            i += 2
            continue
        
        # Check for two-character sequences first
        matched = False
        if i + 1 < len(ipa_text):
            two_char = ipa_text[i:i+2]
            if two_char.lower() in IPA_TO_ARPABET:
                arpabet = IPA_TO_ARPABET[two_char.lower()]
                
                # Add stress to vowels
                if arpabet in ['IY', 'IH', 'EH', 'AE', 'AA', 'AO', 'OW', 'UW', 'UH', 
                              'AH', 'ER', 'AY', 'AW', 'OY', 'EY']:
                    if stress == 1:
                        arpabet += '1'
                    elif stress == 2:
                        arpabet += '2'
                    else:
                        arpabet += '0'
                    stress = 0  # Reset stress
                
                result.append(arpabet)
                i += 2
                matched = True
        
        # Single character
        if not matched and i < len(ipa_text):
            char = ipa_text[i]
            if char in IPA_TO_ARPABET:
                arpabet = IPA_TO_ARPABET[char]
                if arpabet:
                    # Add stress to vowels
                    if arpabet in ['IY', 'IH', 'EH', 'AE', 'AA', 'AO', 'OW', 'UW', 'UH', 
                                  'AH', 'ER', 'AY', 'AW', 'OY', 'EY']:
                        if stress == 1:
                            arpabet += '1'
                        elif stress == 2:
                            arpabet += '2'
                        else:
                            arpabet += '0'
                        stress = 0  # Reset stress
                    
                    result.append(arpabet)
            elif debug:
                print(f"Warning: Unknown IPA character '{char}' (U+{ord(char):04X})", file=sys.stderr)
            i += 1
    
    # If we have explicit uppercase IPA letters, convert them
    cleaned_result = []
    for item in result:
        if item in ['I', 'Y', 'U', 'W']:
            # These are likely meant to be the actual sounds
            if item == 'I': cleaned_result.append('IY')
            elif item == 'Y': cleaned_result.append('Y')
            elif item == 'U': cleaned_result.append('UW')
            elif item == 'W': cleaned_result.append('W')
        else:
            cleaned_result.append(item)
    
    return cleaned_result


def filter_word(word):
    """Filter out problematic words."""
    # Skip words with non-ASCII characters or special symbols
    if not re.match(r'^[a-zA-Z\'-]+$', word):
        return False
    # Skip very short words
    if len(word) < 2:
        return False
    # Skip overly long words
    if len(word) > 20:
        return False
    return True


def filter_pronunciation(pron):
    """Filter out potentially problematic pronunciations."""
    # Check for valid ARPABET phonemes
    valid_phonemes = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
        'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
    }
    
    for phoneme in pron:
        # Remove stress markers
        phoneme_base = phoneme.rstrip('012')
        if phoneme_base not in valid_phonemes:
            return False
    
    # Must have at least one vowel
    has_vowel = False
    vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    for phoneme in pron:
        phoneme_base = phoneme.rstrip('012')
        if phoneme_base in vowels:
            has_vowel = True
            break
    
    return has_vowel


def process_file(input_file, output_file=None, subset_size=100000, debug=False):
    """Process the input TSV file and write the ARPABET output."""
    word_count = 0
    skipped_count = 0
    pronunciations = defaultdict(list)
    
    # First pass: collect all pronunciations
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    skipped_count += 1
                    continue
                
                word, ipa = parts
                
                # Filter problematic words
                if not filter_word(word):
                    skipped_count += 1
                    continue
                
                # Replace space-separated IPA phonemes
                ipa_phonemes = ipa.split()
                arpabet_phonemes = []
                
                for phoneme in ipa_phonemes:
                    arpabet = convert_ipa_to_arpabet(phoneme, debug)
                    arpabet_phonemes.extend(arpabet)
                
                # Filter problematic pronunciations
                if arpabet_phonemes and filter_pronunciation(arpabet_phonemes):
                    pronunciations[word.upper()].append(' '.join(arpabet_phonemes))
                    word_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                if debug:
                    print(f"Error processing line '{line}': {e}", file=sys.stderr)
                skipped_count += 1
    
    # Second pass: select subset of words
    print(f"Total unique words: {len(pronunciations)}", file=sys.stderr)
    
    # Prioritize common words - use simple frequency heuristics
    common_suffixes = ['s', 'ed', 'ing', 'es', 'er', 'ly']
    base_forms = {}
    
    # Group by base forms
    for word in pronunciations:
        base = word
        for suffix in common_suffixes:
            if word.endswith(suffix.upper()) and len(word) - len(suffix) > 2:
                base = word[:-len(suffix)]
                break
        
        if base not in base_forms:
            base_forms[base] = []
        base_forms[base].append(word)
    
    # Select words
    selected_words = {}
    
    # First, get basic English vocabulary
    basic_words = [w for w in pronunciations if len(w) <= 6 and w.isalpha()]
    for word in sorted(basic_words)[:subset_size//2]:
        if word in pronunciations:
            selected_words[word] = pronunciations[word]
    
    # Then add more words
    remaining = subset_size - len(selected_words)
    for word in sorted(pronunciations.keys()):
        if word not in selected_words and remaining > 0:
            selected_words[word] = pronunciations[word]
            remaining -= 1
    
    print(f"Selected {len(selected_words)} words", file=sys.stderr)
    
    # Generate Rust code output
    rust_output = ['// Embedded WikiPron English pronunciation dictionary']
    rust_output.append('// Converted from IPA to ARPABET format')
    rust_output.append(f'// Total words: {len(selected_words)}')
    rust_output.append('pub const EMBEDDED_WIKIPRON_DICTIONARY: &str = r#"')
    
    for word in sorted(selected_words.keys()):
        for i, pron in enumerate(selected_words[word]):
            if i == 0:
                rust_output.append(f'{word} {pron}')
            else:
                rust_output.append(f'{word}({i+1}) {pron}')
    
    rust_output.append('"#;')
    
    # Write output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            if output_file.endswith('.rs'):
                outfile.write('\n'.join(rust_output))
            else:
                # Write as regular CMU dict format
                outfile.write(";;; WikiPron English pronunciation dictionary\n")
                outfile.write(";;; Converted from IPA to ARPABET format\n")
                outfile.write(f";;; Total words: {len(selected_words)}\n")
                outfile.write(";;;\n")
                
                for word in sorted(selected_words.keys()):
                    for i, pron in enumerate(selected_words[word]):
                        if i == 0:
                            outfile.write(f"{word}\t{pron}\n")
                        else:
                            outfile.write(f"{word}({i+1})\t{pron}\n")
    else:
        print('\n'.join(rust_output))
    
    print(f"Processed {word_count} pronunciations, skipped {skipped_count}", file=sys.stderr)
    print(f"Final dictionary contains {len(selected_words)} words", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WikiPron data for Phonesis")
    parser.add_argument("input_file", help="Input WikiPron TSV file")
    parser.add_argument("output_file", nargs='?', help="Output file (.rs for Rust code, .txt for CMU dict format)")
    parser.add_argument("--subset-size", type=int, default=75000, help="Number of words to include in subset")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file, args.subset_size, args.debug)