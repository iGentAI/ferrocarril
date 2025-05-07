#!/usr/bin/env python3
"""
Script to convert IPA pronunciations from WikiPron to ARPABET format.
"""

import sys
import re
from collections import defaultdict

# IPA to ARPABET mapping
# Note: This is a simplified mapping and might need refinement
IPA_TO_ARPABET = {
    # Vowels
    'i': 'IY', 'iː': 'IY', 'ɪ': 'IH',
    'e': 'EH', 'ɛ': 'EH', 'æ': 'AE',
    'a': 'AA', 'ɑ': 'AA', 'ɒ': 'AA',
    'ɔ': 'AO', 'o': 'OW', 'oʊ': 'OW',
    'u': 'UW', 'uː': 'UW', 'ʊ': 'UH',
    'ʌ': 'AH', 'ə': 'AH',
    'ɝ': 'ER', 'ɚ': 'ER', 'ɜ': 'ER',
    'ɐ': 'AH',
    
    # Diphthongs
    'aɪ': 'AY', 'aʊ': 'AW', 'ɔɪ': 'OY',
    'eɪ': 'EY', 'eɪ': 'EY', 'ai': 'AY',
    'au': 'AW', 'oi': 'OY', 
    
    # Consonants
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D',
    'k': 'K', 'g': 'G', 'ɡ': 'G',  # Different unicode g
    'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH',
    's': 'S', 'z': 'Z', 'ʃ': 'SH', 'ʒ': 'ZH',
    'h': 'HH',
    'tʃ': 'CH', 'dʒ': 'JH',
    'm': 'M', 'n': 'N', 'ŋ': 'NG',
    'l': 'L', 'ɹ': 'R', 'r': 'R',
    'w': 'W', 'j': 'Y',
    'ʔ': '',  # Glottal stop (not in ARPABET)
    
    # Special symbols
    'ˈ': '',  # Primary stress (handle separately)
    'ˌ': '',  # Secondary stress (handle separately)
    'ː': '',  # Length mark (already handled in long vowels)
    '.': '',  # Syllable boundary
    'ʰ': '',  # Aspiration (not in ARPABET)
    'ʷ': '',  # Labialization (not in ARPABET)
    'ʲ': '',  # Palatalization (not in ARPABET)
    '̩': '',   # Syllabic consonant (handled separately)
    '̃': '',   # Nasalization (not in ARPABET)
}

# Regular expressions for compound sounds
COMPOUND_PATTERNS = {
    r'tʃ': 'CH',
    r'dʒ': 'JH',
    r'aɪ': 'AY',
    r'aʊ': 'AW',
    r'ɔɪ': 'OY',
    r'eɪ': 'EY',
    r'oʊ': 'OW',
    r'iː': 'IY',
    r'uː': 'UW',
}


def convert_ipa_to_arpabet(ipa_text):
    """Convert IPA transcription to ARPABET."""
    # First normalize some Unicode variations
    ipa_text = ipa_text.replace('ɡ', 'g')  # Unicode ɡ to regular g
    
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
                continue
        
        # Single character
        if i < len(ipa_text):
            char = ipa_text[i]
            if char in IPA_TO_ARPABET:
                arpabet = IPA_TO_ARPABET[char]
                # Words like CH, JH might have been combined already, skip empty replacements
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
            else:
                # Unknown character - skip or handle differently
                print(f"Warning: Unknown IPA character '{char}' (U+{ord(char):04X})", file=sys.stderr)
        i += 1
    
    return result


def process_file(input_file, output_file):
    """Process the input TSV file and write the ARPABET output."""
    word_count = 0
    skipped_count = 0
    pronunciations = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line: {line}", file=sys.stderr)
                    skipped_count += 1
                    continue
                
                word, ipa = parts
                
                # Replace space-separated IPA phonemes
                ipa_phonemes = ipa.split()
                arpabet_phonemes = []
                
                for phoneme in ipa_phonemes:
                    arpabet = convert_ipa_to_arpabet(phoneme)
                    arpabet_phonemes.extend(arpabet)
                
                if arpabet_phonemes:
                    pronunciations[word.upper()].append(' '.join(arpabet_phonemes))
                    word_count += 1
                
            except Exception as e:
                print(f"Error processing line '{line}': {e}", file=sys.stderr)
                skipped_count += 1
    
    # Write output in CMU dict format
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Header
        outfile.write(";;; WikiPron English pronunciation dictionary\n")
        outfile.write(";;; Converted from IPA to ARPABET format\n")
        outfile.write(f";;; Total words: {word_count}\n")
        outfile.write(";;;\n")
        
        for word in sorted(pronunciations.keys()):
            for i, pron in enumerate(pronunciations[word]):
                if i == 0:
                    outfile.write(f"{word}\t{pron}\n")
                else:
                    outfile.write(f"{word}({i+1})\t{pron}\n")
    
    print(f"Processed {word_count} words, skipped {skipped_count} lines")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.tsv output.txt", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_file(input_file, output_file)