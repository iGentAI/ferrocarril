#!/usr/bin/env python3
"""
Generate a binary dictionary file from WikiPron data.
"""

import os
import sys
import struct
import argparse
import json
from collections import defaultdict


class BinaryDictionaryWriter:
    """Write pronunciation dictionary to binary format."""
    
    MAGIC_BYTES = b'PHON'
    VERSION = 1
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.string_table = []
        self.string_indices = {}
        self.nodes = []
        self.values = []
        
    def add_to_string_table(self, s):
        """Add a string to the string table and return its index."""
        if s in self.string_indices:
            return self.string_indices[s]
        
        index = len(self.string_table)
        self.string_table.append(s)
        self.string_indices[s] = index
        return index
    
    def build_trie(self, pronunciations):
        """Build a compact trie from the pronunciation dictionary."""
        # First node is always the root
        self.nodes.append({'children': {}, 'value': None})
        
        for word, pron_list in pronunciations.items():
            for pron in pron_list:
                node_idx = 0  # Start at root
                
                # Navigate/create path
                for char in word.lower():
                    if char not in self.nodes[node_idx]['children']:
                        new_idx = len(self.nodes)
                        self.nodes.append({'children': {}, 'value': None})
                        self.nodes[node_idx]['children'][char] = new_idx
                    
                    node_idx = self.nodes[node_idx]['children'][char]
                
                # Store pronunciation
                value_idx = len(self.values)
                self.values.append(pron)
                self.nodes[node_idx]['value'] = value_idx
    
    def write(self, word_pronunciations, metadata=None):
        """Write the binary dictionary file."""
        # Build the trie structure
        self.build_trie(word_pronunciations)
        
        # Pre-process values to populate string table
        processed_values = []
        for value in self.values:
            phonemes = value.split()
            processed_phonemes = []
            
            for phoneme in phonemes:
                # Split phoneme and stress
                if phoneme[-1] in '012':
                    symbol = phoneme[:-1]
                    stress = int(phoneme[-1])
                else:
                    symbol = phoneme
                    stress = -1  # No stress
                
                symbol_idx = self.add_to_string_table(symbol)
                processed_phonemes.append((symbol_idx, stress))
            
            processed_values.append(processed_phonemes)
        
        # Write to a temporary file first
        temp_path = self.output_path + ".tmp"
        
        with open(temp_path, 'wb') as f:
            # Write header
            f.write(self.MAGIC_BYTES)
            f.write(struct.pack('<H', self.VERSION))  # Version
            f.write(struct.pack('<I', len(self.values)))  # Entry count
            
            # Placeholder for metadata offset (will update later)
            metadata_offset_pos = f.tell()
            f.write(struct.pack('<I', 0))
            
            # Placeholder for checksum
            checksum_pos = f.tell()
            f.write(struct.pack('<H', 0))
            
            # Write string table
            string_table_offset = f.tell()
            f.write(struct.pack('<I', len(self.string_table)))
            
            # Write each string with length prefix
            for s in self.string_table:
                s_bytes = s.encode('utf-8')
                f.write(struct.pack('<H', len(s_bytes)))
                f.write(s_bytes)
            
            # Write nodes
            nodes_offset = f.tell()
            f.write(struct.pack('<I', len(self.nodes)))
            
            for node in self.nodes:
                # Write children count
                f.write(struct.pack('<H', len(node['children'])))
                
                # Write each child: char -> node_index
                for char, idx in node['children'].items():
                    char_bytes = char.encode('utf-8')
                    f.write(struct.pack('<B', len(char_bytes)))
                    f.write(char_bytes)
                    f.write(struct.pack('<I', idx))
                
                # Write value index (-1 if None)
                value_idx = -1 if node['value'] is None else node['value']
                f.write(struct.pack('<i', value_idx))
            
            # Write values (phoneme sequences)
            values_offset = f.tell()
            f.write(struct.pack('<I', len(self.values)))
            
            for processed_phonemes in processed_values:
                f.write(struct.pack('<B', len(processed_phonemes)))
                
                for symbol_idx, stress in processed_phonemes:
                    f.write(struct.pack('<H', symbol_idx))  # Symbol index
                    f.write(struct.pack('<b', stress))  # Stress level
            
            # Write metadata
            metadata_offset = f.tell()
            if metadata:
                metadata_json = json.dumps(metadata).encode('utf-8')
                f.write(struct.pack('<I', len(metadata_json)))
                f.write(metadata_json)
            else:
                f.write(struct.pack('<I', 0))
                
        # Now reopen the file to update metadata offset and checksum
        with open(temp_path, 'r+b') as f:
            # Update metadata offset
            f.seek(metadata_offset_pos)
            f.write(struct.pack('<I', metadata_offset))
            
            # Calculate checksum (simple sum for now)
            f.seek(0)
            content = f.read()
            checksum = sum(content[16:]) % 65536  # Skip header including checksum field
            
            # Update checksum
            f.seek(checksum_pos)
            f.write(struct.pack('<H', checksum))
            
        # Rename temp file to final output
        os.replace(temp_path, self.output_path)


def generate_binary_dictionary(input_file, output_file, subset_size=75000):
    """Generate a binary dictionary file from WikiPron data."""
    print(f"Processing {input_file}...")
    
    # Process WikiPron data
    pronunciations = defaultdict(list)
    word_count = 0
    skipped_count = 0
    
    # Use the existing process_file function
    import process_wikipron
    
    # Capture the output of process_file
    temp_output = "temp_processed.txt"
    process_wikipron.process_file(input_file, temp_output, subset_size=subset_size)
    
    # Parse the processed output
    with open(temp_output, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            word, pron = parts
            if '(' in word:
                word = word[:word.index('(')]
            
            pronunciations[word].append(pron)
            word_count += 1
    
    # Clean up temp file
    os.remove(temp_output)
    
    print(f"Loaded {word_count} pronunciations for {len(pronunciations)} unique words")
    
    # Write binary dictionary
    writer = BinaryDictionaryWriter(output_file)
    
    metadata = {
        "name": "WikiPron English Dictionary",
        "version": "1.0.0",
        "description": "English pronunciation dictionary converted from WikiPron data",
        "entry_count": word_count,
        "source": "WikiPron",
        "license": "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "language": "en-us",
        "phoneme_standard": "ARPABET"
    }
    
    writer.write(pronunciations, metadata)
    
    print(f"Binary dictionary written to {output_file}")
    print(f"File size: {os.path.getsize(output_file):,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary dictionary from WikiPron data")
    parser.add_argument("input_file", help="Input WikiPron TSV file")
    parser.add_argument("output_file", help="Output binary dictionary file")
    parser.add_argument("--subset-size", type=int, default=75000, help="Number of words to include")
    
    args = parser.parse_args()
    
    generate_binary_dictionary(args.input_file, args.output_file, args.subset_size)