#!/usr/bin/env python3
"""
Test loading of the binary dictionary file.
"""

import struct
import json


def read_binary_dictionary(path):
    """Read and parse binary dictionary file."""
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'PHON':
            raise ValueError("Invalid file format")
        
        version = struct.unpack('<H', f.read(2))[0]
        print(f"Version: {version}")
        
        entry_count = struct.unpack('<I', f.read(4))[0]
        print(f"Entry count: {entry_count}")
        
        metadata_offset = struct.unpack('<I', f.read(4))[0]
        print(f"Metadata offset: {metadata_offset}")
        
        checksum = struct.unpack('<H', f.read(2))[0]
        print(f"Checksum: {checksum}")
        
        # Read metadata if available
        if metadata_offset > 0:
            f.seek(metadata_offset)
            metadata_len = struct.unpack('<I', f.read(4))[0]
            if metadata_len > 0:
                metadata_bytes = f.read(metadata_len)
                metadata = json.loads(metadata_bytes.decode('utf-8'))
                print("\nMetadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dictionary.bin>")
        sys.exit(1)
    
    read_binary_dictionary(sys.argv[1])