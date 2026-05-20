import numpy as np
import zlib
import pickle
import os
def serialize_encoded(encoded_frames):
    return pickle.dumps(encoded_frames)
def compress_data(serialized_data):
    return zlib.compress(serialized_data, level=9)
def write_bin(encoded_frames, output_path, metadata):
    payload = {
        'metadata': metadata,
        'frames': encoded_frames
    }
    serialized = serialize_encoded(payload)
    compressed = compress_data(serialized)
    with open(output_path, 'wb') as f:
        f.write(compressed)
    original_size = len(serialized)
    compressed_size = len(compressed)


    print(f"Taille avant compression  : {original_size / 1024:.1f} Kbytes")
    print(f"Taille après compression  : {compressed_size / 1024:.1f} Kbytes")
    ratio = ( original_size/compressed_size )
    print(f"ration: {ratio:.2f}")
    gain=(1 - compressed_size/original_size)*100
    print(f"Reduction de              : {gain:.1f}%")


    print(f"Fichier écrit : {output_path}")
    return original_size, compressed_size
def read_bin(input_path):
    with open(input_path, 'rb') as f:
        compressed = f.read()
    serialized = zlib.decompress(compressed)
    payload = pickle.loads(serialized)
    return payload['frames'], payload['metadata']