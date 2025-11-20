import numpy as np
import os

# Check the structure of a sample keyframe file
sample_file = "/root/movement/data/kinetics_processed/applauding/0BburtHMBts_000069_000079.npz"

if os.path.exists(sample_file):
    data = np.load(sample_file)
    print(f"File: {sample_file}")
    print(f"Keys in the file: {list(data.keys())}")

    for key in data.keys():
        array = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {array.shape}")
        print(f"  Dtype: {array.dtype}")
        if len(array.shape) > 0 and array.shape[0] > 0:
            print(f"  Sample values: {array[0] if array.ndim > 1 else array[:5]}")
else:
    print(f"File not found: {sample_file}")