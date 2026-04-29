#!/usr/bin/env python3
"""Generate ground truth for end-to-end C++ test.
Saves intermediate results at each pipeline stage for debugging."""
import os, json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
GT_DIR = f"{PROJ}/tests/ground_truth"

# Load and save canonical + resampled data for C++ comparison
img = nib.load(f"{PROJ}/tests/example_ct.nii.gz")
canonical = nib.as_closest_canonical(img)
can_data = canonical.get_fdata(dtype=np.float32)

# Save resampled data (this is the input to ONNX model)
current_spacing = np.array(canonical.header.get_zooms()[:3])
target_spacing = np.array([1.5, 1.5, 1.5])
zoom_factors = current_spacing / target_spacing
resampled = zoom(can_data, zoom_factors, order=3, mode='nearest')

# Save as raw binary for C++ to load and compare
np.ascontiguousarray(resampled).tofile(f"{GT_DIR}/e2e_resampled.bin")
with open(f"{GT_DIR}/e2e_resampled.json", "w") as f:
    json.dump({"shape": list(resampled.shape), "dtype": "float32"}, f)
print(f"Resampled: shape={resampled.shape}")

# Save the python seg result as raw binary
seg = np.load(f"{GT_DIR}/python_seg_task291.npy")
np.ascontiguousarray(seg).tofile(f"{GT_DIR}/e2e_python_seg.bin")
with open(f"{GT_DIR}/e2e_python_seg.json", "w") as f:
    json.dump({"shape": list(seg.shape), "dtype": "uint8", "labels": sorted(int(x) for x in np.unique(seg))}, f)
print(f"Python seg: shape={seg.shape}, labels={np.unique(seg)}")
