#!/usr/bin/env python3
"""Dump resampled data as raw binary so C++ can load it directly,
bypassing C++ resampling to isolate the inference comparison."""
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import json

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
GT = f"{PROJ}/tests/ground_truth"

# Reproduce the exact resampled volume
img = nib.load(f"{PROJ}/tests/example_ct.nii.gz")
canonical = nib.as_closest_canonical(img)
can_data = canonical.get_fdata(dtype=np.float32)

current_spacing = np.array(canonical.header.get_zooms()[:3])
target_spacing = np.array([1.5, 1.5, 1.5])
zoom_factors = current_spacing / target_spacing
resampled = zoom(can_data, zoom_factors, order=3, mode='nearest').astype(np.float32)

# Save in C-contiguous order (x-slowest, z-fastest)
np.ascontiguousarray(resampled).tofile(f"{GT}/py_resampled_corder.bin")
meta = {"shape": list(resampled.shape), "dtype": "float32",
        "spacing": [1.5, 1.5, 1.5]}
with open(f"{GT}/py_resampled_corder.json", "w") as f:
    json.dump(meta, f)
print(f"Saved resampled: shape={resampled.shape}, range=[{resampled.min():.1f}, {resampled.max():.1f}]")

# Also save the canonical data for comparing C++ canonical step
np.ascontiguousarray(can_data).tofile(f"{GT}/py_canonical_corder.bin")
meta2 = {"shape": list(can_data.shape), "dtype": "float32"}
with open(f"{GT}/py_canonical_corder.json", "w") as f:
    json.dump(meta2, f)
print(f"Saved canonical: shape={can_data.shape}")
