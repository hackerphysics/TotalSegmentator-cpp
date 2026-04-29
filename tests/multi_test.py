#!/usr/bin/env python3
"""Generate multiple test cases and Python ground truth for C++ validation."""
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from pathlib import Path
import json

OUT = Path("tests/multi_test_data")
OUT.mkdir(exist_ok=True)

def make_ct(shape, spacing, orientation='RAS', noise_seed=42):
    """Create a synthetic CT-like volume with structures."""
    rng = np.random.RandomState(noise_seed % (2**31))
    data = rng.uniform(-1000, -200, shape).astype(np.float32)
    cx, cy, cz = [s//2 for s in shape]
    rx, ry, rz = [max(s//4, 1) for s in shape]
    x0, x1 = max(cx-rx,0), min(cx+rx, shape[0])
    y0, y1 = max(cy-ry,0), min(cy+ry, shape[1])
    z0, z1 = max(cz-rz,0), min(cz+rz, shape[2])
    data[x0:x1, y0:y1, z0:z1] = rng.uniform(0, 200, (x1-x0, y1-y0, z1-z0)).astype(np.float32)
    br = max(min(shape) // 10, 1)
    bx, by, bz = min(cx+rx//2, shape[0]-br-1), min(cy+ry//2, shape[1]-br-1), min(cz+rz//2, shape[2]-br-1)
    bx, by, bz = max(bx, br), max(by, br), max(bz, br)
    data[bx-br:bx+br, by-br:by+br, bz-br:bz+br] = rng.uniform(300, 1000,
        (min(2*br, shape[0]-bx+br), min(2*br, shape[1]-by+br), min(2*br, shape[2]-bz+br))).astype(np.float32)

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = spacing[i]
    if orientation == 'LAS':
        affine[0, 0] = -spacing[0]
        affine[0, 3] = spacing[0] * (shape[0] - 1)
    elif orientation == 'LPS':
        affine[0, 0] = -spacing[0]
        affine[0, 3] = spacing[0] * (shape[0] - 1)
        affine[1, 1] = -spacing[1]
        affine[1, 3] = spacing[1] * (shape[1] - 1)

    return nib.Nifti1Image(data, affine)

def run_python_pipeline(img_path):
    img = nib.load(str(img_path))
    data = img.get_fdata(dtype=np.float32)
    canonical = nib.as_closest_canonical(img)
    can_data = canonical.get_fdata(dtype=np.float32)
    can_spacing = np.array(canonical.header.get_zooms()[:3], dtype=np.float64)
    ts = np.array([1.5, 1.5, 1.5])
    zoom_factors = can_spacing / ts
    resampled = zoom(can_data, zoom_factors, order=3, mode='nearest')
    return {
        'original_shape': list(data.shape),
        'canonical_shape': list(can_data.shape),
        'canonical_spacing': can_spacing.tolist(),
        'resampled_shape': list(resampled.shape),
        'resampled': resampled,
    }

cases = [
    ("case1_small_iso", (64, 64, 32), (2.0, 2.0, 2.0), "RAS"),
    ("case2_medium_aniso", (128, 100, 80), (1.5, 1.5, 3.0), "RAS"),
    ("case3_large_iso", (200, 180, 150), (1.0, 1.0, 1.0), "RAS"),
    ("case4_las_orient", (96, 80, 64), (2.5, 2.5, 2.5), "LAS"),
    ("case5_lps_orient", (110, 90, 70), (2.0, 2.0, 2.0), "LPS"),
    ("case6_tiny", (32, 28, 24), (4.0, 4.0, 4.0), "RAS"),
    ("case7_non_cubic", (150, 60, 200), (1.0, 3.0, 0.8), "RAS"),
    ("case8_already_1p5", (100, 100, 100), (1.5, 1.5, 1.5), "RAS"),
]

print(f"Generating {len(cases)} test cases...")
manifest = {}

for name, shape, spacing, orient in cases:
    print(f"\n--- {name}: shape={shape} spacing={spacing} orient={orient} ---")
    img = make_ct(shape, spacing, orient, noise_seed=abs(hash(name)))
    nii_path = OUT / f"{name}.nii.gz"
    nib.save(img, str(nii_path))
    result = run_python_pipeline(nii_path)
    gt_path = OUT / f"{name}_resampled.bin"
    resampled = np.ascontiguousarray(result['resampled'])
    gt_path.write_bytes(resampled.tobytes())
    manifest[name] = {
        'file': f"{name}.nii.gz",
        'original_shape': result['original_shape'],
        'canonical_shape': result['canonical_shape'],
        'canonical_spacing': result['canonical_spacing'],
        'resampled_shape': result['resampled_shape'],
        'resampled_dtype': str(resampled.dtype),
    }
    print(f"  Canonical: {result['canonical_shape']} Resampled: {result['resampled_shape']}")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nDone. {len(cases)} cases generated.")
