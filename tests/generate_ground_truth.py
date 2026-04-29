#!/usr/bin/env python3
"""Generate ground truth data for C++ unit tests.
Saves numpy arrays as raw binary + metadata json for each module."""

import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from pathlib import Path

OUT = Path("tests/ground_truth")
OUT.mkdir(exist_ok=True)

def save_raw(arr, name):
    """Save array as raw binary + shape/dtype json."""
    arr = np.ascontiguousarray(arr)
    (OUT / f"{name}.bin").write_bytes(arr.tobytes())
    meta = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    (OUT / f"{name}.json").write_text(json.dumps(meta))
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")

# ===== 1. NIfTI I/O =====
print("=== NIfTI I/O ground truth ===")
img = nib.load("tests/example_ct.nii.gz")
data = img.get_fdata(dtype=np.float32)
save_raw(data, "nifti_io_data")
save_raw(np.array(img.affine, dtype=np.float64), "nifti_io_affine")
spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
save_raw(spacing, "nifti_io_spacing")
shape = np.array(data.shape, dtype=np.int32)
save_raw(shape, "nifti_io_shape")

# ===== 2. Alignment (canonical) =====
print("\n=== Alignment ground truth ===")
canonical = nib.as_closest_canonical(img)
can_data = canonical.get_fdata(dtype=np.float32)
save_raw(can_data, "alignment_canonical_data")
save_raw(np.array(canonical.affine, dtype=np.float64), "alignment_canonical_affine")
orig_ornt = nib.io_orientation(img.affine)
can_ornt = nib.io_orientation(canonical.affine)
# Save orientation codes
save_raw(orig_ornt.astype(np.float64), "alignment_orig_ornt")
save_raw(can_ornt.astype(np.float64), "alignment_can_ornt")

# Undo canonical
ornt_transform = nib.orientations.ornt_transform(can_ornt, orig_ornt)
undo_data = nib.orientations.apply_orientation(can_data, ornt_transform)
save_raw(undo_data, "alignment_undo_data")

# ===== 3. Resampling =====
print("\n=== Resampling ground truth ===")
# Resample to 1.5mm spacing (from 3.0mm)
target_spacing = np.array([1.5, 1.5, 1.5])
current_spacing = spacing
zoom_factors = current_spacing / target_spacing
resampled = zoom(data, zoom_factors, order=3, mode='nearest')
save_raw(resampled, "resampling_1p5mm_data")
save_raw(np.array(resampled.shape, dtype=np.int32), "resampling_1p5mm_shape")

# Also do order=1 (trilinear) for comparison
resampled_linear = zoom(data, zoom_factors, order=1, mode='nearest')
save_raw(resampled_linear, "resampling_1p5mm_linear_data")

# Resample back to original shape (nearest neighbor, like label)
back_zoom = np.array(data.shape) / np.array(resampled.shape)
resampled_back = zoom(resampled, back_zoom, order=0, mode='nearest')
save_raw(resampled_back, "resampling_roundtrip_data")

# ===== 4. Cropping =====
print("\n=== Cropping ground truth ===")
seg = nib.load("tests/example_seg.nii.gz").get_fdata().astype(np.uint8)
# Find bounding box of non-zero
nonzero = np.nonzero(seg)
if len(nonzero[0]) > 0:
    bbox_min = [int(np.min(nonzero[i])) for i in range(3)]
    bbox_max = [int(np.max(nonzero[i])) for i in range(3)]
    # Add 3 voxels padding (default addon)
    addon = 3
    bbox_min_padded = [max(0, b - addon) for b in bbox_min]
    bbox_max_padded = [min(seg.shape[i] - 1, bbox_max[i] + addon) for i in range(3)]
    bbox = np.array([bbox_min_padded, bbox_max_padded], dtype=np.int32)
    save_raw(bbox, "cropping_bbox")
    # Crop
    cropped = seg[bbox[0,0]:bbox[1,0]+1, bbox[0,1]:bbox[1,1]+1, bbox[0,2]:bbox[1,2]+1]
    save_raw(cropped, "cropping_cropped_data")
    save_raw(np.array(cropped.shape, dtype=np.int32), "cropping_cropped_shape")

# ===== 5. Sliding Window helpers =====
print("\n=== Sliding Window ground truth ===")
# Gaussian importance map for patch size (128, 128, 128)
patch_size = (128, 128, 128)
sigma_scale = 1.0 / 8.0
tmp = np.zeros(patch_size)
center = [s // 2 for s in patch_size]
tmp[tuple(center)] = 1
sigmas = [s * sigma_scale for s in patch_size]
gauss = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
gauss /= np.max(gauss)
mask = gauss == 0
if np.any(~mask):
    gauss[mask] = np.min(gauss[~mask])
save_raw(gauss.astype(np.float32), "sliding_window_gaussian_128")

# Step computation for image_size=(244, 202, 224), patch=(128,128,128), step=0.5
image_size = (244, 202, 224)
tile_size = (128, 128, 128)
step_size = 0.5
target_steps = [t * step_size for t in tile_size]
num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_steps, tile_size)]
steps = []
for dim in range(3):
    max_step = image_size[dim] - tile_size[dim]
    if num_steps[dim] > 1:
        actual = max_step / (num_steps[dim] - 1)
    else:
        actual = 99999999999
    steps.append([int(np.round(actual * i)) for i in range(num_steps[dim])])
    
steps_flat = []
for s in steps:
    steps_flat.extend(s)
    steps_flat.append(-1)  # delimiter
save_raw(np.array(steps_flat, dtype=np.int32), "sliding_window_steps")
save_raw(np.array(num_steps, dtype=np.int32), "sliding_window_num_steps")

print("\n=== All ground truth generated ===")
print(f"Files in {OUT}:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}: {f.stat().st_size} bytes")
