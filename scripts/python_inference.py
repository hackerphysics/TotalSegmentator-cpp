#!/usr/bin/env python3
"""Run TotalSegmentator Python inference on example_ct as ground truth for C++ comparison.
Uses ONNX Runtime instead of PyTorch for fair comparison."""
import os, sys, json, time
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.ndimage import gaussian_filter, zoom

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
ONNX_DIR = f"{PROJ}/models/onnx"
GT_DIR = f"{PROJ}/tests/ground_truth"

def compute_gaussian(tile_size, sigma_scale=1./8):
    tmp = np.zeros(tile_size)
    center = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center)] = 1
    g = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    g /= np.max(g)
    g[g == 0] = np.min(g[g > 0])
    return g.astype(np.float32)

def compute_steps(image_size, tile_size, step_size=0.5):
    target_steps = [t * step_size for t in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_steps, tile_size)]
    steps = []
    for dim in range(len(tile_size)):
        max_step = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual = max_step / (num_steps[dim] - 1)
        else:
            actual = 99999999999
        steps.append([int(np.round(actual * i)) for i in range(num_steps[dim])])
    return steps

def sliding_window_inference(data_3d, session, patch_size, step_size=0.5, num_classes=25):
    """data_3d: (x, y, z) float32 array."""
    # Pad to at least patch_size
    pad_widths = []
    for i in range(3):
        if data_3d.shape[i] < patch_size[i]:
            pad = patch_size[i] - data_3d.shape[i]
            pad_widths.append((pad // 2, pad - pad // 2))
        else:
            pad_widths.append((0, 0))
    padded = np.pad(data_3d, pad_widths, mode='reflect')
    
    gauss = compute_gaussian(patch_size)
    steps = compute_steps(padded.shape, patch_size, step_size)
    
    pred_sum = np.zeros((num_classes, *padded.shape), dtype=np.float32)
    count = np.zeros(padded.shape, dtype=np.float32)
    
    total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
    print(f"  Sliding window: {total_patches} patches, image={padded.shape}")
    
    for i, sx in enumerate(steps[0]):
        for sy in steps[1]:
            for sz in steps[2]:
                patch = padded[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]]
                # ONNX expects (1, 1, D, H, W) — but nnUNet convention is (1,1,z,y,x)?
                # Actually nnUNet stores as (batch, channel, z, y, x) but our data is (x,y,z)
                # We need to transpose to match: input shape should be (1,1, patch[0], patch[1], patch[2])
                inp = patch[np.newaxis, np.newaxis, ...].astype(np.float32)
                out = session.run(None, {"input": inp})[0]  # (1, C, x, y, z)
                logits = out[0]  # (C, x, y, z)
                for c in range(num_classes):
                    pred_sum[c, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += logits[c] * gauss
                count[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += gauss
    
    # Divide
    for c in range(num_classes):
        pred_sum[c] /= count
    
    # Argmax
    seg = np.argmax(pred_sum, axis=0).astype(np.uint8)
    
    # Unpad
    seg = seg[pad_widths[0][0]:seg.shape[0]-pad_widths[0][1] if pad_widths[0][1] else seg.shape[0],
              pad_widths[1][0]:seg.shape[1]-pad_widths[1][1] if pad_widths[1][1] else seg.shape[1],
              pad_widths[2][0]:seg.shape[2]-pad_widths[2][1] if pad_widths[2][1] else seg.shape[2]]
    
    return seg

# Load CT
print("Loading example_ct.nii.gz...")
img = nib.load(f"{PROJ}/tests/example_ct.nii.gz")
data = img.get_fdata(dtype=np.float32)  # (122, 101, 112)
print(f"  Shape: {data.shape}, Spacing: {img.header.get_zooms()[:3]}")

# Canonical (already RAS for this file based on our test)
canonical = nib.as_closest_canonical(img)
can_data = canonical.get_fdata(dtype=np.float32)

# Resample to 1.5mm
current_spacing = np.array(canonical.header.get_zooms()[:3])
target_spacing = np.array([1.5, 1.5, 1.5])
zoom_factors = current_spacing / target_spacing
resampled = zoom(can_data, zoom_factors, order=3, mode='nearest')
print(f"  Resampled: {resampled.shape}")

# Load ONNX model
onnx_path = f"{ONNX_DIR}/task291_fold0.onnx"
print(f"Loading ONNX model: {onnx_path}")
sess = ort.InferenceSession(onnx_path)

# Run inference
t0 = time.time()
seg_resampled = sliding_window_inference(resampled, sess, [128, 128, 128], 0.5, 25)
t1 = time.time()
print(f"  Inference time: {t1-t0:.1f}s")

# Resample back to original shape
back_zoom = np.array(can_data.shape) / np.array(seg_resampled.shape)
seg_original = zoom(seg_resampled.astype(np.float32), back_zoom, order=0, mode='nearest').astype(np.uint8)
print(f"  Final seg shape: {seg_original.shape}")
print(f"  Unique labels: {np.unique(seg_original)}")

# Save ground truth
np.save(f"{GT_DIR}/python_seg_task291.npy", seg_original)
# Also save as NIfTI for visual inspection
seg_nii = nib.Nifti1Image(seg_original, canonical.affine)
nib.save(seg_nii, f"{GT_DIR}/python_seg_task291.nii.gz")
print(f"  Saved to {GT_DIR}/python_seg_task291.npy and .nii.gz")
