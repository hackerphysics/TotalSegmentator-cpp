#!/usr/bin/env python3
"""Save Python's high-res seg (before resample back) for direct comparison."""
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter, zoom
import nibabel as nib

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
GT = f"{PROJ}/tests/ground_truth"

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

# Load resampled data
data = np.fromfile(f"{GT}/py_resampled_corder.bin", dtype=np.float32).reshape(244, 202, 224)
nc = 25
patch_size = [128, 128, 128]

sess = ort.InferenceSession(f"{PROJ}/models/onnx/task291_fold0.onnx")
gauss = compute_gaussian(patch_size)
steps = compute_steps(list(data.shape), patch_size, 0.5)
print(f"Steps: {steps}")

pred_sum = np.zeros((nc, *data.shape), dtype=np.float32)
count = np.zeros(data.shape, dtype=np.float32)

for i, sx in enumerate(steps[0]):
    for sy in steps[1]:
        for sz in steps[2]:
            patch = data[sx:sx+128, sy:sy+128, sz:sz+128].copy()
            inp = patch[np.newaxis, np.newaxis, ...].astype(np.float32)
            out = sess.run(None, {"input": inp})[0][0]  # (25, 128, 128, 128)
            for c in range(nc):
                pred_sum[c, sx:sx+128, sy:sy+128, sz:sz+128] += out[c] * gauss
            count[sx:sx+128, sy:sy+128, sz:sz+128] += gauss

for c in range(nc):
    pred_sum[c] /= count

seg_hires = np.argmax(pred_sum, axis=0).astype(np.uint8)
print(f"High-res seg shape: {seg_hires.shape}")
print(f"Labels: {np.unique(seg_hires)}")

np.ascontiguousarray(seg_hires).tofile(f"{GT}/py_seg_hires.bin")
print("Saved py_seg_hires.bin")

# Also save pred_sum for the center voxel for debugging
# Save logits at a specific voxel to compare
vx, vy, vz = 122, 101, 112  # center
logits_at_center = pred_sum[:, vx, vy, vz]
print(f"Logits at ({vx},{vy},{vz}): argmax={np.argmax(logits_at_center)} val={logits_at_center[np.argmax(logits_at_center)]:.4f}")
