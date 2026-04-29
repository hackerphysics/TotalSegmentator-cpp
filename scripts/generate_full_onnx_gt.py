#!/usr/bin/env python3
"""Generate full 5-task Python ground truth using ONNX Runtime.
Uses the exact same sliding window logic as python_inference.py (verified against C++).
Memory-efficient: process one task at a time."""
import gc
import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path
import json, sys, time

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
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
    """Exact copy of python_inference.py logic."""
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
    
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                patch = padded[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]]
                inp = patch[np.newaxis, np.newaxis, ...].astype(np.float32)
                out = session.run(None, {"input": inp})[0]
                logits = out[0]
                for c in range(num_classes):
                    pred_sum[c, sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += logits[c] * gauss
                count[sx:sx+patch_size[0], sy:sy+patch_size[1], sz:sz+patch_size[2]] += gauss
    
    for c in range(num_classes):
        pred_sum[c] /= count
    
    seg = np.argmax(pred_sum, axis=0).astype(np.uint8)
    e0 = seg.shape[0] - pad_widths[0][1] if pad_widths[0][1] else seg.shape[0]
    e1 = seg.shape[1] - pad_widths[1][1] if pad_widths[1][1] else seg.shape[1]
    e2 = seg.shape[2] - pad_widths[2][1] if pad_widths[2][1] else seg.shape[2]
    seg = seg[pad_widths[0][0]:e0, pad_widths[1][0]:e1, pad_widths[2][0]:e2]
    return seg

def main(input_path):
    TASKS = [
        (291, 25, 24),
        (292, 27, 26),
        (293, 19, 18),
        (294, 24, 23),
        (295, 27, 26),
    ]
    
    print(f"Loading {input_path}...")
    img = nib.load(input_path)
    canonical = nib.as_closest_canonical(img)
    can_data = canonical.get_fdata(dtype=np.float32)
    
    current_spacing = np.array(canonical.header.get_zooms()[:3])
    target_spacing = np.array([1.5, 1.5, 1.5])
    zoom_factors = current_spacing / target_spacing
    resampled = zoom(can_data, zoom_factors, order=3, mode='nearest')
    print(f"Resampled: {resampled.shape}")
    
    seg_combined = np.zeros(resampled.shape, dtype=np.uint8)
    offset = 0
    
    for tid, nclasses, nfg in TASKS:
        onnx_path = f"{PROJ}/models/onnx/task{tid}_fold0.onnx"
        print(f"Task {tid} ({nclasses} classes)...")
        sess = ort.InferenceSession(onnx_path)
        
        t0 = time.time()
        seg = sliding_window_inference(resampled, sess, [128,128,128], 0.5, nclasses)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s, local labels: {sorted(set(np.unique(seg))-{0})}")
        
        # Map and merge (first-writer wins)
        for local_label in range(1, nfg+1):
            global_label = offset + local_label
            mask = (seg == local_label) & (seg_combined == 0)
            seg_combined[mask] = global_label
        
        offset += nfg
        del sess, seg
        gc.collect()
    
    print(f"Combined: {len(np.unique(seg_combined))} unique, max={seg_combined.max()}")
    
    # Resample back
    back_zoom = np.array(can_data.shape) / np.array(seg_combined.shape)
    seg_original = zoom(seg_combined.astype(np.float32), back_zoom, order=0, mode='nearest').astype(np.uint8)
    print(f"Final: {seg_original.shape}")
    
    out_path = f"{GT_DIR}/e2e_full_onnx_seg.bin"
    np.ascontiguousarray(seg_original).tofile(out_path)
    meta = {'shape': list(seg_original.shape), 'dtype': 'uint8',
            'unique_labels': [int(x) for x in np.unique(seg_original)]}
    Path(out_path.replace('.bin', '.json')).write_text(json.dumps(meta, indent=2))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{PROJ}/tests/example_ct.nii.gz")
