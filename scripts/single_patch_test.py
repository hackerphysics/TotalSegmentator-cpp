#!/usr/bin/env python3
"""Compare single-patch inference between Python ONNX and C++ ONNX.
Extract center patch, run through ONNX, save raw logits for comparison."""
import numpy as np
import onnxruntime as ort
import json

PROJ = "/home/haipw/.openclaw/workspace/totalseg-cpp"
GT = f"{PROJ}/tests/ground_truth"

# Load resampled data
data = np.fromfile(f"{GT}/py_resampled_corder.bin", dtype=np.float32).reshape(244, 202, 224)

# Extract center patch
sx, sy, sz = 58, 37, 48  # center-ish
patch = data[sx:sx+128, sy:sy+128, sz:sz+128].copy()
print(f"Patch range: [{patch.min():.1f}, {patch.max():.1f}], shape={patch.shape}")

# Save patch for C++ to load
np.ascontiguousarray(patch).tofile(f"{GT}/single_patch_input.bin")

# Run through ONNX
sess = ort.InferenceSession(f"{PROJ}/models/onnx/task291_fold0.onnx")
inp = patch[np.newaxis, np.newaxis, ...].astype(np.float32)
print(f"Input shape to ONNX: {inp.shape}")
out = sess.run(None, {"input": inp})[0]  # (1, 25, 128, 128, 128)
print(f"Output shape: {out.shape}")
logits = out[0]  # (25, 128, 128, 128)
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

# Argmax
seg = np.argmax(logits, axis=0).astype(np.uint8)
print(f"Labels in patch: {np.unique(seg)}")

# Save
np.ascontiguousarray(logits).tofile(f"{GT}/single_patch_logits_py.bin")
np.ascontiguousarray(seg).tofile(f"{GT}/single_patch_seg_py.bin")
json.dump({"sx": sx, "sy": sy, "sz": sz, "patch_size": 128,
           "logits_shape": list(logits.shape), "labels": sorted(int(x) for x in np.unique(seg))},
          open(f"{GT}/single_patch_meta.json", "w"))
print("Saved single patch ground truth")
