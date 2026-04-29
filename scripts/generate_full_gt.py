#!/usr/bin/env python3
"""Generate full-pipeline Python ground truth: run all 5 tasks, merge labels."""
import numpy as np
import nibabel as nib
from pathlib import Path
import json, sys, time

def run_full_pipeline(input_path, output_path="tests/ground_truth/e2e_full_python_seg.bin"):
    """Run full TotalSegmentator pipeline in Python, save merged label volume."""
    from totalsegmentator.python_api import totalsegmentator
    
    # Run TotalSegmentator (all 5 tasks)
    print(f"Running TotalSegmentator on {input_path}...")
    t0 = time.time()
    seg_img = totalsegmentator(input_path, None, ml=True, fast=False, verbose=True)
    t1 = time.time()
    print(f"Total time: {t1-t0:.1f}s")
    
    seg = seg_img.get_fdata().astype(np.uint8)
    print(f"Output shape: {seg.shape}")
    
    unique = np.unique(seg)
    print(f"Unique labels: {len(unique)} (max={unique.max()})")
    
    # Save as raw binary (C-order)
    seg_c = np.ascontiguousarray(seg)
    Path(output_path).write_bytes(seg_c.tobytes())
    
    # Save metadata
    meta = {
        'shape': list(seg.shape),
        'dtype': 'uint8',
        'unique_labels': unique.tolist(),
        'affine': seg_img.affine.tolist(),
    }
    Path(output_path.replace('.bin', '.json')).write_text(json.dumps(meta, indent=2))
    print(f"Saved to {output_path}")
    
    return seg

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/example_ct.nii.gz"
    run_full_pipeline(input_path)
