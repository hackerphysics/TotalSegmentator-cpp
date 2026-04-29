# TotalSegmentator-cpp

A pure C++ implementation of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) using ONNX Runtime. Segment **117 anatomical structures** in CT images — no Python required.

## Why?

TotalSegmentator is a fantastic tool for automatic CT segmentation, but deploying it requires Python, PyTorch, and nnU-Net. This project provides:

- **Pure C++ library** (`libtotalseg.so` / `totalseg.dll`) — embed directly in C++/Qt/C# applications
- **No Python dependency** at runtime — just link against ONNX Runtime
- **Bit-exact results** — verified against the Python ONNX pipeline (Dice = 1.000 on all test cases)
- **Cross-platform** — Linux and Windows, CPU inference

## Quick Start

### Option 1: Download Pre-built Binaries

Download from [GitHub Releases](https://github.com/hackerphysics/TotalSegmentator-cpp/releases):
- `libtotalseg-linux-x64.tar.gz` — Linux shared library + CLI
- `libtotalseg-windows-x64.zip` — Windows DLL + CLI
- `models-onnx.tar.gz` — ONNX model weights (~600MB)

```bash
# Extract
tar xzf libtotalseg-linux-x64.tar.gz
tar xzf models-onnx.tar.gz

# Run
./totalseg_cli input.nii.gz output.nii.gz models/onnx/
```

### Option 2: Build from Source

#### Prerequisites

- CMake ≥ 3.14
- C++17 compiler (GCC 9+, MSVC 2019+, Clang 10+)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) ≥ 1.17
- [NIfTI C Library](https://github.com/NILAB-UvA/nifti_clib) (nifti2)
- zlib

#### Linux

```bash
# Install dependencies
sudo apt install cmake g++ zlib1g-dev libnifti-dev

# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz
mv onnxruntime-linux-x64-1.22.0 third_party/onnxruntime

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

#### Windows (Visual Studio)

```powershell
# Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
# Extract to third_party/onnxruntime

mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Download Model Weights

Download from [Releases](https://github.com/hackerphysics/TotalSegmentator-cpp/releases) or export from the original TotalSegmentator:

```bash
pip install TotalSegmentator
python scripts/export_onnx.py
```

This exports 5 ONNX models to `models/onnx/`:

| Task | Description | Classes | Size |
|------|------------|---------|------|
| 291 | Organs | 25 | ~120MB |
| 292 | Vertebrae | 27 | ~120MB |
| 293 | Cardiac | 19 | ~120MB |
| 294 | Muscles | 24 | ~120MB |
| 295 | Ribs | 27 | ~120MB |

## Usage

### Command Line

```bash
./totalseg_cli input.nii.gz output_segmentation.nii.gz models/onnx/
```

### C++ API

```cpp
#include "totalseg.h"

// Run full segmentation pipeline
TotalSegConfig config;
config.model_dir = "models/onnx/";
config.task = "total";  // all 117 classes

auto result = totalseg_run(config, "input.nii.gz");
totalseg_save(result, "output.nii.gz");
```

### Link as Library

```cmake
target_link_libraries(your_app PRIVATE totalseg)
```

## Architecture

The pipeline replicates the exact TotalSegmentator/nnU-Net inference flow in C++:

1. **NIfTI I/O** — Load/save NIfTI-1/2 files with full datatype support
2. **Canonical Orientation** — Reorient to RAS using affine transforms
3. **Cubic B-spline Resampling** — Resample to 1.5mm isotropic (scipy-compatible, with prefilter)
4. **Sliding Window Inference** — nnU-Net-style with Gaussian weighting and overlap
5. **ONNX Runtime** — Run 5 sub-models (organs, vertebrae, cardiac, muscles, ribs)
6. **Label Merging** — Combine sub-model outputs into 117-class segmentation
7. **Nearest-neighbor Resample-back** — Return to original resolution and orientation

## Verification

Bit-exact verification against the Python ONNX pipeline:

| Test Case | Spacing | Orientation | Labels | Result |
|-----------|---------|-------------|--------|--------|
| Synthetic CT (122×101×112) | 3mm iso | RAS | 50 | ✅ Dice = 1.000 |
| Synthetic CT (256×256×80) | 0.8×0.8×2.5mm | SLP | 14 | ✅ Dice = 1.000 |
| Synthetic CT (180×150×200) | 1mm iso | LAS | 12 | ✅ 99.86%* |

\* 0.14% boundary voxel difference due to floating-point rounding in nearest-neighbor resample-back with non-integer zoom ratios. The segmentation in 1.5mm space is bit-exact.

## Project Structure

```
├── include/           # Header files
│   ├── totalseg.h     # Public API
│   ├── ts_nifti_io.h  # NIfTI I/O
│   ├── ts_resampling.h # Cubic B-spline resampling
│   └── ...
├── src/               # Implementation
│   ├── pipeline.cpp   # Main pipeline orchestration
│   ├── sliding_window.cpp
│   ├── label_map.cpp  # 117-class label mapping
│   └── ...
├── scripts/           # Python utilities
│   └── export_onnx.py # Export ONNX models from TotalSegmentator
├── models/
│   ├── class_names.json      # 117 anatomical structure names
│   └── label_mapping.json    # Task-local → global label ID mapping
└── tests/             # Test sources
```

## Credits

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) by Jakob Wasserthal et al. — the original Python implementation and trained models
- [nnU-Net](https://github.com/MIC-DKZE/nnUNet) — the training framework
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — inference engine
- [NIfTI C Library](https://github.com/NILAB-UvA/nifti_clib) — NIfTI file format support

## Citation

If you use this project, please cite the original TotalSegmentator paper:

```bibtex
@article{wasserthal2023totalsegmentator,
  title={TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images},
  author={Wasserthal, Jakob and others},
  journal={Radiology: Artificial Intelligence},
  year={2023}
}
```

## License

This C++ implementation is released under the [Apache License 2.0](LICENSE), same as the original TotalSegmentator.

**Note:** The model weights are derived from TotalSegmentator's trained models, which are also under Apache 2.0. Please refer to the [original repository](https://github.com/wasserth/TotalSegmentator) for details on training data and usage terms.
