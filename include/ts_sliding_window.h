#pragma once
#include "ts_types.h"
#include <functional>

namespace totalseg {

/// Callback type for running inference on a single patch.
/// Input: float tensor [1, C, Z, Y, X], Output: float tensor [1, num_classes, Z, Y, X]
using InferenceFunc = std::function<std::vector<float>(const std::vector<float>& input,
                                                        const std::array<int, 5>& input_shape)>;

/// Sliding window inference with Gaussian weighting.
/// input: preprocessed Volume (already resampled, canonical, etc.)
/// patch_size: nnUNet patch size [z, y, x]
/// step_size: overlap fraction (0.5 = 50% overlap)
/// num_classes: number of output classes
/// infer_fn: callback that runs ONNX inference on one patch
/// Returns: argmax label volume
LabelVolume sliding_window_inference(
    const Volume& input,
    const std::array<int, 3>& patch_size,
    double step_size,
    int num_classes,
    InferenceFunc infer_fn
);

/// Create 3D Gaussian importance map for patch weighting.
std::vector<float> create_gaussian_importance_map(const std::array<int, 3>& patch_size, double sigma_scale = 0.125);

} // namespace totalseg
