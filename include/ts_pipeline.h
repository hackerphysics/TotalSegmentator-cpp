#pragma once
#include "ts_types.h"
#include <string>
#include <vector>

namespace totalseg {

/// Configuration for a TotalSegmentator run.
struct PipelineConfig {
    std::string input_path;          // Input NIfTI path
    std::string output_path;         // Output NIfTI path
    std::string model_dir;           // Directory containing .onnx models + config.json
    std::string task = "total";      // Task name
    bool fast = false;               // Use fast (3mm) mode
    bool use_gpu = false;            // Use GPU (CUDA EP)
    double step_size = 0.5;          // Sliding window overlap
};

/// Run the full TotalSegmentator pipeline.
/// Returns the multilabel segmentation volume.
LabelVolume run_pipeline(const PipelineConfig& config);

} // namespace totalseg
