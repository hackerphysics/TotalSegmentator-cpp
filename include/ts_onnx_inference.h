#pragma once
#include "ts_types.h"
#include <string>
#include <vector>
#include <memory>

namespace totalseg {

/// ONNX Runtime inference session wrapper.
class OnnxModel {
public:
    /// Load an ONNX model from file.
    explicit OnnxModel(const std::string& model_path, bool use_gpu = false);
    ~OnnxModel();

    /// Run inference on a single patch.
    /// input shape: [1, C, Z, Y, X], output shape: [1, num_classes, Z, Y, X]
    std::vector<float> run(const std::vector<float>& input, const std::array<int, 5>& input_shape);

    /// Get number of output classes from model metadata or output shape.
    int num_classes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace totalseg
