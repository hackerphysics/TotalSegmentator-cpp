#include "ts_onnx_inference.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <stdexcept>

namespace totalseg {

struct OnnxModel::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string input_name;
    std::string output_name;
    int num_output_classes;

    Impl(const std::string& model_path, bool use_gpu)
        : env(ORT_LOGGING_LEVEL_WARNING, "totalseg"),
          session(nullptr) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
#ifdef USE_CUDA
            try {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cerr << "[totalseg] Using CUDA execution provider.\n";
            } catch (const Ort::Exception& e) {
                std::cerr << "[totalseg] CUDA EP failed (" << e.what() << "), falling back to CPU.\n";
            }
#else
            std::cerr << "[totalseg] GPU requested but CUDA support not compiled in. Using CPU.\n";
#endif
        }

        session = Ort::Session(env, model_path.c_str(), opts);

        // Get input name
        auto in_name = session.GetInputNameAllocated(0, allocator);
        input_name = in_name.get();

        // Get output name
        auto out_name = session.GetOutputNameAllocated(0, allocator);
        output_name = out_name.get();

        // Determine num_classes from output shape[1]
        auto out_info = session.GetOutputTypeInfo(0);
        auto tensor_info = out_info.GetTensorTypeAndShapeInfo();
        auto out_shape = tensor_info.GetShape();
        // Output shape: [1, num_classes, Z, Y, X]
        if (out_shape.size() >= 2 && out_shape[1] > 0) {
            num_output_classes = static_cast<int>(out_shape[1]);
        } else {
            // Dynamic shape — will be determined at runtime
            num_output_classes = -1;
        }
    }
};

OnnxModel::OnnxModel(const std::string& model_path, bool use_gpu)
    : impl_(std::make_unique<Impl>(model_path, use_gpu)) {}

OnnxModel::~OnnxModel() = default;

int OnnxModel::num_classes() const {
    return impl_->num_output_classes;
}

std::vector<float> OnnxModel::run(const std::vector<float>& input,
                                   const std::array<int, 5>& input_shape) {
    auto& sess = impl_->session;
    auto& allocator = impl_->allocator;

    // Build input tensor
    std::array<int64_t, 5> shape_i64 = {
        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]
    };
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(input.data()), input.size(),
        shape_i64.data(), shape_i64.size());

    const char* input_names[] = {impl_->input_name.c_str()};
    const char* output_names[] = {impl_->output_name.c_str()};

    auto output_tensors = sess.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    // Copy output
    auto& out = output_tensors[0];
    auto out_info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = out_info.GetShape();
    size_t out_count = 1;
    for (auto d : out_shape) out_count *= static_cast<size_t>(d);

    const float* out_data = out.GetTensorData<float>();

    // Update num_classes if it was dynamic
    if (impl_->num_output_classes < 0 && out_shape.size() >= 2) {
        impl_->num_output_classes = static_cast<int>(out_shape[1]);
    }

    return std::vector<float>(out_data, out_data + out_count);
}

} // namespace totalseg
