#include "totalseg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

static const std::string GT = "tests/ground_truth/";

template<typename T>
std::vector<T> load_bin(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

int main() {
    std::cout << "=== Single Patch Inference Test ===\n\n";

    // Load patch
    size_t pvox = 128*128*128;
    auto patch_data = load_bin<float>(GT + "single_patch_input.bin", pvox);
    std::cout << "Loaded patch: 128^3 = " << pvox << " voxels\n";
    
    // Find range
    float mn = patch_data[0], mx = patch_data[0];
    for (auto v : patch_data) { if (v < mn) mn = v; if (v > mx) mx = v; }
    std::cout << "Range: [" << mn << ", " << mx << "]\n";

    // Load ONNX
    totalseg::OnnxModel model("models/onnx/task291_fold0.onnx", false);
    int nc = model.num_classes();
    std::cout << "Num classes: " << nc << "\n";

    // Run inference - input shape (1, 1, 128, 128, 128)
    std::array<int,5> shape = {1, 1, 128, 128, 128};
    auto logits_flat = model.run(patch_data, shape);
    std::cout << "Output size: " << logits_flat.size()
              << " (expected " << nc * pvox << ")\n";

    // Compare with Python logits
    auto py_logits = load_bin<float>(GT + "single_patch_logits_py.bin", nc * pvox);
    
    double max_err = 0;
    double sum_sq = 0;
    for (size_t i = 0; i < logits_flat.size() && i < py_logits.size(); ++i) {
        double diff = std::abs((double)logits_flat[i] - (double)py_logits[i]);
        if (diff > max_err) max_err = diff;
        sum_sq += diff * diff;
    }
    double rmse = std::sqrt(sum_sq / logits_flat.size());
    std::cout << "Logits comparison: max_err=" << max_err << " rmse=" << rmse << "\n";

    // Argmax comparison
    size_t mismatch = 0;
    auto py_seg = load_bin<uint8_t>(GT + "single_patch_seg_py.bin", pvox);
    for (size_t i = 0; i < pvox; ++i) {
        // Find argmax for voxel i
        int best = 0;
        float best_val = logits_flat[i];
        for (int c = 1; c < nc; ++c) {
            float val = logits_flat[c * pvox + i];
            if (val > best_val) { best_val = val; best = c; }
        }
        if ((uint8_t)best != py_seg[i]) mismatch++;
    }
    std::cout << "Argmax mismatches: " << mismatch << " / " << pvox
              << " (" << 100.0 * mismatch / pvox << "%)\n";

    return 0;
}
