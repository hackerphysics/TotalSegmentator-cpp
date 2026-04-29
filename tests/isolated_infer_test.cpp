#include "totalseg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <chrono>
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
    std::cout << "=== Isolated Inference Test ===\n"
              << "Uses Python-resampled data as input to isolate resampling vs inference diff\n\n";

    // Load Python resampled data directly
    size_t nvox = 244 * 202 * 224;
    auto py_data = load_bin<float>(GT + "py_resampled_corder.bin", nvox);
    std::cout << "Loaded Python resampled: 244x202x224, " << nvox << " voxels\n";

    // Build a Volume from it
    totalseg::Volume vol;
    vol.shape = {244, 202, 224};
    vol.spacing = {1.5, 1.5, 1.5};
    vol.data = std::move(py_data);
    // affine doesn't matter for inference

    // Load ONNX
    std::cout << "Loading ONNX model...\n";
    totalseg::OnnxModel model("models/onnx/task291_fold0.onnx", false);
    int nc = model.num_classes();
    std::cout << "Num classes: " << nc << "\n";

    // Sliding window
    auto t0 = std::chrono::high_resolution_clock::now();
    auto infer_fn = [&](const std::vector<float>& input, const std::array<int,5>& shape) {
        return model.run(input, shape);
    };
    auto seg = totalseg::sliding_window_inference(vol, {128,128,128}, 0.5f, nc, infer_fn);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference: " << std::chrono::duration<double>(t1-t0).count() << "s\n";
    std::cout << "Seg shape: " << seg.shape[0] << "x" << seg.shape[1] << "x" << seg.shape[2] << "\n";

    // Resample back to 122x101x112
    auto seg_back = totalseg::resample_to_shape_label(seg, {122, 101, 112});
    std::cout << "Resampled back: " << seg_back.shape[0] << "x" << seg_back.shape[1] << "x" << seg_back.shape[2] << "\n";

    // Compare
    size_t orig_nvox = 122 * 101 * 112;
    auto gt_seg = load_bin<uint8_t>(GT + "e2e_python_seg.bin", orig_nvox);

    std::set<uint8_t> cpp_labels, py_labels;
    for (size_t i = 0; i < orig_nvox; ++i) {
        if (seg_back.data[i] > 0) cpp_labels.insert(seg_back.data[i]);
        if (gt_seg[i] > 0) py_labels.insert(gt_seg[i]);
    }
    std::cout << "\nC++ labels: ";
    for (auto l : cpp_labels) std::cout << (int)l << " ";
    std::cout << "\nPython labels: ";
    for (auto l : py_labels) std::cout << (int)l << " ";
    std::cout << "\n\n";

    // Per-label Dice
    std::set<uint8_t> all_labels;
    for (auto l : cpp_labels) all_labels.insert(l);
    for (auto l : py_labels) all_labels.insert(l);

    double sum_dice = 0;
    for (auto lbl : all_labels) {
        size_t tp = 0, ac = 0, bc = 0;
        for (size_t i = 0; i < orig_nvox; ++i) {
            if (seg_back.data[i] == lbl) ac++;
            if (gt_seg[i] == lbl) bc++;
            if (seg_back.data[i] == lbl && gt_seg[i] == lbl) tp++;
        }
        double d = (ac + bc > 0) ? 2.0 * tp / (ac + bc) : 1.0;
        std::cout << "  Label " << (int)lbl << ": Dice=" << d << "\n";
        sum_dice += d;
    }
    double mean_dice = sum_dice / all_labels.size();
    std::cout << "\nMean Dice: " << mean_dice
              << (mean_dice > 0.99 ? " PASS" : " FAIL") << "\n";
    return 0;
}
