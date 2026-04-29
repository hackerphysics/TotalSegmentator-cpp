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
    std::cout << "=== High-Res Inference Comparison ===\n\n";

    size_t nvox = 244 * 202 * 224;
    auto py_data = load_bin<float>(GT + "py_resampled_corder.bin", nvox);

    totalseg::Volume vol;
    vol.shape = {244, 202, 224};
    vol.spacing = {1.5, 1.5, 1.5};
    vol.data = std::move(py_data);

    totalseg::OnnxModel model("models/onnx/task291_fold0.onnx", false);
    int nc = model.num_classes();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto infer_fn = [&](const std::vector<float>& input, const std::array<int,5>& shape) {
        return model.run(input, shape);
    };
    auto seg = totalseg::sliding_window_inference(vol, {128,128,128}, 0.5f, nc, infer_fn);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference: " << std::chrono::duration<double>(t1-t0).count() << "s\n";
    std::cout << "Seg shape: " << seg.shape[0] << "x" << seg.shape[1] << "x" << seg.shape[2] << "\n";

    // Compare at high-res directly (no resample back)
    auto py_hires = load_bin<uint8_t>(GT + "py_seg_hires.bin", nvox);

    size_t match = 0, total = nvox;
    std::map<uint8_t, size_t> tp_map, cpp_cnt, py_cnt;
    for (size_t i = 0; i < nvox; ++i) {
        uint8_t c = seg.data[i], p = py_hires[i];
        if (c == p) match++;
        if (c > 0) cpp_cnt[c]++;
        if (p > 0) py_cnt[p]++;
        if (c > 0 && c == p) tp_map[c]++;
    }
    std::cout << "\nVoxel accuracy: " << match << "/" << total
              << " = " << 100.0*match/total << "%\n";

    std::set<uint8_t> all_labels;
    for (auto& [k,v] : cpp_cnt) all_labels.insert(k);
    for (auto& [k,v] : py_cnt) all_labels.insert(k);

    double sum_dice = 0;
    for (auto lbl : all_labels) {
        size_t t = tp_map.count(lbl) ? tp_map[lbl] : 0;
        size_t ac = cpp_cnt.count(lbl) ? cpp_cnt[lbl] : 0;
        size_t bc = py_cnt.count(lbl) ? py_cnt[lbl] : 0;
        double d = (ac+bc>0) ? 2.0*t/(ac+bc) : 1.0;
        std::cout << "  Label " << (int)lbl << ": Dice=" << d << "\n";
        sum_dice += d;
    }
    double mean_dice = sum_dice / all_labels.size();
    std::cout << "\nMean Dice (high-res): " << mean_dice
              << (mean_dice > 0.99 ? " PASS" : " FAIL") << "\n";

    // Save C++ hires seg for further analysis
    {
        std::ofstream f(GT + "cpp_seg_hires.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(seg.data.data()), seg.data.size());
    }
    std::cout << "Saved cpp_seg_hires.bin\n";
    return 0;
}
