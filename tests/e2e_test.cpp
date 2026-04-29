#include "totalseg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <set>
#include <map>

static const std::string GT = "tests/ground_truth/";

template<typename T>
std::vector<T> load_bin(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

double dice_multilabel(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    // Per-label Dice, then average
    std::map<uint8_t, size_t> tp, a_cnt, b_cnt;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > 0) a_cnt[a[i]]++;
        if (b[i] > 0) b_cnt[b[i]]++;
        if (a[i] > 0 && a[i] == b[i]) tp[a[i]]++;
    }
    // Union of labels
    std::set<uint8_t> labels;
    for (auto& [k,v] : a_cnt) labels.insert(k);
    for (auto& [k,v] : b_cnt) labels.insert(k);
    
    if (labels.empty()) return 1.0;
    
    double sum_dice = 0;
    for (auto lbl : labels) {
        size_t t = tp.count(lbl) ? tp[lbl] : 0;
        size_t ac = a_cnt.count(lbl) ? a_cnt[lbl] : 0;
        size_t bc = b_cnt.count(lbl) ? b_cnt[lbl] : 0;
        double d = (ac + bc > 0) ? 2.0 * t / (ac + bc) : 1.0;
        std::cout << "  Label " << (int)lbl << ": Dice=" << d
                  << " (tp=" << t << " a=" << ac << " b=" << bc << ")\n";
        sum_dice += d;
    }
    return sum_dice / labels.size();
}

int main() {
    std::cout << "=== End-to-End Test: Task 291, Fold 0 ===\n\n";
    
    // 1. Load CT
    std::cout << "1. Loading NIfTI...\n";
    auto vol = totalseg::load_nifti("tests/example_ct.nii.gz");
    std::cout << "   Shape: " << vol.shape[0] << "x" << vol.shape[1] << "x" << vol.shape[2] << "\n";
    
    // 2. Canonical
    std::cout << "2. To canonical (RAS)...\n";
    totalseg::OrientCode orig_orient;
    auto canonical = totalseg::to_canonical(vol, orig_orient);
    std::cout << "   Canonical shape: " << canonical.shape[0] << "x" << canonical.shape[1] << "x" << canonical.shape[2] << "\n";
    
    // 3. Resample to 1.5mm
    std::cout << "3. Resample to 1.5mm...\n";
    auto resampled = totalseg::change_spacing(canonical, {1.5, 1.5, 1.5}, 3);
    std::cout << "   Resampled shape: " << resampled.shape[0] << "x" << resampled.shape[1] << "x" << resampled.shape[2] << "\n";
    
    // 4. Load ONNX model
    std::cout << "4. Loading ONNX model...\n";
    totalseg::OnnxModel model("models/onnx/task291_fold0.onnx", false);
    int num_classes = model.num_classes();
    std::cout << "   Num classes: " << num_classes << "\n";
    
    // 5. Sliding window inference
    std::cout << "5. Running sliding window inference...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    
    auto infer_fn = [&](const std::vector<float>& input, const std::array<int, 5>& shape) -> std::vector<float> {
        return model.run(input, shape);
    };
    
    auto seg = totalseg::sliding_window_inference(
        resampled, {128, 128, 128}, 0.5f, num_classes, infer_fn
    );
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "   Inference time: " << elapsed << "s\n";
    std::cout << "   Seg shape: " << seg.shape[0] << "x" << seg.shape[1] << "x" << seg.shape[2] << "\n";
    
    // 6. Resample back
    std::cout << "6. Resample back to original shape...\n";
    auto seg_back = totalseg::resample_to_shape_label(seg, canonical.shape);
    std::cout << "   Shape: " << seg_back.shape[0] << "x" << seg_back.shape[1] << "x" << seg_back.shape[2] << "\n";
    
    // 7. Compare with Python ground truth
    std::cout << "7. Comparing with Python ground truth...\n";
    size_t nvox = seg_back.numel();
    auto gt_seg = load_bin<uint8_t>(GT + "e2e_python_seg.bin", nvox);
    
    // Count unique labels
    std::set<uint8_t> cpp_labels, py_labels;
    for (size_t i = 0; i < nvox; ++i) {
        if (seg_back.data[i] > 0) cpp_labels.insert(seg_back.data[i]);
        if (gt_seg[i] > 0) py_labels.insert(gt_seg[i]);
    }
    std::cout << "   C++ labels: ";
    for (auto l : cpp_labels) std::cout << (int)l << " ";
    std::cout << "\n   Python labels: ";
    for (auto l : py_labels) std::cout << (int)l << " ";
    std::cout << "\n\n";
    
    double mean_dice = dice_multilabel(seg_back.data, gt_seg);
    std::cout << "\n   Mean Dice: " << mean_dice << "\n";
    std::cout << "   " << (mean_dice > 0.99 ? "PASS (>0.99)" : "FAIL (<0.99)") << "\n";
    
    return mean_dice > 0.99 ? 0 : 1;
}
