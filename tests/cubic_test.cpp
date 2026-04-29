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
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

int main() {
    std::cout << "=== Cubic Resampling Test ===\n";
    
    auto vol = totalseg::load_nifti("tests/example_ct.nii.gz");
    totalseg::OrientCode orient;
    auto can = totalseg::to_canonical(vol, orient);
    
    // Resample with order=3 (cubic + prefilter)
    auto resampled = totalseg::change_spacing(can, {1.5, 1.5, 1.5}, 3);
    std::cout << "C++ shape: " << resampled.shape[0] << "x" << resampled.shape[1] << "x" << resampled.shape[2] << "\n";
    
    // Compare with Python order=3
    auto gt = load_bin<float>(GT + "resampling_1p5mm_data.bin", resampled.numel());
    
    double max_err = 0, sum_sq = 0, sum_gt_sq = 0;
    for (size_t i = 0; i < gt.size(); ++i) {
        double d = std::abs(resampled.data[i] - gt[i]);
        if (d > max_err) max_err = d;
        sum_sq += d * d;
        sum_gt_sq += (double)gt[i] * gt[i];
    }
    double rel_l2 = std::sqrt(sum_sq / (sum_gt_sq + 1e-30));
    
    std::cout << "max_abs_error: " << max_err << "\n";
    std::cout << "rel_l2: " << rel_l2 << "\n";
    std::cout << (rel_l2 < 0.001 ? "PASS" : "FAIL") << "\n";
    return 0;
}
