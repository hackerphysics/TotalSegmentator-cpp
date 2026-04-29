#include "totalseg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <cstring>

static const std::string GT = "tests/ground_truth/";

// Load raw binary file
template<typename T>
std::vector<T> load_bin(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

// Compute max absolute error
template<typename T>
double max_abs_error(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    double mx = 0;
    for (size_t i = 0; i < a.size(); ++i)
        mx = std::max(mx, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return mx;
}

// Compute L2 relative error
template<typename T>
double rel_l2_error(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    double sum_sq_diff = 0, sum_sq_ref = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq_diff += d * d;
        sum_sq_ref += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    }
    if (sum_sq_ref == 0) return sum_sq_diff == 0 ? 0 : 1e30;
    return std::sqrt(sum_sq_diff / sum_sq_ref);
}

// Dice for uint8 label volumes
double dice_score(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    assert(a.size() == b.size());
    size_t tp = 0, a_pos = 0, b_pos = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > 0) a_pos++;
        if (b[i] > 0) b_pos++;
        if (a[i] > 0 && a[i] == b[i]) tp++;
    }
    if (a_pos + b_pos == 0) return 1.0;
    return 2.0 * tp / (a_pos + b_pos);
}

bool test_nifti_io() {
    std::cout << "=== Test: NIfTI I/O ===\n";
    auto vol = totalseg::load_nifti("tests/example_ct.nii.gz");
    
    // Check shape
    auto gt_shape = load_bin<int32_t>(GT + "nifti_io_shape.bin", 3);
    bool shape_ok = (vol.shape[0] == gt_shape[0] && vol.shape[1] == gt_shape[1] && vol.shape[2] == gt_shape[2]);
    std::cout << "  Shape: C++=" << vol.shape[0] << "x" << vol.shape[1] << "x" << vol.shape[2]
              << " Python=" << gt_shape[0] << "x" << gt_shape[1] << "x" << gt_shape[2]
              << (shape_ok ? " PASS" : " FAIL") << "\n";
    
    // Check spacing
    auto gt_spacing = load_bin<double>(GT + "nifti_io_spacing.bin", 3);
    double sp_err = std::max({std::abs(vol.spacing[0] - gt_spacing[0]),
                              std::abs(vol.spacing[1] - gt_spacing[1]),
                              std::abs(vol.spacing[2] - gt_spacing[2])});
    std::cout << "  Spacing error: " << sp_err << (sp_err < 1e-4 ? " PASS" : " FAIL") << "\n";
    
    // Check data
    size_t nvox = vol.numel();
    auto gt_data = load_bin<float>(GT + "nifti_io_data.bin", nvox);
    double data_err = max_abs_error(vol.data, gt_data);
    double data_rel = rel_l2_error(vol.data, gt_data);
    std::cout << "  Data max_abs_error: " << data_err << " rel_l2: " << data_rel
              << (data_err < 1.0 ? " PASS" : " FAIL") << "\n";
    
    // Check affine
    auto gt_aff = load_bin<double>(GT + "nifti_io_affine.bin", 16);
    double aff_err = 0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            aff_err = std::max(aff_err, std::abs(vol.affine[r][c] - gt_aff[r * 4 + c]));
    std::cout << "  Affine max_abs_error: " << aff_err << (aff_err < 1e-4 ? " PASS" : " FAIL") << "\n";
    
    // Roundtrip test
    totalseg::save_nifti(vol, "/tmp/test_cpp_roundtrip.nii.gz");
    auto vol2 = totalseg::load_nifti("/tmp/test_cpp_roundtrip.nii.gz");
    double rt_err = max_abs_error(vol.data, vol2.data);
    std::cout << "  Roundtrip max_abs_error: " << rt_err << (rt_err < 1e-5 ? " PASS" : " FAIL") << "\n";
    
    return shape_ok && sp_err < 1e-4 && data_err < 1.0 && aff_err < 1e-4 && rt_err < 1e-5;
}

bool test_alignment() {
    std::cout << "\n=== Test: Alignment ===\n";
    auto vol = totalseg::load_nifti("tests/example_ct.nii.gz");
    
    totalseg::OrientCode orig_orient;
    auto canonical = totalseg::to_canonical(vol, orig_orient);
    
    // Check canonical data
    auto gt_data = load_bin<float>(GT + "alignment_canonical_data.bin", canonical.numel());
    double data_err = max_abs_error(canonical.data, gt_data);
    std::cout << "  Canonical data max_abs_error: " << data_err << (data_err < 1.0 ? " PASS" : " FAIL") << "\n";
    
    // Check canonical affine
    auto gt_aff = load_bin<double>(GT + "alignment_canonical_affine.bin", 16);
    double aff_err = 0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            aff_err = std::max(aff_err, std::abs(canonical.affine[r][c] - gt_aff[r * 4 + c]));
    std::cout << "  Canonical affine error: " << aff_err << (aff_err < 1e-3 ? " PASS" : " FAIL") << "\n";
    
    // Undo canonical
    auto restored = totalseg::undo_canonical(canonical, orig_orient);
    auto gt_undo = load_bin<float>(GT + "alignment_undo_data.bin", restored.numel());
    double undo_err = max_abs_error(restored.data, gt_undo);
    std::cout << "  Undo canonical max_abs_error: " << undo_err << (undo_err < 1.0 ? " PASS" : " FAIL") << "\n";
    
    return data_err < 1.0 && aff_err < 1e-3 && undo_err < 1.0;
}

bool test_resampling() {
    std::cout << "\n=== Test: Resampling ===\n";
    auto vol = totalseg::load_nifti("tests/example_ct.nii.gz");
    
    // Resample to 1.5mm with trilinear (order=1)
    auto resampled = totalseg::change_spacing(vol, {1.5, 1.5, 1.5}, 1);
    auto gt_shape = load_bin<int32_t>(GT + "resampling_1p5mm_shape.bin", 3);
    bool shape_ok = (resampled.shape[0] == gt_shape[0] && resampled.shape[1] == gt_shape[1] && resampled.shape[2] == gt_shape[2]);
    std::cout << "  Resampled shape: C++=" << resampled.shape[0] << "x" << resampled.shape[1] << "x" << resampled.shape[2]
              << " Python=" << gt_shape[0] << "x" << gt_shape[1] << "x" << gt_shape[2]
              << (shape_ok ? " PASS" : " FAIL") << "\n";
    
    if (shape_ok) {
        auto gt_data = load_bin<float>(GT + "resampling_1p5mm_linear_data.bin", resampled.numel());
        double data_err = max_abs_error(resampled.data, gt_data);
        double data_rel = rel_l2_error(resampled.data, gt_data);
        std::cout << "  Trilinear max_abs_error: " << data_err << " rel_l2: " << data_rel
                  << (data_rel < 0.05 ? " PASS" : " FAIL") << "\n";
    }
    
    return shape_ok;
}

bool test_cropping() {
    std::cout << "\n=== Test: Cropping ===\n";
    auto seg = totalseg::load_nifti_label("tests/example_seg.nii.gz");
    
    auto bbox = totalseg::get_bbox_from_mask(seg, {3, 3, 3});
    auto gt_bbox = load_bin<int32_t>(GT + "cropping_bbox.bin", 6);
    // gt_bbox is [min0,min1,min2, max0,max1,max2] — python stores as (2,3)
    bool bbox_ok = (bbox[0][0] == gt_bbox[0] && bbox[1][0] == gt_bbox[1] && bbox[2][0] == gt_bbox[2] &&
                    bbox[0][1] == gt_bbox[3] && bbox[1][1] == gt_bbox[4] && bbox[2][1] == gt_bbox[5]);
    std::cout << "  BBox: C++=[" << bbox[0][0] << ":" << bbox[0][1] << ", " << bbox[1][0] << ":" << bbox[1][1] << ", " << bbox[2][0] << ":" << bbox[2][1] << "]"
              << " Python=[" << gt_bbox[0] << ":" << gt_bbox[3] << ", " << gt_bbox[1] << ":" << gt_bbox[4] << ", " << gt_bbox[2] << ":" << gt_bbox[5] << "]"
              << (bbox_ok ? " PASS" : " FAIL") << "\n";
    
    auto cropped = totalseg::crop_to_bbox(seg, bbox);
    auto gt_cshape = load_bin<int32_t>(GT + "cropping_cropped_shape.bin", 3);
    bool cs_ok = (cropped.shape[0] == gt_cshape[0] && cropped.shape[1] == gt_cshape[1] && cropped.shape[2] == gt_cshape[2]);
    std::cout << "  Cropped shape: C++=" << cropped.shape[0] << "x" << cropped.shape[1] << "x" << cropped.shape[2]
              << " Python=" << gt_cshape[0] << "x" << gt_cshape[1] << "x" << gt_cshape[2]
              << (cs_ok ? " PASS" : " FAIL") << "\n";
    
    if (cs_ok) {
        auto gt_cdata = load_bin<uint8_t>(GT + "cropping_cropped_data.bin", cropped.numel());
        double dice = dice_score(cropped.data, gt_cdata);
        std::cout << "  Cropped data Dice: " << dice << (dice > 0.99 ? " PASS" : " FAIL") << "\n";
    }
    
    return bbox_ok && cs_ok;
}

bool test_sliding_window_helpers() {
    std::cout << "\n=== Test: Sliding Window Helpers ===\n";
    
    // Test Gaussian importance map
    auto gauss = totalseg::create_gaussian_importance_map({128, 128, 128});
    auto gt_gauss = load_bin<float>(GT + "sliding_window_gaussian_128.bin", 128*128*128);
    double gauss_err = max_abs_error(gauss, gt_gauss);
    double gauss_rel = rel_l2_error(gauss, gt_gauss);
    std::cout << "  Gaussian map max_abs_error: " << gauss_err << " rel_l2: " << gauss_rel
              << (gauss_rel < 0.05 ? " PASS" : " FAIL") << "\n";
    
    return gauss_rel < 0.05;
}

int main() {
    std::cout << "============================================\n";
    std::cout << "TotalSegmentator C++ Unit Tests vs Python GT\n";
    std::cout << "============================================\n\n";
    
    int pass = 0, fail = 0;
    
    auto run = [&](const char* name, bool (*fn)()) {
        try {
            if (fn()) { pass++; std::cout << ">>> " << name << ": PASS\n"; }
            else { fail++; std::cout << ">>> " << name << ": FAIL\n"; }
        } catch (const std::exception& e) {
            fail++;
            std::cout << ">>> " << name << ": EXCEPTION: " << e.what() << "\n";
        }
    };
    
    run("NIfTI I/O", test_nifti_io);
    run("Alignment", test_alignment);
    run("Resampling", test_resampling);
    run("Cropping", test_cropping);
    run("SlidingWindow Helpers", test_sliding_window_helpers);
    
    std::cout << "\n============================================\n";
    std::cout << "Results: " << pass << " passed, " << fail << " failed\n";
    std::cout << "============================================\n";
    return fail > 0 ? 1 : 0;
}
