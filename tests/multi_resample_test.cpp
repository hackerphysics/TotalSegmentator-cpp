// Multi-case resampling validation test
// Tests C++ preprocessing (load + canonical + cubic resample) against Python ground truth
#include "ts_nifti_io.h"
#include "ts_alignment.h"
#include "ts_resampling.h"
#include "ts_types.h"
#include <fstream>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>
#include <cstring>

struct TestResult {
    std::string name;
    std::array<int,3> orig_shape, can_shape, res_shape, gt_shape;
    double rel_l2;
    double max_abs;
    size_t total_voxels;
    size_t nonzero_diffs;
    bool pass;
};

int main() {
    std::string base = "tests/multi_test_data/";
    // Read manifest to get case list
    std::vector<std::string> cases = {
        "case1_small_iso", "case2_medium_aniso", "case3_large_iso",
        "case4_las_orient", "case5_lps_orient", "case6_tiny",
        "case7_non_cubic", "case8_already_1p5"
    };
    
    // Also test the original example_ct
    std::vector<TestResult> results;
    int pass_count = 0;
    
    // Original example_ct (already validated, include for completeness)
    {
        printf("=== Original: example_ct.nii.gz ===\n");
        auto input = totalseg::load_nifti("tests/example_ct.nii.gz");
        totalseg::OrientCode orig_orient;
        auto canonical = totalseg::to_canonical(input, orig_orient);
        std::array<double,3> sp15 = {1.5, 1.5, 1.5};
        auto resampled = totalseg::change_spacing(canonical, sp15, 3);
        
        // Load ground truth
        std::ifstream gt_file("tests/ground_truth/resampling_1p5mm_data.bin", std::ios::binary);
        std::vector<float> gt(resampled.data.size());
        gt_file.read(reinterpret_cast<char*>(gt.data()), gt.size() * 4);
        
        double sum_sq_diff = 0, sum_sq_gt = 0, max_abs = 0;
        size_t nz = 0;
        for (size_t i = 0; i < gt.size(); ++i) {
            double d = resampled.data[i] - gt[i];
            sum_sq_diff += d * d;
            sum_sq_gt += (double)gt[i] * gt[i];
            double ad = std::abs(d);
            if (ad > max_abs) max_abs = ad;
            if (ad > 0) nz++;
        }
        double rel_l2 = std::sqrt(sum_sq_diff / sum_sq_gt);
        bool pass = rel_l2 < 1e-6;
        printf("  Shape: %dx%dx%d  rel_l2=%.12f  max_abs=%.8f  %s\n",
               resampled.shape[0], resampled.shape[1], resampled.shape[2],
               rel_l2, max_abs, pass ? "PASS" : "FAIL");
        
        TestResult r;
        r.name = "example_ct";
        r.res_shape = resampled.shape;
        r.rel_l2 = rel_l2;
        r.max_abs = max_abs;
        r.total_voxels = gt.size();
        r.nonzero_diffs = nz;
        r.pass = pass;
        results.push_back(r);
        if (pass) pass_count++;
    }
    
    // Multi-test cases
    for (const auto& name : cases) {
        printf("\n=== %s ===\n", name.c_str());
        std::string nii_path = base + name + ".nii.gz";
        std::string gt_path = base + name + "_resampled.bin";
        
        auto input = totalseg::load_nifti(nii_path);
        printf("  Original: %dx%dx%d\n", input.shape[0], input.shape[1], input.shape[2]);
        
        totalseg::OrientCode orig_orient;
        auto canonical = totalseg::to_canonical(input, orig_orient);
        printf("  Canonical: %dx%dx%d\n", canonical.shape[0], canonical.shape[1], canonical.shape[2]);
        
        std::array<double,3> sp15 = {1.5, 1.5, 1.5};
        auto resampled = totalseg::change_spacing(canonical, sp15, 3);
        printf("  Resampled: %dx%dx%d\n", resampled.shape[0], resampled.shape[1], resampled.shape[2]);
        
        // Load Python ground truth
        std::ifstream gt_file(gt_path, std::ios::binary);
        if (!gt_file.good()) {
            printf("  ERROR: cannot open %s\n", gt_path.c_str());
            continue;
        }
        gt_file.seekg(0, std::ios::end);
        size_t gt_bytes = gt_file.tellg();
        gt_file.seekg(0);
        size_t gt_count = gt_bytes / 4;
        std::vector<float> gt(gt_count);
        gt_file.read(reinterpret_cast<char*>(gt.data()), gt_bytes);
        
        if (gt_count != resampled.data.size()) {
            printf("  SIZE MISMATCH: cpp=%zu gt=%zu\n", resampled.data.size(), gt_count);
            TestResult r;
            r.name = name;
            r.res_shape = resampled.shape;
            r.rel_l2 = -1;
            r.max_abs = -1;
            r.total_voxels = gt_count;
            r.nonzero_diffs = gt_count;
            r.pass = false;
            results.push_back(r);
            continue;
        }
        
        double sum_sq_diff = 0, sum_sq_gt = 0, max_abs = 0;
        size_t nz = 0;
        for (size_t i = 0; i < gt.size(); ++i) {
            double d = resampled.data[i] - gt[i];
            sum_sq_diff += d * d;
            sum_sq_gt += (double)gt[i] * gt[i];
            double ad = std::abs(d);
            if (ad > max_abs) max_abs = ad;
            if (ad > 0) nz++;
        }
        double rel_l2 = (sum_sq_gt > 0) ? std::sqrt(sum_sq_diff / sum_sq_gt) : (sum_sq_diff > 0 ? 1.0 : 0.0);
        bool pass = rel_l2 < 1e-6;
        printf("  rel_l2=%.12f  max_abs=%.8f  nonzero=%zu/%zu  %s\n",
               rel_l2, max_abs, nz, gt.size(), pass ? "PASS" : "FAIL");
        
        TestResult r;
        r.name = name;
        r.res_shape = resampled.shape;
        r.rel_l2 = rel_l2;
        r.max_abs = max_abs;
        r.total_voxels = gt.size();
        r.nonzero_diffs = nz;
        r.pass = pass;
        results.push_back(r);
        if (pass) pass_count++;
    }
    
    printf("\n========================================\n");
    printf("SUMMARY: %d/%zu PASSED\n", pass_count, results.size());
    printf("========================================\n");
    for (const auto& r : results) {
        printf("  %-25s  shape=%3dx%3dx%3d  rel_l2=%.2e  max_abs=%.2e  %s\n",
               r.name.c_str(), r.res_shape[0], r.res_shape[1], r.res_shape[2],
               r.rel_l2, r.max_abs, r.pass ? "PASS" : "FAIL");
    }
    
    return (pass_count == (int)results.size()) ? 0 : 1;
}
