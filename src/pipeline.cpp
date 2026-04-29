#include "ts_pipeline.h"
#include "ts_nifti_io.h"
#include "ts_alignment.h"
#include "ts_resampling.h"
#include "ts_sliding_window.h"
#include "ts_onnx_inference.h"
#include "ts_label_map.h"
#include <iostream>
#include <filesystem>

namespace totalseg {

// nnUNet patch sizes per task (Z, Y, X) — from TotalSegmentator model configs.
static std::array<int, 3> get_patch_size(int task_id, bool fast) {
    // All "total" subtasks use the same nnUNet patch size.
    if (fast) {
        return {48, 48, 48};
    }
    return {128, 128, 128};
}

static std::string model_filename(int task_id) {
    return "task" + std::to_string(task_id) + "_fold0.onnx";
}

LabelVolume run_pipeline(const PipelineConfig& config) {
    namespace fs = std::filesystem;

    std::cerr << "[totalseg] Loading input: " << config.input_path << "\n";
    Volume input = load_nifti(config.input_path);

    // 1. Convert to canonical (RAS) orientation
    OrientCode original_orient;
    Volume canonical = to_canonical(input, original_orient);
    std::cerr << "[totalseg] Canonical orientation applied.\n";

    // 2. Resample to target spacing
    double target_sp = config.fast ? 3.0 : 1.5;
    std::array<double, 3> target_spacing = {target_sp, target_sp, target_sp};
    Volume resampled = change_spacing(canonical, target_spacing, 3);
    std::cerr << "[totalseg] Resampled to " << target_sp << "mm spacing, shape: ["
              << resampled.shape[0] << ", " << resampled.shape[1] << ", "
              << resampled.shape[2] << "]\n";

    // 3. Run each sub-task
    auto task_ids = get_task_ids(config.task);
    std::vector<LabelVolume> task_results;
    task_results.reserve(task_ids.size());

    for (int tid : task_ids) {
        std::string part = get_part_name(tid);
        std::string model_path = (fs::path(config.model_dir) / model_filename(tid)).string();
        std::cerr << "[totalseg] Running task " << tid << " (" << part << ") ...\n";

        OnnxModel model(model_path, config.use_gpu);

        auto patch_size = get_patch_size(tid, config.fast);
        int nclasses = model.num_classes();
        if (nclasses < 0) {
            // Fallback: per-task class count (bg + fg classes)
            nclasses = get_task_num_classes(tid);
        }

        // Wrap model.run as InferenceFunc
        InferenceFunc infer = [&model](const std::vector<float>& patch,
                                        const std::array<int, 5>& shape) -> std::vector<float> {
            return model.run(patch, shape);
        };

        LabelVolume seg = sliding_window_inference(
            resampled, patch_size, config.step_size, nclasses, infer);

        task_results.push_back(std::move(seg));
        std::cerr << "[totalseg] Task " << tid << " done.\n";
    }

    // 4. Merge multilabel
    std::cerr << "[totalseg] Merging " << task_results.size() << " sub-task results...\n";
    LabelVolume merged = merge_multilabel(task_results, task_ids, config.task);

    // 5. Resample back to original spacing (nearest neighbor)
    LabelVolume resampled_back = resample_to_shape_label(merged, canonical.shape);
    resampled_back.spacing = canonical.spacing;
    resampled_back.affine = canonical.affine;

    // 6. Undo canonical orientation
    LabelVolume final_seg = undo_canonical(resampled_back, original_orient);
    final_seg.affine = input.affine;
    final_seg.spacing = input.spacing;

    // 7. Save output
    std::cerr << "[totalseg] Saving output: " << config.output_path << "\n";
    save_nifti_label(final_seg, config.output_path);

    return final_seg;
}

} // namespace totalseg
