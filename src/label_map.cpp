#include "ts_label_map.h"
#include <stdexcept>
#include <algorithm>

namespace totalseg {

// ── Per-task class maps ──────────────────────────────────────────────
// Index 0 = background (implicit), labels start at 1.
// The map returns local_label -> name.

static LabelMap task291_organs() {
    return {
        {1, "spleen"}, {2, "kidney_right"}, {3, "kidney_left"},
        {4, "gallbladder"}, {5, "liver"}, {6, "stomach"},
        {7, "pancreas"}, {8, "adrenal_gland_right"}, {9, "adrenal_gland_left"},
        {10, "lung_upper_lobe_left"}, {11, "lung_lower_lobe_left"},
        {12, "lung_upper_lobe_right"}, {13, "lung_middle_lobe_right"},
        {14, "lung_lower_lobe_right"}, {15, "esophagus"}, {16, "trachea"},
        {17, "thyroid_gland"}, {18, "small_bowel"}, {19, "duodenum"},
        {20, "colon"}, {21, "urinary_bladder"}, {22, "prostate"},
        {23, "kidney_cyst_left"}, {24, "kidney_cyst_right"}
    };
}

static LabelMap task292_vertebrae() {
    return {
        {1, "sacrum"}, {2, "vertebrae_S1"},
        {3, "vertebrae_L5"}, {4, "vertebrae_L4"}, {5, "vertebrae_L3"},
        {6, "vertebrae_L2"}, {7, "vertebrae_L1"},
        {8, "vertebrae_T12"}, {9, "vertebrae_T11"}, {10, "vertebrae_T10"},
        {11, "vertebrae_T9"}, {12, "vertebrae_T8"}, {13, "vertebrae_T7"},
        {14, "vertebrae_T6"}, {15, "vertebrae_T5"}, {16, "vertebrae_T4"},
        {17, "vertebrae_T3"}, {18, "vertebrae_T2"}, {19, "vertebrae_T1"},
        {20, "vertebrae_C7"}, {21, "vertebrae_C6"}, {22, "vertebrae_C5"},
        {23, "vertebrae_C4"}, {24, "vertebrae_C3"}, {25, "vertebrae_C2"},
        {26, "vertebrae_C1"}
    };
}

static LabelMap task293_cardiac() {
    return {
        {1, "heart"}, {2, "aorta"}, {3, "pulmonary_vein"},
        {4, "brachiocephalic_trunk"}, {5, "subclavian_artery_right"},
        {6, "subclavian_artery_left"}, {7, "common_carotid_artery_right"},
        {8, "common_carotid_artery_left"}, {9, "brachiocephalic_vein_left"},
        {10, "brachiocephalic_vein_right"}, {11, "atrial_appendage_left"},
        {12, "superior_vena_cava"}, {13, "inferior_vena_cava"},
        {14, "portal_vein_and_splenic_vein"}, {15, "iliac_artery_left"},
        {16, "iliac_artery_right"}, {17, "iliac_vena_left"},
        {18, "iliac_vena_right"}
    };
}

static LabelMap task294_muscles() {
    return {
        {1, "humerus_left"}, {2, "humerus_right"}, {3, "scapula_left"},
        {4, "scapula_right"}, {5, "clavicula_left"}, {6, "clavicula_right"},
        {7, "femur_left"}, {8, "femur_right"}, {9, "hip_left"},
        {10, "hip_right"}, {11, "spinal_cord"},
        {12, "gluteus_maximus_left"}, {13, "gluteus_maximus_right"},
        {14, "gluteus_medius_left"}, {15, "gluteus_medius_right"},
        {16, "gluteus_minimus_left"}, {17, "gluteus_minimus_right"},
        {18, "autochthon_left"}, {19, "autochthon_right"},
        {20, "iliopsoas_left"}, {21, "iliopsoas_right"},
        {22, "brain"}, {23, "skull"}
    };
}

static LabelMap task295_ribs() {
    return {
        {1, "rib_left_1"}, {2, "rib_left_2"}, {3, "rib_left_3"},
        {4, "rib_left_4"}, {5, "rib_left_5"}, {6, "rib_left_6"},
        {7, "rib_left_7"}, {8, "rib_left_8"}, {9, "rib_left_9"},
        {10, "rib_left_10"}, {11, "rib_left_11"}, {12, "rib_left_12"},
        {13, "rib_right_1"}, {14, "rib_right_2"}, {15, "rib_right_3"},
        {16, "rib_right_4"}, {17, "rib_right_5"}, {18, "rib_right_6"},
        {19, "rib_right_7"}, {20, "rib_right_8"}, {21, "rib_right_9"},
        {22, "rib_right_10"}, {23, "rib_right_11"}, {24, "rib_right_12"},
        {25, "sternum"}, {26, "costal_cartilages"}
    };
}

// ── Task-id helpers ──────────────────────────────────────────────────

static LabelMap task_local_map(int task_id) {
    switch (task_id) {
        case 291: return task291_organs();
        case 292: return task292_vertebrae();
        case 293: return task293_cardiac();
        case 294: return task294_muscles();
        case 295: return task295_ribs();
        default:
            throw std::runtime_error("Unknown task_id: " + std::to_string(task_id));
    }
}

std::vector<int> get_task_ids(const std::string& task_name) {
    if (task_name == "total") {
        return {291, 292, 293, 294, 295};
    }
    throw std::runtime_error("Unknown task: " + task_name);
}

std::string get_part_name(int task_id) {
    switch (task_id) {
        case 291: return "organs";
        case 292: return "vertebrae";
        case 293: return "cardiac";
        case 294: return "muscles";
        case 295: return "ribs";
        default: return "unknown";
    }
}

int get_task_num_classes(int task_id) {
    // Returns total output channels (including background channel 0)
    switch (task_id) {
        case 291: return 25;  // 24 organs + bg
        case 292: return 27;  // 26 vertebrae + bg
        case 293: return 19;  // 18 cardiac + bg
        case 294: return 24;  // 23 muscles + bg
        case 295: return 27;  // 26 ribs + bg
        default:
            throw std::runtime_error("Unknown task_id: " + std::to_string(task_id));
    }
}

// ── Global class map ─────────────────────────────────────────────────
// "total" task: all 117 classes, unique global IDs 1..117.

LabelMap get_class_map(const std::string& task_name) {
    if (task_name != "total") {
        throw std::runtime_error("get_class_map: unsupported task " + task_name);
    }
    LabelMap global;
    uint8_t gid = 1;
    for (int tid : {291, 292, 293, 294, 295}) {
        auto local = task_local_map(tid);
        for (uint8_t li = 1; li <= static_cast<uint8_t>(local.size()); ++li) {
            global[gid++] = local.at(li);
        }
    }
    return global;
}

// ── Build offset table (task_id -> global_start) ─────────────────────

static std::map<int, uint8_t> build_global_offsets() {
    std::map<int, uint8_t> offsets;
    uint8_t gid = 0; // offset before first label
    for (int tid : {291, 292, 293, 294, 295}) {
        offsets[tid] = gid;
        gid += static_cast<uint8_t>(task_local_map(tid).size());
    }
    return offsets;
}

// ── Merge multilabel ─────────────────────────────────────────────────

LabelVolume merge_multilabel(const std::vector<LabelVolume>& parts,
                              const std::vector<int>& task_ids,
                              const std::string& task_name) {
    if (parts.empty()) {
        throw std::runtime_error("merge_multilabel: no parts to merge");
    }
    if (parts.size() != task_ids.size()) {
        throw std::runtime_error("merge_multilabel: parts/task_ids size mismatch");
    }

    auto offsets = build_global_offsets();

    // Use the first part as shape/affine reference
    LabelVolume merged;
    merged.shape = parts[0].shape;
    merged.spacing = parts[0].spacing;
    merged.affine = parts[0].affine;
    merged.data.assign(merged.numel(), 0);

    for (size_t p = 0; p < parts.size(); ++p) {
        int tid = task_ids[p];
        uint8_t offset = offsets.at(tid);
        const auto& vol = parts[p];

        for (size_t i = 0; i < vol.data.size(); ++i) {
            uint8_t local_label = vol.data[i];
            if (local_label == 0) continue;
            // Only overwrite if merged is still background (first-writer wins)
            if (merged.data[i] == 0) {
                merged.data[i] = offset + local_label;
            }
        }
    }

    return merged;
}

} // namespace totalseg
