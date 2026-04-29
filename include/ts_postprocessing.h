#pragma once
#include "ts_types.h"

namespace totalseg {

/// Keep only the largest connected component for each label.
LabelVolume keep_largest_blob_multilabel(const LabelVolume& input, const std::vector<uint8_t>& label_ids);

/// Remove connected components smaller than min_size voxels.
LabelVolume remove_small_blobs_multilabel(const LabelVolume& input, const std::vector<uint8_t>& label_ids,
                                           int min_size = 10, int max_size = -1);

/// Set all voxels outside mask to 0.
LabelVolume remove_outside_of_mask(const LabelVolume& seg, const LabelVolume& mask, int dilation_iters = 1);

/// Remove auxiliary training labels.
LabelVolume remove_auxiliary_labels(const LabelVolume& input, const std::vector<uint8_t>& aux_label_ids);

} // namespace totalseg
