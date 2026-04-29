#pragma once
#include "ts_types.h"

namespace totalseg {

/// Compute bounding box of non-zero region in a label volume.
BBox get_bbox_from_mask(const LabelVolume& mask, const std::array<int, 3>& addon_voxels = {0, 0, 0});

/// Crop a float volume to a bounding box. Updates affine accordingly.
Volume crop_to_bbox(const Volume& input, const BBox& bbox);

/// Crop a label volume to a bounding box.
LabelVolume crop_to_bbox(const LabelVolume& input, const BBox& bbox);

/// Undo crop: place cropped volume back into original-sized volume.
Volume undo_crop(const Volume& cropped, const Volume& reference, const BBox& bbox);
LabelVolume undo_crop(const LabelVolume& cropped, const LabelVolume& reference, const BBox& bbox);

} // namespace totalseg
