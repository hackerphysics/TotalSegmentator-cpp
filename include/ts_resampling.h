#pragma once
#include "ts_types.h"

namespace totalseg {

/// Resample volume to new spacing.
/// order: 0 = nearest neighbor (for labels), 1 = trilinear, 3 = cubic (for images).
Volume change_spacing(const Volume& input, const std::array<double, 3>& new_spacing, int order = 3);

/// Resample volume to exact target shape (spacing is computed from shape).
Volume resample_to_shape(const Volume& input, const std::array<int, 3>& target_shape, int order = 3);

/// Resample label volume to new spacing (always nearest-neighbor).
LabelVolume change_spacing_label(const LabelVolume& input, const std::array<double, 3>& new_spacing);

/// Resample label volume to exact target shape.
LabelVolume resample_to_shape_label(const LabelVolume& input, const std::array<int, 3>& target_shape);

} // namespace totalseg
