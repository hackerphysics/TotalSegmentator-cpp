#pragma once
#include "ts_types.h"

namespace totalseg {

/// Convert volume to closest canonical orientation (RAS).
/// Returns the reoriented volume. Original orientation is stored for undo.
Volume to_canonical(const Volume& input, OrientCode& original_orient);

/// Undo canonical reorientation — transform back to original orientation.
Volume undo_canonical(const Volume& canonical, const OrientCode& original_orient);
LabelVolume undo_canonical(const LabelVolume& canonical, const OrientCode& original_orient);

/// Get orientation code from affine matrix.
OrientCode get_orientation(const std::array<std::array<double, 4>, 4>& affine);

} // namespace totalseg
