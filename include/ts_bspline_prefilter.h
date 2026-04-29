#pragma once
#include <vector>
#include <array>

namespace totalseg {

// Apply scipy-compatible B-spline prefilter (order=3) to 3D volume
// Returns double-precision coefficients (matching scipy's internal float64 pipeline)
void bspline_prefilter_3d(const std::vector<float>& data, const std::array<int, 3>& shape,
                         std::vector<double>& coeffs);

} // namespace totalseg
