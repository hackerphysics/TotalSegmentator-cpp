#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <memory>
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace totalseg {

/// 3D volume stored as contiguous float array in [x][y][z] order (nibabel convention).
struct Volume {
    std::vector<float> data;
    std::array<int, 3> shape;          // {x, y, z}
    std::array<double, 3> spacing;     // voxel spacing in mm
    std::array<std::array<double, 4>, 4> affine; // 4x4 affine matrix

    size_t numel() const { return (size_t)shape[0] * shape[1] * shape[2]; }
        // C order: index = x*Ny*Nz + y*Nz + z (matches numpy default after nibabel load)
    float& at(int x, int y, int z) { return data[(size_t)x * shape[1] * shape[2] + y * shape[2] + z]; }
    float  at(int x, int y, int z) const { return data[(size_t)x * shape[1] * shape[2] + y * shape[2] + z]; }
};

/// Integer-label volume (segmentation mask).
struct LabelVolume {
    std::vector<uint8_t> data;
    std::array<int, 3> shape;
    std::array<double, 3> spacing;
    std::array<std::array<double, 4>, 4> affine;

    size_t numel() const { return (size_t)shape[0] * shape[1] * shape[2]; }
        // C order: index = x*Ny*Nz + y*Nz + z
    uint8_t& at(int x, int y, int z) { return data[(size_t)x * shape[1] * shape[2] + y * shape[2] + z]; }
    uint8_t  at(int x, int y, int z) const { return data[(size_t)x * shape[1] * shape[2] + y * shape[2] + z]; }
};

/// Bounding box: {{min_x, max_x}, {min_y, max_y}, {min_z, max_z}}
using BBox = std::array<std::array<int, 2>, 3>;

/// Orientation code (RAS, LPS, etc.)
using OrientCode = std::array<char, 3>;

} // namespace totalseg
