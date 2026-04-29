#include "ts_cropping.h"
#include <limits>

namespace totalseg {

BBox get_bbox_from_mask(const LabelVolume& mask, const std::array<int, 3>& addon_voxels) {
    int min_x = mask.shape[0], max_x = -1;
    int min_y = mask.shape[1], max_y = -1;
    int min_z = mask.shape[2], max_z = -1;

    for (int x = 0; x < mask.shape[0]; ++x) {
        for (int y = 0; y < mask.shape[1]; ++y) {
            for (int z = 0; z < mask.shape[2]; ++z) {
                if (mask.at(x, y, z) > 0) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                    if (z < min_z) min_z = z;
                    if (z > max_z) max_z = z;
                }
            }
        }
    }

    // If mask is entirely zero, return full volume bbox
    if (max_x < 0) {
        return {{{0, mask.shape[0] - 1}, {0, mask.shape[1] - 1}, {0, mask.shape[2] - 1}}};
    }

    // Add padding and clamp
    min_x = std::max(0, min_x - addon_voxels[0]);
    max_x = std::min(mask.shape[0] - 1, max_x + addon_voxels[0]);
    min_y = std::max(0, min_y - addon_voxels[1]);
    max_y = std::min(mask.shape[1] - 1, max_y + addon_voxels[1]);
    min_z = std::max(0, min_z - addon_voxels[2]);
    max_z = std::min(mask.shape[2] - 1, max_z + addon_voxels[2]);

    return {{{min_x, max_x}, {min_y, max_y}, {min_z, max_z}}};
}

// Helper: compute updated affine after cropping
static std::array<std::array<double, 4>, 4> compute_cropped_affine(
    const std::array<std::array<double, 4>, 4>& affine, const BBox& bbox)
{
    // new_origin = affine * [bbox[0][0], bbox[1][0], bbox[2][0], 1]^T
    double vx = bbox[0][0], vy = bbox[1][0], vz = bbox[2][0];
    auto result = affine;
    result[0][3] = affine[0][0] * vx + affine[0][1] * vy + affine[0][2] * vz + affine[0][3];
    result[1][3] = affine[1][0] * vx + affine[1][1] * vy + affine[1][2] * vz + affine[1][3];
    result[2][3] = affine[2][0] * vx + affine[2][1] * vy + affine[2][2] * vz + affine[2][3];
    return result;
}

Volume crop_to_bbox(const Volume& input, const BBox& bbox) {
    int nx = bbox[0][1] - bbox[0][0] + 1;
    int ny = bbox[1][1] - bbox[1][0] + 1;
    int nz = bbox[2][1] - bbox[2][0] + 1;

    Volume out;
    out.shape = {nx, ny, nz};
    out.spacing = input.spacing;
    out.affine = compute_cropped_affine(input.affine, bbox);
    out.data.resize(out.numel());

    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                out.at(x, y, z) = input.at(x + bbox[0][0], y + bbox[1][0], z + bbox[2][0]);

    return out;
}

LabelVolume crop_to_bbox(const LabelVolume& input, const BBox& bbox) {
    int nx = bbox[0][1] - bbox[0][0] + 1;
    int ny = bbox[1][1] - bbox[1][0] + 1;
    int nz = bbox[2][1] - bbox[2][0] + 1;

    LabelVolume out;
    out.shape = {nx, ny, nz};
    out.spacing = input.spacing;
    out.affine = compute_cropped_affine(input.affine, bbox);
    out.data.resize(out.numel());

    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                out.at(x, y, z) = input.at(x + bbox[0][0], y + bbox[1][0], z + bbox[2][0]);

    return out;
}

Volume undo_crop(const Volume& cropped, const Volume& reference, const BBox& bbox) {
    Volume out;
    out.shape = reference.shape;
    out.spacing = reference.spacing;
    out.affine = reference.affine;
    out.data.assign(out.numel(), 0.0f);

    for (int x = 0; x < cropped.shape[0]; ++x)
        for (int y = 0; y < cropped.shape[1]; ++y)
            for (int z = 0; z < cropped.shape[2]; ++z)
                out.at(x + bbox[0][0], y + bbox[1][0], z + bbox[2][0]) = cropped.at(x, y, z);

    return out;
}

LabelVolume undo_crop(const LabelVolume& cropped, const LabelVolume& reference, const BBox& bbox) {
    LabelVolume out;
    out.shape = reference.shape;
    out.spacing = reference.spacing;
    out.affine = reference.affine;
    out.data.assign(out.numel(), 0);

    for (int x = 0; x < cropped.shape[0]; ++x)
        for (int y = 0; y < cropped.shape[1]; ++y)
            for (int z = 0; z < cropped.shape[2]; ++z)
                out.at(x + bbox[0][0], y + bbox[1][0], z + bbox[2][0]) = cropped.at(x, y, z);

    return out;
}

} // namespace totalseg
