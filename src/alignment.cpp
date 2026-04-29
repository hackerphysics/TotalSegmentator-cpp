#include "ts_alignment.h"
#include <cmath>
#include <algorithm>

namespace totalseg {

// Axis labels: R/L for axis 0, A/P for axis 1, S/I for axis 2
// Positive direction = R, A, S (RAS convention)
static const char POS_LABELS[3] = {'R', 'A', 'S'};
static const char NEG_LABELS[3] = {'L', 'P', 'I'};

OrientCode get_orientation(const std::array<std::array<double, 4>, 4>& affine) {
    // Extract the 3x3 direction cosine matrix (upper-left of affine)
    // Each column j of the rotation matrix tells us the anatomical direction of voxel axis j.
    // We find which anatomical axis each voxel axis most closely aligns with.
    OrientCode orient;

    // For each voxel axis (column j), find the row with largest absolute value
    for (int j = 0; j < 3; ++j) {
        double max_val = 0.0;
        int max_row = 0;
        for (int i = 0; i < 3; ++i) {
            double v = std::fabs(affine[i][j]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        orient[j] = (affine[max_row][j] > 0) ? POS_LABELS[max_row] : NEG_LABELS[max_row];
    }
    return orient;
}

// Map an orientation character to (anatomical_axis, is_positive)
static std::pair<int, bool> orient_char_to_axis(char c) {
    switch (c) {
        case 'R': return {0, true};
        case 'L': return {0, false};
        case 'A': return {1, true};
        case 'P': return {1, false};
        case 'S': return {2, true};
        case 'I': return {2, false};
    }
    return {0, true}; // unreachable
}

// Compute the transform from source orientation to target orientation.
// Returns (perm, flip) where:
//   perm[i] = which source axis maps to target axis i
//   flip[i] = whether that axis needs to be flipped
static void compute_orient_transform(const OrientCode& src, const OrientCode& tgt,
                                     std::array<int, 3>& perm, std::array<bool, 3>& flip) {
    // For each target axis i, find which source axis j maps to the same anatomical axis
    for (int i = 0; i < 3; ++i) {
        auto [tgt_anat, tgt_pos] = orient_char_to_axis(tgt[i]);
        for (int j = 0; j < 3; ++j) {
            auto [src_anat, src_pos] = orient_char_to_axis(src[j]);
            if (src_anat == tgt_anat) {
                perm[i] = j;
                flip[i] = (src_pos != tgt_pos);
                break;
            }
        }
    }
}

// Apply axis permutation + flips to a volume's data
template<typename T>
static std::vector<T> permute_and_flip(const std::vector<T>& data,
                                       const std::array<int, 3>& old_shape,
                                       const std::array<int, 3>& perm,
                                       const std::array<bool, 3>& flip,
                                       std::array<int, 3>& new_shape) {
    // New shape after permutation
    for (int i = 0; i < 3; ++i)
        new_shape[i] = old_shape[perm[i]];

    std::vector<T> out(data.size());
    int nx = new_shape[0], ny = new_shape[1], nz = new_shape[2];

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                // Map new coords to old coords
                int new_coords[3] = {x, y, z};
                int old_coords[3];
                for (int i = 0; i < 3; ++i) {
                    int c = new_coords[i];
                    if (flip[i]) c = new_shape[i] - 1 - c;
                    old_coords[perm[i]] = c;
                }
                size_t old_idx = (size_t)old_coords[0] * old_shape[1] * old_shape[2]
                               + old_coords[1] * old_shape[2] + old_coords[2];
                size_t new_idx = (size_t)x * ny * nz + y * nz + z;
                out[new_idx] = data[old_idx];
            }
        }
    }
    return out;
}

// Transform affine given permutation and flips
static std::array<std::array<double, 4>, 4> transform_affine(
        const std::array<std::array<double, 4>, 4>& affine,
        const std::array<int, 3>& old_shape,
        const std::array<int, 3>& perm,
        const std::array<bool, 3>& flip) {
    auto out = affine;
    // Step 1: Permute columns of the 3x3 part
    std::array<std::array<double, 4>, 4> tmp = affine;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = tmp[i][perm[j]];
        }
    }
    // Step 2: Flip — for flipped axes, negate the column and shift the origin
    for (int j = 0; j < 3; ++j) {
        if (flip[j]) {
            int src_axis = perm[j];
            int n = old_shape[src_axis]; // length along the source axis
            for (int i = 0; i < 3; ++i) {
                out[i][3] += out[i][j] * (n - 1);
                out[i][j] = -out[i][j];
            }
        }
    }
    return out;
}

Volume to_canonical(const Volume& input, OrientCode& original_orient) {
    original_orient = get_orientation(input.affine);
    OrientCode ras = {'R', 'A', 'S'};

    // Check if already RAS
    if (original_orient == ras) {
        return input; // already canonical
    }

    std::array<int, 3> perm;
    std::array<bool, 3> flip;
    compute_orient_transform(original_orient, ras, perm, flip);

    Volume out;
    out.data = permute_and_flip(input.data, input.shape, perm, flip, out.shape);
    out.affine = transform_affine(input.affine, input.shape, perm, flip);

    // Update spacing
    for (int i = 0; i < 3; ++i)
        out.spacing[i] = input.spacing[perm[i]];

    return out;
}

Volume undo_canonical(const Volume& canonical, const OrientCode& original_orient) {
    OrientCode ras = {'R', 'A', 'S'};
    if (original_orient == ras) return canonical;

    // Inverse: go from RAS back to original_orient
    std::array<int, 3> perm;
    std::array<bool, 3> flip;
    compute_orient_transform(ras, original_orient, perm, flip);

    Volume out;
    out.data = permute_and_flip(canonical.data, canonical.shape, perm, flip, out.shape);
    out.affine = transform_affine(canonical.affine, canonical.shape, perm, flip);

    for (int i = 0; i < 3; ++i)
        out.spacing[i] = canonical.spacing[perm[i]];

    return out;
}

LabelVolume undo_canonical(const LabelVolume& canonical, const OrientCode& original_orient) {
    OrientCode ras = {'R', 'A', 'S'};
    if (original_orient == ras) {
        return canonical;
    }

    std::array<int, 3> perm;
    std::array<bool, 3> flip;
    compute_orient_transform(ras, original_orient, perm, flip);

    LabelVolume out;
    out.data = permute_and_flip(canonical.data, canonical.shape, perm, flip, out.shape);
    out.affine = transform_affine(canonical.affine, canonical.shape, perm, flip);

    for (int i = 0; i < 3; ++i)
        out.spacing[i] = canonical.spacing[perm[i]];

    return out;
}

} // namespace totalseg
