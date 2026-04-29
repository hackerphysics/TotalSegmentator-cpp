#include "ts_sliding_window.h"
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace totalseg {

// ── 1D Gaussian blur (in-place, along a given axis of a 3D array) ────────────

static void gaussian_blur_1d(std::vector<float>& data,
                             const std::array<int, 3>& shape,
                             int axis, double sigma) {
    if (sigma <= 0.0) return;

    // Build 1D kernel (truncated at 4*sigma)
    int radius = static_cast<int>(std::ceil(sigma * 4.0));
    int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        double d = i - radius;
        kernel[i] = static_cast<float>(std::exp(-0.5 * d * d / (sigma * sigma)));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= static_cast<float>(sum);

    // Determine iteration extents
    int dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];
    int len = shape[axis];

    // Strides in the flat array
    // layout: data[x * shape[1]*shape[2] + y * shape[2] + z]
    int stride_axis;
    if (axis == 0) stride_axis = dim1 * dim2;
    else if (axis == 1) stride_axis = dim2;
    else stride_axis = 1;

    // Number of lines to process
    int n_outer = static_cast<int>(data.size()) / len;

    // We iterate over all 1D lines along `axis`
    // Compute outer loop by enumerating the other two dimensions
    auto idx = [&](int a, int b, int c) -> size_t {
        return (size_t)a * dim1 * dim2 + b * dim2 + c;
    };

    std::vector<float> tmp(len);

    if (axis == 0) {
        for (int y = 0; y < dim1; ++y) {
            for (int z = 0; z < dim2; ++z) {
                for (int x = 0; x < dim0; ++x) {
                    float val = 0.0f;
                    for (int ki = 0; ki < ksize; ++ki) {
                        int sx = x + ki - radius;
                        // constant boundary (0)
                        if (sx >= 0 && sx < dim0)
                            val += kernel[ki] * data[idx(sx, y, z)];
                    }
                    tmp[x] = val;
                }
                for (int x = 0; x < dim0; ++x)
                    data[idx(x, y, z)] = tmp[x];
            }
        }
    } else if (axis == 1) {
        for (int x = 0; x < dim0; ++x) {
            for (int z = 0; z < dim2; ++z) {
                for (int y = 0; y < dim1; ++y) {
                    float val = 0.0f;
                    for (int ki = 0; ki < ksize; ++ki) {
                        int sy = y + ki - radius;
                        if (sy >= 0 && sy < dim1)
                            val += kernel[ki] * data[idx(x, sy, z)];
                    }
                    tmp[y] = val;
                }
                for (int y = 0; y < dim1; ++y)
                    data[idx(x, y, z)] = tmp[y];
            }
        }
    } else { // axis == 2
        for (int x = 0; x < dim0; ++x) {
            for (int y = 0; y < dim1; ++y) {
                for (int z = 0; z < dim2; ++z) {
                    float val = 0.0f;
                    for (int ki = 0; ki < ksize; ++ki) {
                        int sz = z + ki - radius;
                        if (sz >= 0 && sz < dim2)
                            val += kernel[ki] * data[idx(x, y, sz)];
                    }
                    tmp[z] = val;
                }
                for (int z = 0; z < dim2; ++z)
                    data[idx(x, y, z)] = tmp[z];
            }
        }
    }
}

// ── Gaussian importance map ──────────────────────────────────────────────────

std::vector<float> create_gaussian_importance_map(const std::array<int, 3>& patch_size, double sigma_scale) {
    // patch_size is [z, y, x]
    int pz = patch_size[0], py = patch_size[1], px = patch_size[2];
    size_t numel = (size_t)pz * py * px;

    // We store as shape [pz, py, px] with layout [z * py*px + y * px + x]
    std::vector<float> gauss(numel, 0.0f);

    // Place a 1 at center
    int cz = pz / 2, cy = py / 2, cx = px / 2;
    gauss[(size_t)cz * py * px + cy * px + cx] = 1.0f;

    // Separable Gaussian blur: shape for the blur routines is {pz, py, px}
    std::array<int, 3> shape = {pz, py, px};
    double sigma_z = pz * sigma_scale;
    double sigma_y = py * sigma_scale;
    double sigma_x = px * sigma_scale;

    gaussian_blur_1d(gauss, shape, 0, sigma_z);
    gaussian_blur_1d(gauss, shape, 1, sigma_y);
    gaussian_blur_1d(gauss, shape, 2, sigma_x);

    // Normalize so max = 1
    float max_val = *std::max_element(gauss.begin(), gauss.end());
    if (max_val > 0.0f) {
        for (auto& v : gauss) v /= max_val;
    }

    // Replace zeros with minimum non-zero value
    float min_nonzero = std::numeric_limits<float>::max();
    for (auto v : gauss) {
        if (v > 0.0f && v < min_nonzero) min_nonzero = v;
    }
    for (auto& v : gauss) {
        if (v == 0.0f) v = min_nonzero;
    }

    return gauss;
}

// ── Compute sliding window steps ─────────────────────────────────────────────

static std::array<std::vector<int>, 3> compute_steps(
    const std::array<int, 3>& image_size,
    const std::array<int, 3>& tile_size,
    double tile_step_size)
{
    std::array<std::vector<int>, 3> steps;
    for (int dim = 0; dim < 3; ++dim) {
        double target_step = tile_size[dim] * tile_step_size;
        int num_steps = static_cast<int>(std::ceil(
            static_cast<double>(image_size[dim] - tile_size[dim]) / target_step)) + 1;
        if (num_steps < 1) num_steps = 1;

        int max_step_value = image_size[dim] - tile_size[dim];
        double actual_step = (num_steps > 1)
            ? static_cast<double>(max_step_value) / (num_steps - 1)
            : 99999999999.0;

        steps[dim].resize(num_steps);
        for (int i = 0; i < num_steps; ++i) {
            steps[dim][i] = static_cast<int>(std::round(actual_step * i));
        }
    }
    return steps;
}

// ── Mirror padding helpers ───────────────────────────────────────────────────

static inline int mirror_index(int idx, int size) {
    // Reflect index into [0, size-1]
    if (idx < 0) idx = -idx - 1;
    if (idx >= size) idx = 2 * size - idx - 1;
    // Clamp for safety
    if (idx < 0) idx = 0;
    if (idx >= size) idx = size - 1;
    return idx;
}

static Volume mirror_pad_volume(const Volume& input, const std::array<int, 3>& pad_before) {
    // pad_before[d] = amount to pad on each side of dim d
    // We pad equally on both sides, but only need enough so padded_size >= patch_size
    // Actually, we compute the padded size from the caller.
    // Let's define: padded_shape[d] = input.shape[d] + 2*pad_before[d]

    Volume padded;
    for (int d = 0; d < 3; ++d)
        padded.shape[d] = input.shape[d] + 2 * pad_before[d];
    padded.spacing = input.spacing;
    padded.affine = input.affine;
    padded.data.resize(padded.numel());

    int ox = pad_before[0], oy = pad_before[1], oz = pad_before[2];
    int sx = input.shape[0], sy = input.shape[1], sz = input.shape[2];
    int px = padded.shape[0], py = padded.shape[1], pz = padded.shape[2];

    for (int x = 0; x < px; ++x) {
        int src_x = mirror_index(x - ox, sx);
        for (int y = 0; y < py; ++y) {
            int src_y = mirror_index(y - oy, sy);
            for (int z = 0; z < pz; ++z) {
                int src_z = mirror_index(z - oz, sz);
                padded.at(x, y, z) = input.at(src_x, src_y, src_z);
            }
        }
    }
    return padded;
}

// ── Main sliding window inference ────────────────────────────────────────────

LabelVolume sliding_window_inference(
    const Volume& input,
    const std::array<int, 3>& patch_size,  // [z, y, x]
    double step_size,
    int num_classes,
    InferenceFunc infer_fn)
{
    // All dimensions in (x, y, z) order, matching Volume convention.
    // patch_size is also (x, y, z) = (128, 128, 128) typically.
    int img_x = input.shape[0], img_y = input.shape[1], img_z = input.shape[2];
    int px = patch_size[0], py = patch_size[1], pz = patch_size[2];

    // Step 1: Mirror-pad so each dim >= corresponding patch dim
    std::array<int, 3> pad_before = {0, 0, 0};
    if (img_x < px) pad_before[0] = (px - img_x + 1) / 2;
    if (img_y < py) pad_before[1] = (py - img_y + 1) / 2;
    if (img_z < pz) pad_before[2] = (pz - img_z + 1) / 2;

    bool needs_padding = (pad_before[0] > 0 || pad_before[1] > 0 || pad_before[2] > 0);
    Volume padded = needs_padding ? mirror_pad_volume(input, pad_before) : input;

    int pdx = padded.shape[0], pdy = padded.shape[1], pdz = padded.shape[2];

    // Step 2: Gaussian importance map — stored in (x, y, z) C-order
    // We create it using the same function but interpret as (x, y, z)
    auto gauss = create_gaussian_importance_map(patch_size);

    // Step 3: Compute sliding window steps in (x, y, z) order
    std::array<int, 3> img_size_xyz = {pdx, pdy, pdz};
    auto steps = compute_steps(img_size_xyz, patch_size, step_size);
    // steps[0] = x-steps, steps[1] = y-steps, steps[2] = z-steps

    // Step 4: Allocate prediction and count buffers
    size_t spatial_numel = (size_t)pdx * pdy * pdz;
    std::vector<float> pred_buf(num_classes * spatial_numel, 0.0f);
    std::vector<float> count_buf(spatial_numel, 0.0f);

    // Index in C-order (x-slowest, z-fastest) matching Volume.at()
    auto spatial_idx = [&](int x, int y, int z) -> size_t {
        return (size_t)x * pdy * pdz + y * pdz + z;
    };
    auto pred_idx = [&](int cls, int x, int y, int z) -> size_t {
        return (size_t)cls * spatial_numel + spatial_idx(x, y, z);
    };

    size_t patch_numel = (size_t)px * py * pz;

    // Iterate over all window positions
    for (int sx : steps[0]) {        // x start positions
        for (int sy : steps[1]) {    // y start positions
            for (int sz : steps[2]) {// z start positions
                // Extract patch in C-order (x-slowest): matches numpy/Python convention
                // This is what gets fed to ONNX as (1, 1, x, y, z) matching how Python does it
                std::vector<float> patch(patch_numel);
                for (int dx = 0; dx < px; ++dx) {
                    for (int dy = 0; dy < py; ++dy) {
                        for (int dz = 0; dz < pz; ++dz) {
                            patch[(size_t)dx * py * pz + dy * pz + dz] =
                                padded.at(sx + dx, sy + dy, sz + dz);
                        }
                    }
                }

                // Call inference: shape (1, 1, px, py, pz)
                std::array<int, 5> input_shape = {1, 1, px, py, pz};
                auto logits = infer_fn(patch, input_shape);
                // logits shape: (1, num_classes, px, py, pz) stored C-contiguous

                // Multiply by gaussian and accumulate
                for (int cls = 0; cls < num_classes; ++cls) {
                    size_t cls_offset = (size_t)cls * patch_numel;
                    for (int dx = 0; dx < px; ++dx) {
                        for (int dy = 0; dy < py; ++dy) {
                            for (int dz = 0; dz < pz; ++dz) {
                                size_t p_idx = (size_t)dx * py * pz + dy * pz + dz;
                                float g = gauss[p_idx];
                                float logit = logits[cls_offset + p_idx];
                                pred_buf[pred_idx(cls, sx + dx, sy + dy, sz + dz)] += logit * g;
                            }
                        }
                    }
                }

                // Accumulate count
                for (int dx = 0; dx < px; ++dx) {
                    for (int dy = 0; dy < py; ++dy) {
                        for (int dz = 0; dz < pz; ++dz) {
                            size_t p_idx = (size_t)dx * py * pz + dy * pz + dz;
                            count_buf[spatial_idx(sx + dx, sy + dy, sz + dz)] += gauss[p_idx];
                        }
                    }
                }
            }
        }
    }

    // Step 5: Divide prediction by count
    for (int cls = 0; cls < num_classes; ++cls) {
        for (size_t i = 0; i < spatial_numel; ++i) {
            if (count_buf[i] > 0.0f) {
                pred_buf[(size_t)cls * spatial_numel + i] /= count_buf[i];
            }
        }
    }

    // Step 6: Argmax → LabelVolume (in padded space)
    // Step 7: Crop back to original size
    LabelVolume result;
    result.shape = input.shape;
    result.spacing = input.spacing;
    result.affine = input.affine;
    result.data.resize(input.numel(), 0);

    for (int x = 0; x < img_x; ++x) {
        for (int y = 0; y < img_y; ++y) {
            for (int z = 0; z < img_z; ++z) {
                // Padded coords
                int px_coord = x + pad_before[0];
                int py_coord = y + pad_before[1];
                int pz_coord = z + pad_before[2];

                // Argmax across classes
                uint8_t best_cls = 0;
                float best_val = pred_buf[pred_idx(0, px_coord, py_coord, pz_coord)];
                for (int cls = 1; cls < num_classes; ++cls) {
                    float val = pred_buf[pred_idx(cls, px_coord, py_coord, pz_coord)];
                    if (val > best_val) {
                        best_val = val;
                        best_cls = static_cast<uint8_t>(cls);
                    }
                }
                result.at(x, y, z) = best_cls;
            }
        }
    }

    return result;
}

} // namespace totalseg
