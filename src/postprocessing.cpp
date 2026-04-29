#include "ts_postprocessing.h"
#include <queue>
#include <unordered_map>

namespace totalseg {

// 6-connectivity offsets for 3D flood fill
static constexpr int DX[6] = {-1, 1, 0, 0, 0, 0};
static constexpr int DY[6] = {0, 0, -1, 1, 0, 0};
static constexpr int DZ[6] = {0, 0, 0, 0, -1, 1};

/// 3D connected component labeling via BFS on a binary mask.
/// Returns component labels (1-based) and the number of components found.
static std::pair<std::vector<int>, int> label_components_3d(
    const std::vector<uint8_t>& binary_mask, const std::array<int, 3>& shape)
{
    size_t n = (size_t)shape[0] * shape[1] * shape[2];
    std::vector<int> labels(n, 0);
    int num_components = 0;

    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)x * shape[1] * shape[2] + y * shape[2] + z;
    };

    std::queue<std::array<int, 3>> q;

    for (int x = 0; x < shape[0]; ++x) {
        for (int y = 0; y < shape[1]; ++y) {
            for (int z = 0; z < shape[2]; ++z) {
                size_t i = idx(x, y, z);
                if (binary_mask[i] == 0 || labels[i] != 0) continue;

                ++num_components;
                labels[i] = num_components;
                q.push({x, y, z});

                while (!q.empty()) {
                    auto [cx, cy, cz] = q.front(); q.pop();
                    for (int d = 0; d < 6; ++d) {
                        int nx = cx + DX[d], ny = cy + DY[d], nz = cz + DZ[d];
                        if (nx < 0 || nx >= shape[0] || ny < 0 || ny >= shape[1] || nz < 0 || nz >= shape[2])
                            continue;
                        size_t ni = idx(nx, ny, nz);
                        if (binary_mask[ni] != 0 && labels[ni] == 0) {
                            labels[ni] = num_components;
                            q.push({nx, ny, nz});
                        }
                    }
                }
            }
        }
    }
    return {labels, num_components};
}

LabelVolume keep_largest_blob_multilabel(const LabelVolume& input, const std::vector<uint8_t>& label_ids) {
    LabelVolume out = input; // copy

    for (uint8_t label : label_ids) {
        // Extract binary mask
        std::vector<uint8_t> binary(out.numel());
        for (size_t i = 0; i < out.numel(); ++i)
            binary[i] = (out.data[i] == label) ? 1 : 0;

        auto [comp_labels, num_comp] = label_components_3d(binary, out.shape);
        if (num_comp <= 1) continue;

        // Find largest component
        std::vector<size_t> comp_sizes(num_comp + 1, 0);
        for (size_t i = 0; i < comp_labels.size(); ++i)
            if (comp_labels[i] > 0)
                comp_sizes[comp_labels[i]]++;

        int largest = 1;
        for (int c = 2; c <= num_comp; ++c)
            if (comp_sizes[c] > comp_sizes[largest])
                largest = c;

        // Zero out non-largest
        for (size_t i = 0; i < out.numel(); ++i)
            if (out.data[i] == label && comp_labels[i] != largest)
                out.data[i] = 0;
    }

    return out;
}

LabelVolume remove_small_blobs_multilabel(const LabelVolume& input, const std::vector<uint8_t>& label_ids,
                                           int min_size, int max_size) {
    LabelVolume out = input;

    for (uint8_t label : label_ids) {
        std::vector<uint8_t> binary(out.numel());
        for (size_t i = 0; i < out.numel(); ++i)
            binary[i] = (out.data[i] == label) ? 1 : 0;

        auto [comp_labels, num_comp] = label_components_3d(binary, out.shape);
        if (num_comp == 0) continue;

        std::vector<size_t> comp_sizes(num_comp + 1, 0);
        for (size_t i = 0; i < comp_labels.size(); ++i)
            if (comp_labels[i] > 0)
                comp_sizes[comp_labels[i]]++;

        for (size_t i = 0; i < out.numel(); ++i) {
            if (out.data[i] == label) {
                int c = comp_labels[i];
                size_t sz = comp_sizes[c];
                if ((int)sz < min_size || (max_size > 0 && (int)sz > max_size))
                    out.data[i] = 0;
            }
        }
    }

    return out;
}

LabelVolume remove_outside_of_mask(const LabelVolume& seg, const LabelVolume& mask, int dilation_iters) {
    // Binary dilate the mask
    std::vector<uint8_t> dilated(mask.numel());
    for (size_t i = 0; i < mask.numel(); ++i)
        dilated[i] = (mask.data[i] > 0) ? 1 : 0;

    auto idx = [&](int x, int y, int z) -> size_t {
        return (size_t)x * mask.shape[1] * mask.shape[2] + y * mask.shape[2] + z;
    };

    for (int iter = 0; iter < dilation_iters; ++iter) {
        std::vector<uint8_t> prev = dilated;
        for (int x = 0; x < mask.shape[0]; ++x) {
            for (int y = 0; y < mask.shape[1]; ++y) {
                for (int z = 0; z < mask.shape[2]; ++z) {
                    if (prev[idx(x, y, z)] != 0) continue;
                    // Check 6-neighbors
                    for (int d = 0; d < 6; ++d) {
                        int nx = x + DX[d], ny = y + DY[d], nz = z + DZ[d];
                        if (nx >= 0 && nx < mask.shape[0] &&
                            ny >= 0 && ny < mask.shape[1] &&
                            nz >= 0 && nz < mask.shape[2] &&
                            prev[idx(nx, ny, nz)] != 0) {
                            dilated[idx(x, y, z)] = 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    // Apply mask
    LabelVolume out = seg;
    for (size_t i = 0; i < out.numel(); ++i)
        if (dilated[i] == 0)
            out.data[i] = 0;

    return out;
}

LabelVolume remove_auxiliary_labels(const LabelVolume& input, const std::vector<uint8_t>& aux_label_ids) {
    LabelVolume out = input;
    for (size_t i = 0; i < out.numel(); ++i)
        for (uint8_t aux : aux_label_ids)
            if (out.data[i] == aux) {
                out.data[i] = 0;
                break;
            }
    return out;
}

} // namespace totalseg
