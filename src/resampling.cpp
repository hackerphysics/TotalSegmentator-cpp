#include "ts_resampling.h"
#include "ts_bspline_prefilter.h"
#include <cmath>
#include <algorithm>

namespace totalseg {

// Nearest-neighbor rounding matching scipy.ndimage.zoom(order=0, grid_mode=False).
// scipy uses: zoom_ratio = (old_size-1)/(new_size-1), src = dst * zoom_ratio, rint(src)
static inline int scipy_nearest(double src_coord) {
    // rint = round to nearest, ties to even (banker's rounding)
    return static_cast<int>(std::rint(src_coord));
}

// Clamp value to [lo, hi]
static inline int clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Cubic B-spline basis (used by scipy.ndimage.zoom with order=3)
static inline double cubic_weight(double x) {
    double ax = std::fabs(x);
    if (ax <= 1.0)
        return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0;
    else if (ax < 2.0)
        return (8.0 - 12.0 * ax + 6.0 * ax * ax - ax * ax * ax) / 6.0;
    return 0.0;
}

// Nearest-neighbor resampling
static std::vector<float> resample_nearest(const std::vector<float>& data,
                                           const std::array<int, 3>& old_shape,
                                           const std::array<int, 3>& new_shape,
                                           const std::array<double, 3>& /*zoom_unused*/) {
    int nx = new_shape[0], ny = new_shape[1], nz = new_shape[2];
    std::vector<float> out(static_cast<size_t>(nx) * ny * nz);

    std::array<double, 3> zr;
    for (int d = 0; d < 3; ++d)
        zr[d] = (new_shape[d] > 1) ? (double)(old_shape[d] - 1) / (new_shape[d] - 1) : 0.0;

    for (int x = 0; x < nx; ++x) {
        int ox = clamp(static_cast<int>(std::rint(x * zr[0])), 0, old_shape[0] - 1);
        for (int y = 0; y < ny; ++y) {
            int oy = clamp(static_cast<int>(std::rint(y * zr[1])), 0, old_shape[1] - 1);
            for (int z = 0; z < nz; ++z) {
                int oz = clamp(static_cast<int>(std::rint(z * zr[2])), 0, old_shape[2] - 1);
                out[static_cast<size_t>(x) * ny * nz + y * nz + z] =
                    data[static_cast<size_t>(ox) * old_shape[1] * old_shape[2] + oy * old_shape[2] + oz];
            }
        }
    }
    return out;
}

// Trilinear resampling
static std::vector<float> resample_trilinear(const std::vector<float>& data,
                                             const std::array<int, 3>& old_shape,
                                             const std::array<int, 3>& new_shape,
                                             const std::array<double, 3>& /*zoom_unused*/) {
    int nx = new_shape[0], ny = new_shape[1], nz = new_shape[2];
    int ox_max = old_shape[0] - 1, oy_max = old_shape[1] - 1, oz_max = old_shape[2] - 1;
    size_t old_stride0 = static_cast<size_t>(old_shape[1]) * old_shape[2];
    size_t old_stride1 = old_shape[2];

    // scipy zoom_ratio = (old-1)/(new-1) for grid_mode=False
    std::array<double, 3> zr;
    for (int d = 0; d < 3; ++d)
        zr[d] = (new_shape[d] > 1) ? (double)(old_shape[d] - 1) / (new_shape[d] - 1) : 0.0;

    std::vector<float> out(static_cast<size_t>(nx) * ny * nz);

    for (int x = 0; x < nx; ++x) {
        double fx = x * zr[0];
        int x0 = static_cast<int>(std::floor(fx));
        double dx = fx - x0;
        int x1 = std::min(x0 + 1, ox_max);
        x0 = clamp(x0, 0, ox_max);

        for (int y = 0; y < ny; ++y) {
            double fy = y * zr[1];
            int y0 = static_cast<int>(std::floor(fy));
            double dy = fy - y0;
            int y1 = std::min(y0 + 1, oy_max);
            y0 = clamp(y0, 0, oy_max);

            for (int z = 0; z < nz; ++z) {
                double fz = z * zr[2];
                int z0 = static_cast<int>(std::floor(fz));
                double dz = fz - z0;
                int z1 = std::min(z0 + 1, oz_max);
                z0 = clamp(z0, 0, oz_max);

                // Trilinear interpolation
                double c00 = data[x0 * old_stride0 + y0 * old_stride1 + z0] * (1.0 - dz)
                           + data[x0 * old_stride0 + y0 * old_stride1 + z1] * dz;
                double c01 = data[x0 * old_stride0 + y1 * old_stride1 + z0] * (1.0 - dz)
                           + data[x0 * old_stride0 + y1 * old_stride1 + z1] * dz;
                double c10 = data[x1 * old_stride0 + y0 * old_stride1 + z0] * (1.0 - dz)
                           + data[x1 * old_stride0 + y0 * old_stride1 + z1] * dz;
                double c11 = data[x1 * old_stride0 + y1 * old_stride1 + z0] * (1.0 - dz)
                           + data[x1 * old_stride0 + y1 * old_stride1 + z1] * dz;

                double c0 = c00 * (1.0 - dy) + c01 * dy;
                double c1 = c10 * (1.0 - dy) + c11 * dy;

                out[static_cast<size_t>(x) * ny * nz + y * nz + z] =
                    static_cast<float>(c0 * (1.0 - dx) + c1 * dx);
            }
        }
    }
    return out;
}

// Cubic B-spline resampling (order=3)
// Pad 3D float data by npad on each side using edge (nearest) mode
static std::vector<float> pad_edge_3d(const std::vector<float>& data,
                                      const std::array<int, 3>& shape, int npad,
                                      std::array<int, 3>& padded_shape) {
    for (int d = 0; d < 3; ++d) padded_shape[d] = shape[d] + 2 * npad;
    int pnx = padded_shape[0], pny = padded_shape[1], pnz = padded_shape[2];
    size_t ps0 = static_cast<size_t>(pny) * pnz;
    size_t os0 = static_cast<size_t>(shape[1]) * shape[2];
    std::vector<float> out(static_cast<size_t>(pnx) * pny * pnz);
    for (int px = 0; px < pnx; ++px) {
        int ox = clamp(px - npad, 0, shape[0] - 1);
        for (int py = 0; py < pny; ++py) {
            int oy = clamp(py - npad, 0, shape[1] - 1);
            for (int pz = 0; pz < pnz; ++pz) {
                int oz = clamp(pz - npad, 0, shape[2] - 1);
                out[px * ps0 + py * pnz + pz] = data[ox * os0 + oy * shape[2] + oz];
            }
        }
    }
    return out;
}

static std::vector<float> resample_cubic(const std::vector<float>& data,
                                         const std::array<int, 3>& old_shape,
                                         const std::array<int, 3>& new_shape,
                                         const std::array<double, 3>& /*zoom_unused*/) {
    // scipy zoom(mode='nearest') pads by 12, then prefilters, then interpolates with offset
    constexpr int NPAD = 12;
    std::array<int, 3> pad_shape;
    auto padded = pad_edge_3d(data, old_shape, NPAD, pad_shape);

    // Prefilter padded data in double precision
    std::vector<double> coeffs;
    bspline_prefilter_3d(padded, pad_shape, coeffs);
    padded.clear(); padded.shrink_to_fit();  // free padded float data

    int nx = new_shape[0], ny = new_shape[1], nz = new_shape[2];
    size_t ps0 = static_cast<size_t>(pad_shape[1]) * pad_shape[2];
    size_t ps1 = pad_shape[2];

    // zoom_ratio = (old-1)/(new-1); coord = kk * zoom_ratio + NPAD
    std::array<double, 3> zr;
    for (int d = 0; d < 3; ++d)
        zr[d] = (new_shape[d] > 1) ? (double)(old_shape[d] - 1) / (new_shape[d] - 1) : 0.0;

    std::vector<float> out(static_cast<size_t>(nx) * ny * nz);

    for (int x = 0; x < nx; ++x) {
        double fx = x * zr[0] + NPAD;
        int ix = static_cast<int>(std::floor(fx));
        double dx = fx - ix;

        for (int y = 0; y < ny; ++y) {
            double fy = y * zr[1] + NPAD;
            int iy = static_cast<int>(std::floor(fy));
            double dy = fy - iy;

            for (int z = 0; z < nz; ++z) {
                double fz = z * zr[2] + NPAD;
                int iz = static_cast<int>(std::floor(fz));
                double dz = fz - iz;

                double val = 0.0;
                for (int di = -1; di <= 2; ++di) {
                    double wx = cubic_weight(dx - di);
                    int cx = clamp(ix + di, 0, pad_shape[0] - 1);
                    for (int dj = -1; dj <= 2; ++dj) {
                        double wy = cubic_weight(dy - dj);
                        int cy = clamp(iy + dj, 0, pad_shape[1] - 1);
                        for (int dk = -1; dk <= 2; ++dk) {
                            double wz = cubic_weight(dz - dk);
                            int cz = clamp(iz + dk, 0, pad_shape[2] - 1);
                            val += wx * wy * wz * coeffs[static_cast<size_t>(cx) * ps0 + cy * ps1 + cz];
                        }
                    }
                }
                out[static_cast<size_t>(x) * ny * nz + y * nz + z] = static_cast<float>(val);
            }
        }
    }
    return out;
}

// Nearest-neighbor resampling for label volumes
// Nearest-neighbor resampling matching scipy.ndimage.zoom(order=0, grid_mode=False).
static std::vector<uint8_t> resample_nearest_label(const std::vector<uint8_t>& data,
                                                   const std::array<int, 3>& old_shape,
                                                   const std::array<int, 3>& new_shape,
                                                   const std::array<double, 3>& /*zoom_unused*/) {
    int nx = new_shape[0], ny = new_shape[1], nz = new_shape[2];
    std::vector<uint8_t> out(static_cast<size_t>(nx) * ny * nz);

    // scipy zoom_ratio = (old-1)/(new-1), src = dst * zoom_ratio, rint(src)
    std::array<double, 3> zr;
    for (int d = 0; d < 3; ++d)
        zr[d] = (new_shape[d] > 1) ? (double)(old_shape[d] - 1) / (new_shape[d] - 1) : 0.0;

    for (int x = 0; x < nx; ++x) {
        int ox = clamp(static_cast<int>(std::rint(x * zr[0])), 0, old_shape[0] - 1);
        for (int y = 0; y < ny; ++y) {
            int oy = clamp(static_cast<int>(std::rint(y * zr[1])), 0, old_shape[1] - 1);
            for (int z = 0; z < nz; ++z) {
                int oz = clamp(static_cast<int>(std::rint(z * zr[2])), 0, old_shape[2] - 1);
                out[static_cast<size_t>(x) * ny * nz + y * nz + z] =
                    data[static_cast<size_t>(ox) * old_shape[1] * old_shape[2] + oy * old_shape[2] + oz];
            }
        }
    }
    return out;
}

// Update affine after resampling: scale the direction columns by 1/zoom
static std::array<std::array<double, 4>, 4> scale_affine(
        const std::array<std::array<double, 4>, 4>& affine,
        const std::array<double, 3>& zoom) {
    auto out = affine;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] /= zoom[j];
        }
    }
    // Origin stays the same
    return out;
}

Volume change_spacing(const Volume& input, const std::array<double, 3>& new_spacing, int order) {
    std::array<double, 3> zoom;
    std::array<int, 3> new_shape;
    for (int i = 0; i < 3; ++i) {
        zoom[i] = input.spacing[i] / new_spacing[i];
        new_shape[i] = static_cast<int>(std::round(input.shape[i] * zoom[i]));
        if (new_shape[i] < 1) new_shape[i] = 1;
    }

    Volume out;
    out.shape = new_shape;
    out.spacing = new_spacing;
    out.affine = scale_affine(input.affine, zoom);

    if (order == 0)
        out.data = resample_nearest(input.data, input.shape, new_shape, zoom);
    else if (order == 1)
        out.data = resample_trilinear(input.data, input.shape, new_shape, zoom);
    else
        out.data = resample_cubic(input.data, input.shape, new_shape, zoom);

    return out;
}

Volume resample_to_shape(const Volume& input, const std::array<int, 3>& target_shape, int order) {
    std::array<double, 3> zoom;
    for (int i = 0; i < 3; ++i) {
        zoom[i] = static_cast<double>(target_shape[i]) / input.shape[i];
    }

    std::array<double, 3> new_spacing;
    for (int i = 0; i < 3; ++i) {
        new_spacing[i] = input.spacing[i] / zoom[i];
    }

    Volume out;
    out.shape = target_shape;
    out.spacing = new_spacing;
    out.affine = scale_affine(input.affine, zoom);

    if (order == 0)
        out.data = resample_nearest(input.data, input.shape, target_shape, zoom);
    else if (order == 1)
        out.data = resample_trilinear(input.data, input.shape, target_shape, zoom);
    else
        out.data = resample_cubic(input.data, input.shape, target_shape, zoom);

    return out;
}

LabelVolume change_spacing_label(const LabelVolume& input, const std::array<double, 3>& new_spacing) {
    std::array<double, 3> zoom;
    std::array<int, 3> new_shape;
    for (int i = 0; i < 3; ++i) {
        zoom[i] = input.spacing[i] / new_spacing[i];
        new_shape[i] = static_cast<int>(std::round(input.shape[i] * zoom[i]));
        if (new_shape[i] < 1) new_shape[i] = 1;
    }

    LabelVolume out;
    out.shape = new_shape;
    out.spacing = new_spacing;
    out.affine = scale_affine(input.affine, zoom);
    out.data = resample_nearest_label(input.data, input.shape, new_shape, zoom);
    return out;
}

LabelVolume resample_to_shape_label(const LabelVolume& input, const std::array<int, 3>& target_shape) {
    std::array<double, 3> zoom;
    for (int i = 0; i < 3; ++i) {
        zoom[i] = static_cast<double>(target_shape[i]) / input.shape[i];
    }

    std::array<double, 3> new_spacing;
    for (int i = 0; i < 3; ++i) {
        new_spacing[i] = input.spacing[i] / zoom[i];
    }

    LabelVolume out;
    out.shape = target_shape;
    out.spacing = new_spacing;
    out.affine = scale_affine(input.affine, zoom);
    out.data = resample_nearest_label(input.data, input.shape, target_shape, zoom);
    return out;
}

} // namespace totalseg
