// B-spline prefilter for scipy-compatible order=3 resampling
// Ported from scipy/ndimage/src/ni_splines.c
// Outputs double-precision coefficients to match scipy's float64 internal pipeline.

#include "ts_bspline_prefilter.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace totalseg {

static constexpr double BSPLINE3_POLE = -0.267949192431122706472553658494127633;

static void init_causal_reflect(double* c, int n, double z) {
    double z_i = z;
    double z_n = std::pow(z, n);
    double sum = c[0] + z_n * c[n - 1];
    for (int i = 1; i < n; ++i) {
        sum += z_i * (c[i] + z_n * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] += sum * z / (1.0 - z_n * z_n);
}

static void init_anticausal_reflect(double* c, int n, double z) {
    c[n - 1] *= z / (z - 1.0);
}

static void filter_1d_inplace(double* c, int n, double pole) {
    if (n < 2) return;
    double gain = (1.0 - pole) * (1.0 - 1.0 / pole);
    for (int i = 0; i < n; ++i) c[i] *= gain;
    init_causal_reflect(c, n, pole);
    for (int i = 1; i < n; ++i) c[i] += pole * c[i - 1];
    init_anticausal_reflect(c, n, pole);
    for (int i = n - 2; i >= 0; --i) c[i] = pole * (c[i + 1] - c[i]);
}

void bspline_prefilter_3d(const std::vector<float>& data, const std::array<int, 3>& shape,
                          std::vector<double>& coeffs) {
    int nx = shape[0], ny = shape[1], nz = shape[2];
    size_t total = static_cast<size_t>(nx) * ny * nz;
    size_t sy = static_cast<size_t>(ny) * nz;
    double pole = BSPLINE3_POLE;

    // Convert to double
    coeffs.resize(total);
    for (size_t i = 0; i < total; ++i) coeffs[i] = data[i];

    // Filter along z (contiguous)
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            filter_1d_inplace(&coeffs[x * sy + y * nz], nz, pole);
        }
    }

    // Filter along y
    std::vector<double> line(std::max({nx, ny, nz}));
    for (int x = 0; x < nx; ++x) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) line[y] = coeffs[x * sy + y * nz + z];
            filter_1d_inplace(line.data(), ny, pole);
            for (int y = 0; y < ny; ++y) coeffs[x * sy + y * nz + z] = line[y];
        }
    }

    // Filter along x
    for (int y = 0; y < ny; ++y) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) line[x] = coeffs[x * sy + y * nz + z];
            filter_1d_inplace(line.data(), nx, pole);
            for (int x = 0; x < nx; ++x) coeffs[x * sy + y * nz + z] = line[x];
        }
    }
}

} // namespace totalseg
