#include "ts_nifti_io.h"
#include <cmath>

int main(int argc, char* argv[]) {
    const char* output = argc > 1 ? argv[1] : "tests/smoke_ct.nii.gz";

    totalseg::Volume vol;
    vol.shape = {32, 32, 32};
    vol.spacing = {6.0, 6.0, 6.0};
    vol.affine = {{{6.0, 0.0, 0.0, 0.0},
                   {0.0, 6.0, 0.0, 0.0},
                   {0.0, 0.0, 6.0, 0.0},
                   {0.0, 0.0, 0.0, 1.0}}};
    vol.data.resize(vol.numel(), -1000.0f);

    const float cx = 15.5f;
    const float cy = 15.5f;
    const float cz = 15.5f;
    for (int x = 0; x < vol.shape[0]; ++x) {
        for (int y = 0; y < vol.shape[1]; ++y) {
            for (int z = 0; z < vol.shape[2]; ++z) {
                float dx = x - cx;
                float dy = y - cy;
                float dz = z - cz;
                float r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 11.0f) {
                    vol.at(x, y, z) = 40.0f;
                }
                if (r < 4.0f) {
                    vol.at(x, y, z) = 150.0f;
                }
            }
        }
    }

    totalseg::save_nifti(vol, output);
    return 0;
}
