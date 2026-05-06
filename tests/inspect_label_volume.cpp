#include "ts_nifti_io.h"
#include <array>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inspect_label_volume <label.nii.gz>\n";
        return 1;
    }

    auto vol = totalseg::load_nifti_label(argv[1]);
    std::array<size_t, 256> counts{};
    for (uint8_t value : vol.data) {
        counts[value]++;
    }

    size_t nonzero = 0;
    size_t labels = 0;
    for (size_t i = 1; i < counts.size(); ++i) {
        if (counts[i] > 0) {
            labels++;
            nonzero += counts[i];
        }
    }

    std::cout << "shape=" << vol.shape[0] << "x" << vol.shape[1] << "x" << vol.shape[2]
              << " labels=" << labels
              << " nonzero_voxels=" << nonzero << "\n";
    return nonzero > 0 ? 0 : 2;
}
