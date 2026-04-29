#include "totalseg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>

static const std::string GT = "tests/ground_truth/";

template<typename T>
std::vector<T> load_bin(const std::string& path, size_t count) {
    std::vector<T> v(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    return v;
}

int main() {
    std::cout << "=== Resample-Back Label Test ===\n";

    // Load hires seg
    size_t hires_nvox = 244 * 202 * 224;
    auto hires_data = load_bin<uint8_t>(GT + "py_seg_hires.bin", hires_nvox);

    totalseg::LabelVolume hires;
    hires.shape = {244, 202, 224};
    hires.spacing = {1.5, 1.5, 1.5};
    hires.data = std::move(hires_data);

    // Resample back to 122x101x112
    auto result = totalseg::resample_to_shape_label(hires, {122, 101, 112});
    std::cout << "Result shape: " << result.shape[0] << "x" << result.shape[1] << "x" << result.shape[2] << "\n";

    // Compare with Python resample-back
    size_t nvox = 122 * 101 * 112;
    auto py_back = load_bin<uint8_t>(GT + "py_seg_resample_back.bin", nvox);

    size_t match = 0;
    for (size_t i = 0; i < nvox; ++i) {
        if (result.data[i] == py_back[i]) match++;
    }
    std::cout << "Match: " << match << "/" << nvox << " = " << 100.0*match/nvox << "%\n";

    // Show first mismatches
    int shown = 0;
    for (size_t i = 0; i < nvox && shown < 10; ++i) {
        if (result.data[i] != py_back[i]) {
            int z = i % 112;
            int y = (i / 112) % 101;
            int x = i / (101 * 112);
            std::cout << "  [" << x << "," << y << "," << z << "]: cpp=" << (int)result.data[i]
                      << " py=" << (int)py_back[i] << "\n";
            shown++;
        }
    }
    return 0;
}
