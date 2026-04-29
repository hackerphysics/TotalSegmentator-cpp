#include "totalseg.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: totalseg_cli <input.nii.gz> <output.nii.gz> <model_dir>\n";
        return 1;
    }
    totalseg::PipelineConfig config;
    config.input_path = argv[1];
    config.output_path = argv[2];
    config.model_dir = argv[3];
    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fast") config.fast = true;
        if (arg == "--gpu") config.use_gpu = true;
    }
    auto result = totalseg::run_pipeline(config);
    std::cout << "Done. Output shape: " << result.shape[0] << "x" << result.shape[1] << "x" << result.shape[2] << "\n";
    return 0;
}
