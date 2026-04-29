#pragma once
#include "ts_types.h"
#include <string>

namespace totalseg {

/// Load a NIfTI (.nii.gz) file into a float Volume.
Volume load_nifti(const std::string& path);

/// Load a NIfTI (.nii.gz) file into a LabelVolume (uint8).
LabelVolume load_nifti_label(const std::string& path);

/// Save a float Volume to NIfTI (.nii.gz).
void save_nifti(const Volume& vol, const std::string& path);

/// Save a LabelVolume to NIfTI (.nii.gz).
void save_nifti_label(const LabelVolume& vol, const std::string& path);

} // namespace totalseg
