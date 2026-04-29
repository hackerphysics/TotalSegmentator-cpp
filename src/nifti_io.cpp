#include "ts_nifti_io.h"
#include <nifti2_io.h>
#include <stdexcept>
#include <cstring>

namespace totalseg {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void extract_affine(nifti_image* nim, std::array<std::array<double,4>,4>& aff) {
    // nifti2 uses nifti_dmat44 (double), not mat44 (float)
    if (nim->sform_code > 0) {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                aff[r][c] = nim->sto_xyz.m[r][c];
    } else if (nim->qform_code > 0) {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                aff[r][c] = nim->qto_xyz.m[r][c];
    } else {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                aff[r][c] = 0.0;
        aff[0][0] = nim->dx;
        aff[1][1] = nim->dy;
        aff[2][2] = nim->dz;
        aff[3][3] = 1.0;
    }
}

static void set_affine(nifti_image* nim, const std::array<std::array<double,4>,4>& aff) {
    // Set sform directly using nifti_dmat44
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            nim->sto_xyz.m[r][c] = aff[r][c];
    nim->sform_code = NIFTI_XFORM_SCANNER_ANAT;
    nim->sto_ijk = nifti_dmat44_inverse(nim->sto_xyz);
    // Also set qform
    nim->qform_code = NIFTI_XFORM_SCANNER_ANAT;
    nim->qto_xyz = nim->sto_xyz;
    nim->qto_ijk = nim->sto_ijk;
    // Set quaternion params from the matrix
    nifti_dmat44_to_quatern(nim->qto_xyz,
        &nim->quatern_b, &nim->quatern_c, &nim->quatern_d,
        &nim->qoffset_x, &nim->qoffset_y, &nim->qoffset_z,
        nullptr, nullptr, nullptr, &nim->qfac);
}

// Convert arbitrary voxel data to float, returning a newly allocated float array.
static std::vector<float> convert_to_float(const void* data, size_t nvox, int datatype) {
    std::vector<float> out(nvox);
    switch (datatype) {
        case DT_UINT8: {
            auto* p = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_INT16: {
            auto* p = static_cast<const int16_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_INT32: {
            auto* p = static_cast<const int32_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_FLOAT32: {
            auto* p = static_cast<const float*>(data);
            std::memcpy(out.data(), p, nvox * sizeof(float));
            break;
        }
        case DT_FLOAT64: {
            auto* p = static_cast<const double*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_UINT16: {
            auto* p = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_UINT32: {
            auto* p = static_cast<const uint32_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_INT8: {
            auto* p = static_cast<const int8_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DT_INT64: {
            auto* p = static_cast<const int64_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        default:
            throw std::runtime_error("nifti_io: unsupported NIfTI datatype " + std::to_string(datatype));
    }
    return out;
}

static std::vector<uint8_t> convert_to_uint8(const void* data, size_t nvox, int datatype) {
    std::vector<uint8_t> out(nvox);
    switch (datatype) {
        case DT_UINT8: {
            std::memcpy(out.data(), data, nvox);
            break;
        }
        case DT_INT16: {
            auto* p = static_cast<const int16_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<uint8_t>(p[i]);
            break;
        }
        case DT_INT32: {
            auto* p = static_cast<const int32_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<uint8_t>(p[i]);
            break;
        }
        case DT_FLOAT32: {
            auto* p = static_cast<const float*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<uint8_t>(p[i]);
            break;
        }
        case DT_FLOAT64: {
            auto* p = static_cast<const double*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<uint8_t>(p[i]);
            break;
        }
        case DT_UINT16: {
            auto* p = static_cast<const uint16_t*>(data);
            for (size_t i = 0; i < nvox; ++i) out[i] = static_cast<uint8_t>(p[i]);
            break;
        }
        default:
            throw std::runtime_error("nifti_io: unsupported NIfTI datatype " + std::to_string(datatype) + " for label volume");
    }
    return out;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

Volume load_nifti(const std::string& path) {
    nifti_image* nim = nifti_image_read(path.c_str(), 1 /* read data */);
    if (!nim)
        throw std::runtime_error("nifti_io: failed to read " + path);

    Volume vol;
    vol.shape = {static_cast<int>(nim->nx),
                 static_cast<int>(nim->ny),
                 static_cast<int>(nim->nz)};
    vol.spacing = {nim->dx, nim->dy, nim->dz};
    extract_affine(nim, vol.affine);

    size_t nvox = vol.numel();
    auto raw = convert_to_float(nim->data, nvox, nim->datatype);

    // Transpose from NIfTI Fortran order (x-fastest) to C order (x-slowest)
    int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2];
    vol.data.resize(nvox);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                vol.data[(size_t)x * ny * nz + y * nz + z] = raw[(size_t)z * ny * nx + y * nx + x];

    // Apply scl_slope / scl_inter if set and valid (not NaN).
    float slope = nim->scl_slope;
    float inter = nim->scl_inter;
    if (!std::isnan(slope) && slope != 0.0f && !(slope == 1.0f && inter == 0.0f)) {
        for (size_t i = 0; i < nvox; ++i)
            vol.data[i] = vol.data[i] * slope + inter;
    }

    nifti_image_free(nim);
    return vol;
}

LabelVolume load_nifti_label(const std::string& path) {
    nifti_image* nim = nifti_image_read(path.c_str(), 1);
    if (!nim)
        throw std::runtime_error("nifti_io: failed to read " + path);

    LabelVolume vol;
    vol.shape = {static_cast<int>(nim->nx),
                 static_cast<int>(nim->ny),
                 static_cast<int>(nim->nz)};
    vol.spacing = {nim->dx, nim->dy, nim->dz};
    extract_affine(nim, vol.affine);

    auto raw = convert_to_uint8(nim->data, vol.numel(), nim->datatype);

    // Transpose from NIfTI Fortran order to C order
    int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2];
    size_t nvox = vol.numel();
    vol.data.resize(nvox);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                vol.data[(size_t)x * ny * nz + y * nz + z] = raw[(size_t)z * ny * nx + y * nx + x];

    nifti_image_free(nim);
    return vol;
}

void save_nifti(const Volume& vol, const std::string& path) {
    nifti_image* nim = nifti_simple_init_nim();
    if (!nim)
        throw std::runtime_error("nifti_io: failed to allocate nifti_image");

    nim->ndim = 3;
    nim->dim[0] = 3;
    nim->nx = nim->dim[1] = vol.shape[0];
    nim->ny = nim->dim[2] = vol.shape[1];
    nim->nz = nim->dim[3] = vol.shape[2];
    nim->nt = nim->dim[4] = 1;
    nim->nu = nim->dim[5] = 1;
    nim->nv = nim->dim[6] = 1;
    nim->nw = nim->dim[7] = 1;

    nim->dx = nim->pixdim[1] = static_cast<float>(vol.spacing[0]);
    nim->dy = nim->pixdim[2] = static_cast<float>(vol.spacing[1]);
    nim->dz = nim->pixdim[3] = static_cast<float>(vol.spacing[2]);

    nim->datatype = DT_FLOAT32;
    nim->nbyper = sizeof(float);
    nim->nvox = vol.numel();

    set_affine(nim, vol.affine);

    // Transpose from C order back to NIfTI Fortran order for writing
    int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2];
    nim->data = std::malloc(nim->nvox * nim->nbyper);
    if (!nim->data)
        throw std::runtime_error("nifti_io: malloc failed");
    float* dst = static_cast<float*>(nim->data);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                dst[(size_t)z * ny * nx + y * nx + x] = vol.data[(size_t)x * ny * nz + y * nz + z];

    nim->fname = nifti_strdup(path.c_str());
    nim->iname = nifti_strdup(path.c_str());
    nim->nifti_type = NIFTI_FTYPE_NIFTI1_1; // single .nii(.gz)

    nifti_set_iname_offset(nim, 2);
    nifti_image_write(nim);
    nifti_image_free(nim);
}

void save_nifti_label(const LabelVolume& vol, const std::string& path) {
    nifti_image* nim = nifti_simple_init_nim();
    if (!nim)
        throw std::runtime_error("nifti_io: failed to allocate nifti_image");

    nim->ndim = 3;
    nim->dim[0] = 3;
    nim->nx = nim->dim[1] = vol.shape[0];
    nim->ny = nim->dim[2] = vol.shape[1];
    nim->nz = nim->dim[3] = vol.shape[2];
    nim->nt = nim->dim[4] = 1;
    nim->nu = nim->dim[5] = 1;
    nim->nv = nim->dim[6] = 1;
    nim->nw = nim->dim[7] = 1;

    nim->dx = nim->pixdim[1] = static_cast<float>(vol.spacing[0]);
    nim->dy = nim->pixdim[2] = static_cast<float>(vol.spacing[1]);
    nim->dz = nim->pixdim[3] = static_cast<float>(vol.spacing[2]);

    nim->datatype = DT_UINT8;
    nim->nbyper = sizeof(uint8_t);
    nim->nvox = vol.numel();

    set_affine(nim, vol.affine);

    // Transpose from C order to NIfTI Fortran order
    int nx = vol.shape[0], ny = vol.shape[1], nz = vol.shape[2];
    nim->data = std::malloc(nim->nvox * nim->nbyper);
    if (!nim->data)
        throw std::runtime_error("nifti_io: malloc failed");
    uint8_t* dst = static_cast<uint8_t*>(nim->data);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z)
                dst[(size_t)z * ny * nx + y * nx + x] = vol.data[(size_t)x * ny * nz + y * nz + z];

    nim->fname = nifti_strdup(path.c_str());
    nim->iname = nifti_strdup(path.c_str());
    nim->nifti_type = NIFTI_FTYPE_NIFTI1_1;

    nifti_set_iname_offset(nim, 2);
    nifti_image_write(nim);
    nifti_image_free(nim);
}

} // namespace totalseg
