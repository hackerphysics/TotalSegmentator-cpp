cmake_minimum_required(VERSION 3.16)
project(TotalSegmentatorCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── ONNX Runtime ──
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)

# ── nifticlib ──
find_path(NIFTI_INCLUDE_DIRS nifti2_io.h
    PATH_SUFFIXES nifti
    PATHS /usr/include /usr/local/include
)
find_library(NIFTI2_LIB NAMES nifti2)
find_library(ZNZ_LIB NAMES znz)
if(NIFTI2_LIB AND ZNZ_LIB)
    set(NIFTI_LIBRARIES ${NIFTI2_LIB} ${ZNZ_LIB})
else()
    # vcpkg or other package managers
    find_package(NIFTI QUIET CONFIG)
    if(NIFTI_FOUND)
        set(NIFTI_LIBRARIES NIFTI::nifti2 NIFTI::znz)
    else()
        message(FATAL_ERROR "NIfTI library not found")
    endif()
endif()

# ── zlib ──
find_package(ZLIB REQUIRED)

# ── Sources ──
set(SOURCES
    src/nifti_io.cpp
    src/alignment.cpp
    src/resampling.cpp
    src/bspline_prefilter.cpp
    src/cropping.cpp
    src/postprocessing.cpp
    src/sliding_window.cpp
    src/onnx_inference.cpp
    src/pipeline.cpp
    src/label_map.cpp
)

# ── Shared library (.so / .dll) ──
add_library(totalseg SHARED ${SOURCES})

target_include_directories(totalseg PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_DIR}/include
    ${NIFTI_INCLUDE_DIRS}
)

target_link_libraries(totalseg PUBLIC
    ${ONNXRUNTIME_LIB}
    ${NIFTI_LIBRARIES}
    ZLIB::ZLIB
)

# ── Test executable ──
add_executable(totalseg_test tests/main_test.cpp)
target_link_libraries(totalseg_test PRIVATE totalseg)

add_executable(totalseg_e2e tests/e2e_test.cpp)
target_link_libraries(totalseg_e2e PRIVATE totalseg)

add_executable(totalseg_iso tests/isolated_infer_test.cpp)
target_link_libraries(totalseg_iso PRIVATE totalseg)

add_executable(totalseg_spatch tests/single_patch_test.cpp)
target_link_libraries(totalseg_spatch PRIVATE totalseg)

add_executable(totalseg_hires tests/hires_compare.cpp)
target_link_libraries(totalseg_hires PRIVATE totalseg)
target_include_directories(totalseg_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── CLI executable ──
add_executable(totalseg_cli src/main.cpp)
target_link_libraries(totalseg_cli PRIVATE totalseg)
target_include_directories(totalseg_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── Install ──
install(TARGETS totalseg LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
