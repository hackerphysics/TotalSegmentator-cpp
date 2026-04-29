cmake_minimum_required(VERSION 3.16)
project(TotalSegmentatorCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── ONNX Runtime ──
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)

# ── nifticlib (system, manual) ──
set(NIFTI_INCLUDE_DIRS /usr/include/nifti)
set(NIFTI_LIBRARIES nifti2 znz)

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
