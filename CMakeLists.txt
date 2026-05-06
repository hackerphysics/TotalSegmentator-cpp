cmake_minimum_required(VERSION 3.16)
project(TotalSegmentatorCpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32 AND MSVC)
    if(CMAKE_VERSION VERSION_LESS 3.21)
        message(FATAL_ERROR "Windows runtime DLL packaging requires CMake 3.21 or newer")
    endif()

    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
    include(InstallRequiredSystemLibraries)
endif()

# ── ONNX Runtime ──
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_DIR}/lib" REQUIRED)
if(WIN32)
    find_file(ONNXRUNTIME_DLL onnxruntime.dll PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
    find_file(ONNXRUNTIME_PROVIDERS_SHARED_DLL onnxruntime_providers_shared.dll PATHS "${ONNXRUNTIME_DIR}/lib" NO_DEFAULT_PATH)
endif()

# ── nifticlib ──
if(WIN32)
    find_package(NIFTI REQUIRED CONFIG)
    set(NIFTI_LIBRARIES NIFTI::nifti2 NIFTI::znz)
else()
    find_path(NIFTI_INCLUDE_DIRS nifti2_io.h
        PATH_SUFFIXES nifti
        PATHS /usr/include /usr/local/include
    )
    find_library(NIFTI2_LIB NAMES nifti2)
    find_library(ZNZ_LIB NAMES znz)
    if(NIFTI2_LIB AND ZNZ_LIB)
        set(NIFTI_LIBRARIES ${NIFTI2_LIB} ${ZNZ_LIB})
    else()
        find_package(NIFTI QUIET CONFIG)
        if(NIFTI_FOUND)
            set(NIFTI_LIBRARIES NIFTI::nifti2 NIFTI::znz)
        else()
            message(FATAL_ERROR "NIfTI library not found")
        endif()
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

if(WIN32)
    set_target_properties(totalseg PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

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

add_executable(generate_smoke_input tests/generate_smoke_input.cpp)
target_link_libraries(generate_smoke_input PRIVATE totalseg)

add_executable(inspect_label_volume tests/inspect_label_volume.cpp)
target_link_libraries(inspect_label_volume PRIVATE totalseg)

# ── CLI executable ──
add_executable(totalseg_cli src/main.cpp)
target_link_libraries(totalseg_cli PRIVATE totalseg)
target_include_directories(totalseg_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(WIN32)
    add_custom_command(TARGET totalseg_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:totalseg_cli>
            $<TARGET_FILE_DIR:totalseg_cli>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying Windows runtime DLLs next to totalseg_cli"
    )

    add_custom_command(TARGET totalseg_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            $<TARGET_FILE_DIR:totalseg_cli>
        COMMENT "Copying ONNX Runtime DLL next to totalseg_cli"
    )

    if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
        add_custom_command(TARGET totalseg_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}"
                $<TARGET_FILE_DIR:totalseg_cli>
            COMMENT "Copying ONNX Runtime providers DLL next to totalseg_cli"
        )
    endif()

    if(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        add_custom_command(TARGET totalseg_cli POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
                $<TARGET_FILE_DIR:totalseg_cli>
            COMMENT "Copying MSVC runtime DLLs next to totalseg_cli"
        )
    endif()
endif()

# ── Install ──
if(WIN32)
    install(TARGETS totalseg totalseg_cli
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
    )
else()
    install(TARGETS totalseg LIBRARY DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include)

if(WIN32)
    install(FILES "${ONNXRUNTIME_DLL}" DESTINATION bin)
    if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
        install(FILES "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}" DESTINATION bin)
    endif()
    install(FILES $<TARGET_RUNTIME_DLLS:totalseg_cli> DESTINATION bin)
    install(DIRECTORY $<TARGET_FILE_DIR:totalseg_cli>/ DESTINATION bin FILES_MATCHING PATTERN "*.dll")
endif()
