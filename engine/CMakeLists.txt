cmake_minimum_required(VERSION 3.18)
project(jetson_llm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Jetson Orin = sm_87 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Find Python for pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pybind11 (fetch if not found)
find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find CUDA toolkit include dir
find_package(CUDAToolkit REQUIRED)

# Build the CUDA kernels as a static library
add_library(kernels STATIC kernels.cu nf4_gemv_fast.cu)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Build the Python module
pybind11_add_module(jetson_engine engine_py.cpp engine.cpp weights.cpp)
target_link_libraries(jetson_engine PRIVATE kernels CUDA::cudart CUDA::cublas)
target_include_directories(jetson_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
# Required for CUDA separable compilation to resolve device symbols in the final .so
set_target_properties(jetson_engine PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Install to workspace for easy import
install(TARGETS jetson_engine LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
