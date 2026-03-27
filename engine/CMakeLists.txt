cmake_minimum_required(VERSION 3.18)
project(jetson_llm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# GPU architecture (default: Jetson Orin sm_87, override with -DCMAKE_CUDA_ARCHITECTURES=89 for RTX)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# Fast math for CUDA kernels (~5% speedup from faster intrinsics)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

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
    POSITION_INDEPENDENT_CODE ON
)

# Build the Python module
pybind11_add_module(jetson_engine engine_py.cpp engine.cpp weights.cpp)
target_link_libraries(jetson_engine PRIVATE kernels CUDA::cudart CUDA::cublas)
target_include_directories(jetson_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

# Install to workspace for easy import
install(TARGETS jetson_engine LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
