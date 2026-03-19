cmake_minimum_required(VERSION 3.18)

# Set CUDA paths
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

project(beaver_gpu LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Set CUDA architectures - use more compatible settings
# RTX 4090 is compute capability 8.9, but let's also support older cards
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Include directories
include_directories(include)
include_directories(/usr/local/cuda/include)

# Source files
file(GLOB_RECURSE SOURCES "src/core/*.cu" "src/core/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.cuh")

message(STATUS "Source files: ${SOURCES}")
message(STATUS "Header files: ${HEADERS}")

# Create main library
add_library(beaver_gpu_lib ${SOURCES} ${HEADERS})

# Set properties for CUDA compilation
set_target_properties(beaver_gpu_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries and libpmem
target_link_libraries(beaver_gpu_lib
    CUDA::cudart
    CUDA::cuda_driver
    pmem
    pci
)

# Test executables
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Compiler flags
target_compile_options(beaver_gpu_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        -Xcompiler -fPIC
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fPIC
    >
)