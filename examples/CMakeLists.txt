# Examples CMakeLists.txt

# Phase 2: GPM interface validation
add_executable(gpm_hello_world gpm_hello_world.cu)
target_link_libraries(gpm_hello_world beaver_gpu_lib)
set_target_properties(gpm_hello_world PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# beaver_cow_test.cu: Phase 3 COW layer exploration (kept for reference)
# Not in build: superseded by tests/cow_kernel_test
