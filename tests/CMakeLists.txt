# Tests CMakeLists.txt

add_executable(gpm_direct_pm_test gpm_direct_pm_test.cu)
target_link_libraries(gpm_direct_pm_test beaver_gpu_lib)
set_target_properties(gpm_direct_pm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(cow_kernel_test cow_kernel_test.cu ddio_helper.cpp)
target_link_libraries(cow_kernel_test beaver_gpu_lib)
set_target_properties(cow_kernel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(f2fs_kernel_test f2fs_kernel_test.cu ddio_helper.cpp)
target_link_libraries(f2fs_kernel_test beaver_gpu_lib)
set_target_properties(f2fs_kernel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
