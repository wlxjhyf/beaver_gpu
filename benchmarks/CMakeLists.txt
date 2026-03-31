# Benchmarks CMakeLists.txt

# ── perf_benchmark (Phase 5: COW vs Checkpoint analysis) ─────────────
add_executable(perf_benchmark
    perf_benchmark.cu
    ${CMAKE_SOURCE_DIR}/tests/ddio_helper.cpp)

target_link_libraries(perf_benchmark beaver_gpu_lib)

set_target_properties(perf_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ── microbench (GoFS §4.1/4.2 aligned: Basic / cuFile / beaver) ──────
add_executable(microbench
    microbench.cu
    baseline_basic.cu
    baseline_pmem.cu
    baseline_cufile.cu
    baseline_beaver.cu
    ${CMAKE_SOURCE_DIR}/tests/ddio_helper.cpp)

target_link_libraries(microbench beaver_gpu_lib pmem)

# Link cufile if available (GDS / GPU Direct Storage)
find_library(CUFILE_LIB cufile HINTS /usr/local/cuda/lib64)
if(CUFILE_LIB)
    target_link_libraries(microbench ${CUFILE_LIB})
    message(STATUS "cuFile found: ${CUFILE_LIB}")
else()
    message(STATUS "cuFile NOT found — cuFile baseline will report N/A at runtime")
endif()

set_target_properties(microbench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Allow overriding MB_BEAVER_WARPS at configure time:
#   cmake -DMB_BEAVER_WARPS=4 ..
if(DEFINED MB_BEAVER_WARPS)
    target_compile_definitions(microbench PRIVATE MB_BEAVER_WARPS=${MB_BEAVER_WARPS})
    message(STATUS "microbench: MB_BEAVER_WARPS=${MB_BEAVER_WARPS}")
endif()

# ── e2_prefetch_bench (E2: DRAM prefetch validation & perf) ──────────
add_executable(e2_prefetch_bench
    e2_prefetch_bench.cu
    ${CMAKE_SOURCE_DIR}/tests/ddio_helper.cpp)

target_link_libraries(e2_prefetch_bench beaver_gpu_lib)

set_target_properties(e2_prefetch_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# ── rawgpm_bench (Phase A: raw GPU→PM bandwidth ceiling) ─────────────
add_executable(rawgpm_bench
    rawgpm_bench.cu
    ${CMAKE_SOURCE_DIR}/tests/ddio_helper.cpp)

target_link_libraries(rawgpm_bench beaver_gpu_lib)

set_target_properties(rawgpm_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
