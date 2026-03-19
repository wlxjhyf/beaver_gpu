#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "gpm_interface.cuh"

// Device-side atomic operations (inline in this file for now)
__device__ uint64_t gpm_atomic_read_u64(uint64_t* addr) {
    return atomicAdd((unsigned long long*)addr, 0ULL);
}

__device__ void gpm_atomic_write_u64(uint64_t* addr, uint64_t value) {
    atomicExch((unsigned long long*)addr, (unsigned long long)value);
}

__device__ uint64_t gpm_atomic_cas_u64(uint64_t* addr, uint64_t expected, uint64_t desired) {
    return atomicCAS((unsigned long long*)addr,
                     (unsigned long long)expected,
                     (unsigned long long)desired);
}

// Simple GPU kernel to test PM access
__global__ void test_pm_access(uint64_t* pm_data, int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_threads) {
        // Each thread writes its ID to a different location
        uint64_t value = (uint64_t)tid + 1000;
        gpm_atomic_write_u64(&pm_data[tid], value);

        // Read back and verify
        uint64_t read_value = gpm_atomic_read_u64(&pm_data[tid]);
        if (read_value != value) {
            printf("Thread %d: Write/Read mismatch! Expected %llu, got %llu\n",
                   tid, value, read_value);
        }
    }
}

int main() {
    printf("=== Beaver GPU: GPM Hello World Test ===\n");

    // Initialize GPM
    gpm_error_t err = gpm_init();
    if (err != GPM_SUCCESS) {
        printf("Failed to initialize GPM: %d\n", err);
        return -1;
    }

    // Allocate PM region
    gpm_region_t region;
    const size_t data_size = sizeof(uint64_t) * 1024;
    err = gpm_alloc(&region, data_size, "hello");
    if (err != GPM_SUCCESS) {
        printf("Failed to allocate GPM region: %d\n", err);
        gpm_cleanup();
        return -1;
    }

    printf("Allocated %zu bytes of GPU-accessible persistent memory\n", data_size);
    printf("PM addr (host+device via UVA): %p\n", region.addr);

    // Launch GPU kernel
    const int num_threads = 1024;
    const int threads_per_block = 256;
    const int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    printf("Launching %d threads in %d blocks...\n", num_threads, num_blocks);

    uint64_t* pm_data = (uint64_t*)region.addr;
    test_pm_access<<<num_blocks, threads_per_block>>>(pm_data, num_threads);

    // Wait for completion
    cudaError_t cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(cuda_err));
        gpm_free(&region);
        gpm_cleanup();
        return -1;
    }

    printf("GPU kernel completed successfully\n");

    // Persist data: on host side use pmem_persist (gpm_persist is __device__ only)
    if (region.is_pmem)
        pmem_persist(region.addr, data_size);
    else
        pmem_msync(region.addr, data_size);
    printf("Data persisted to PM successfully\n");

    // Verify data on host
    printf("Verifying data on host...\n");
    int errors = 0;
    for (int i = 0; i < num_threads; i++) {
        uint64_t expected = i + 1000;
        if (pm_data[i] != expected) {
            printf("Host verification error at index %d: expected %llu, got %llu\n",
                   i, expected, pm_data[i]);
            errors++;
            if (errors > 10) {
                printf("Too many errors, stopping verification\n");
                break;
            }
        }
    }

    if (errors == 0) {
        printf("✅ All data verified successfully!\n");
        printf("✅ GPU can directly access persistent memory\n");
        printf("✅ Atomic operations working correctly\n");
    } else {
        printf("❌ Found %d verification errors\n", errors);
    }

    // Cleanup
    gpm_free(&region);
    gpm_cleanup();

    printf("=== GPM Hello World Test Complete ===\n");
    return errors == 0 ? 0 : -1;
}