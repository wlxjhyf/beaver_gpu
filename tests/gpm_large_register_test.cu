/*
 * gpm_large_register_test.cu
 *
 * Finds the maximum cudaHostRegister size for devdax (VM_PFNMAP) memory
 * that produces valid GPU page-table mappings.
 *
 * For each test size:
 *   1. cudaHostRegister(devdax_base, size, cudaHostRegisterMapped)
 *   2. cudaHostGetDevicePointer → dev_ptr
 *   3. Launch a tiny GPU write kernel to a word at dev_ptr
 *   4. cudaDeviceSynchronize → check for illegal memory access
 *   5. CPU read-back to verify the value
 *   6. cudaHostUnregister
 *
 * Run without PyTorch to establish the "bare CUDA" baseline.
 * Then compare with Python/beaver_ext to determine if PyTorch context matters.
 *
 * Build: part of tests/CMakeLists.txt
 * Run:   sudo ./build/tests/gpm_large_register_test
 */

#include <cuda_runtime.h>
#include <libpmem.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define GPM_DEVDAX_PATH "/dev/dax1.0"
#define MAGIC 0xDEADBEEFCAFEBABEULL

/* ------------------------------------------------------------------ */
/* GPU kernel: write one uint64 word at dev_ptr[word_idx]              */
/* ------------------------------------------------------------------ */
__global__ void kernel_write_one(volatile uint64_t *dst, uint64_t val, size_t word_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[word_idx] = val;
        __threadfence_system();
    }
}

/* ------------------------------------------------------------------ */
/* GPU kernel: write a small pattern across the region to stress PTEs  */
/* stride = 2MB so we touch different PMD-level page table entries     */
/* ------------------------------------------------------------------ */
#define STRIDE_BYTES (2UL * 1024 * 1024)

__global__ void kernel_write_strided(volatile uint64_t *dst, size_t region_bytes)
{
    size_t stride_words = STRIDE_BYTES / sizeof(uint64_t);
    size_t n_strides    = region_bytes / STRIDE_BYTES;
    size_t tid          = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_strides) {
        dst[tid * stride_words] = (uint64_t)tid + MAGIC;
    }
    __threadfence_system();
}

/* ------------------------------------------------------------------ */
/* Helper                                                               */
/* ------------------------------------------------------------------ */
#define CUDA_OK(call) do {                                              \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "  CUDA error: %s\n", cudaGetErrorString(_e));\
        return _e;                                                     \
    }                                                                  \
} while(0)

static const char *human(size_t bytes)
{
    static char buf[64];
    if (bytes >= (1UL << 30))
        snprintf(buf, sizeof(buf), "%.1f GB", (double)bytes / (1UL << 30));
    else
        snprintf(buf, sizeof(buf), "%zu MB", bytes >> 20);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Single-size registration test                                        */
/* Returns 0 on success, non-zero on failure                           */
/* ------------------------------------------------------------------ */
static int test_register_size(void *devdax_base, size_t devdax_total,
                               size_t size)
{
    if (size > devdax_total) {
        printf("  [SKIP] %s > devdax total (%.1f GB)\n",
               human(size), (double)devdax_total / (1UL << 30));
        return 0;
    }

    printf("  Testing cudaHostRegister(%s) ... ", human(size));
    fflush(stdout);

    /* 1. Register */
    cudaError_t e = cudaHostRegister(devdax_base, size, cudaHostRegisterMapped);
    if (e != cudaSuccess) {
        printf("REGISTER FAILED: %s\n", cudaGetErrorString(e));
        return 1;
    }

    /* 2. Get device pointer */
    void *dev_ptr = NULL;
    e = cudaHostGetDevicePointer(&dev_ptr, devdax_base, 0);
    if (e != cudaSuccess) {
        printf("GetDevicePointer FAILED: %s\n", cudaGetErrorString(e));
        cudaHostUnregister(devdax_base);
        return 2;
    }

    /* 3. Write one word near the START of the region */
    kernel_write_one<<<1, 1>>>((volatile uint64_t *)dev_ptr, MAGIC, 0);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("KERNEL(start) FAILED: %s\n", cudaGetErrorString(e));
        cudaHostUnregister(devdax_base);
        return 3;
    }

    /* 4. Write one word near the END of the region */
    size_t end_word = (size / sizeof(uint64_t)) - 1;
    kernel_write_one<<<1, 1>>>((volatile uint64_t *)dev_ptr, MAGIC ^ end_word, end_word);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("KERNEL(end) FAILED: %s\n", cudaGetErrorString(e));
        cudaHostUnregister(devdax_base);
        return 4;
    }

    /* 5. Strided write to exercise many PTEs */
    size_t n_strides = size / STRIDE_BYTES;
    if (n_strides > 0) {
        int threads = 256;
        int blocks  = ((int)n_strides + threads - 1) / threads;
        kernel_write_strided<<<blocks, threads>>>((volatile uint64_t *)dev_ptr, size);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("KERNEL(strided) FAILED: %s\n", cudaGetErrorString(e));
            cudaHostUnregister(devdax_base);
            return 5;
        }
    }

    /* 6. CPU readback: spot-check start, end, and a few strides */
    volatile uint64_t *cpu_view = (volatile uint64_t *)devdax_base;

    int cpu_errors = 0;
    if (cpu_view[0] != MAGIC) {
        printf("\n    CPU readback mismatch at word 0: got %llx expected %llx",
               (unsigned long long)cpu_view[0], (unsigned long long)MAGIC);
        cpu_errors++;
    }
    if (cpu_view[end_word] != (MAGIC ^ end_word)) {
        printf("\n    CPU readback mismatch at word %zu", end_word);
        cpu_errors++;
    }
    /* spot check first few strided words */
    for (size_t s = 0; s < n_strides && s < 4; s++) {
        uint64_t exp = s + MAGIC;
        uint64_t got = cpu_view[s * (STRIDE_BYTES / sizeof(uint64_t))];
        if (got != exp) {
            printf("\n    CPU readback mismatch at stride %zu: got %llx exp %llx",
                   s, (unsigned long long)got, (unsigned long long)exp);
            cpu_errors++;
        }
    }

    cudaHostUnregister(devdax_base);

    if (cpu_errors) {
        printf("PASS (GPU write) but CPU readback MISMATCH (%d errors)\n", cpu_errors);
        return 6;
    }

    printf("PASS\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    /*
     * --simulate-torch : before testing, allocate 3 GB VRAM + 2.75 GB
     *   pinned mapped host memory to replicate what happens after PyTorch
     *   loads GPT-2 XL and beaver_ext_init allocates the dram_pool.
     */
    int simulate_torch = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--simulate-torch") == 0)
            simulate_torch = 1;
    }

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   GPU-PM Large cudaHostRegister Diagnostic Test      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);

    /* Warm up CUDA context */
    cudaFree(0);

    void  *vram_block  = NULL;
    void  *dram_pool   = NULL;

    if (simulate_torch) {
        printf("\n[simulate-torch] Pre-allocating resources to mimic PyTorch E2e context:\n");

        /* 1. Simulate GPT-2 XL model on GPU: ~3 GB VRAM */
        size_t vram_bytes = 3UL * 1024 * 1024 * 1024;
        cudaError_t ce = cudaMalloc(&vram_block, vram_bytes);
        if (ce != cudaSuccess) {
            fprintf(stderr, "  cudaMalloc(3GB VRAM) failed: %s\n",
                    cudaGetErrorString(ce));
            return 1;
        }
        printf("  cudaMalloc(3 GB VRAM): OK  ptr=%p\n", vram_block);

        /* 2. Simulate beaver_ext dram_pool: 2.75 GB pinned + mapped */
        size_t pool_bytes = (size_t)720880 * 4096; /* dram_pool_pages * PAGE_SIZE */
        ce = cudaHostAlloc(&dram_pool, pool_bytes,
                           cudaHostAllocPortable | cudaHostAllocMapped);
        if (ce != cudaSuccess) {
            fprintf(stderr, "  cudaHostAlloc(2.75GB dram_pool) failed: %s\n",
                    cudaGetErrorString(ce));
            cudaFree(vram_block);
            return 1;
        }
        printf("  cudaHostAlloc(%.2f GB, Portable|Mapped): OK  ptr=%p\n",
               (double)pool_bytes / (1UL << 30), dram_pool);
        printf("  Total pre-allocated: 3 GB VRAM + %.2f GB pinned\n\n",
               (double)pool_bytes / (1UL << 30));
    }

    /* Map devdax */
    size_t devdax_total = 0;
    int    is_pmem      = 0;
    void  *devdax_base  = pmem_map_file(GPM_DEVDAX_PATH, 0, 0, 0666,
                                        &devdax_total, &is_pmem);
    if (!devdax_base) {
        fprintf(stderr, "pmem_map_file(%s) failed: %s\n",
                GPM_DEVDAX_PATH, pmem_errormsg());
        return 1;
    }
    printf("devdax: %.1f GB at %p  is_pmem=%d\n\n",
           (double)devdax_total / (1UL << 30), devdax_base, is_pmem);

    /*
     * Test sizes — we use the SAME base pointer each time, unregistering
     * before the next attempt.
     */
    static const size_t test_sizes[] = {
        64UL   << 20,   /*   64 MB — known-working (T3 in gpm_direct_pm_test) */
        174UL  << 20,   /*  174 MB — benchmark upper bound (known-working)      */
        256UL  << 20,   /*  256 MB                                              */
        512UL  << 20,   /*  512 MB                                              */
        1UL    << 30,   /*    1 GB                                              */
        2UL    << 30,   /*    2 GB                                              */
        4UL    << 30,   /*    4 GB                                              */
        8UL    << 30,   /*    8 GB  (≈ E2e scenario: 8.25 GB)                  */
        16UL   << 30,   /*   16 GB                                              */
    };
    int n_sizes = (int)(sizeof(test_sizes) / sizeof(test_sizes[0]));

    int first_failure_idx = -1;

    printf("Probing cudaHostRegister sizes (same base ptr each time):\n");
    for (int i = 0; i < n_sizes; i++) {
        int rc = test_register_size(devdax_base, devdax_total, test_sizes[i]);
        if (rc != 0 && first_failure_idx < 0)
            first_failure_idx = i;
    }

    pmem_unmap(devdax_base, devdax_total);
    if (vram_block)  cudaFree(vram_block);
    if (dram_pool)   cudaFreeHost(dram_pool);

    printf("\n");
    if (first_failure_idx < 0) {
        printf("All sizes PASSED.\n");
        if (simulate_torch)
            printf("  → cudaHostRegister works even with 3GB VRAM + 2.75GB pinned pre-allocated.\n"
                   "  → Python/PyTorch context does NOT break cudaHostRegister.\n"
                   "  → Previous 'illegal memory access' was a BUG in the write kernel,\n"
                   "    NOT a cudaHostRegister size issue.\n"
                   "  → Next step: restore GPU direct write path in beaver_ext.cu.\n");
        else
            printf("  → cudaHostRegister is not the bottleneck in bare C context.\n"
                   "  → Run with --simulate-torch to test PyTorch-equivalent pre-allocations.\n");
    } else {
        printf("First failure at: %s\n", human(test_sizes[first_failure_idx]));
        if (first_failure_idx > 0)
            printf("Last success at : %s\n",
                   human(test_sizes[first_failure_idx - 1]));
        printf("\n");
        if (simulate_torch && first_failure_idx >= 0) {
            printf("  → Pre-allocation (VRAM + pinned dram_pool) DOES cause failure.\n");
            printf("  → Reducing dram_pool size or registering PM before dram_pool\n"
                   "    may fix the issue.\n");
        }
        printf("  dmesg check: sudo dmesg | grep -i 'foll\\|pfnmap\\|pin\\|nvidia'\n");
    }

    return (first_failure_idx >= 0) ? 1 : 0;
}
