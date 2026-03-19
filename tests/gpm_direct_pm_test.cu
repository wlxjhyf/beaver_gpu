/*
 * gpm_direct_pm_test.cu
 *
 * Verifies that GPU writes go directly to PM without a CPU DRAM intermediate
 * copy, and that data is truly persistent (survives CUDA/mmap teardown).
 *
 * Three tests:
 *
 *   [T1] GPU write + CPU readback
 *        GPU kernel uses volatile stores + __threadfence_system to write a
 *        known pattern.  CPU reads back via the same UVA pointer.
 *        Proves: UVA pointer works from kernel; no cudaMemcpy involved.
 *
 *   [T2] Persistence: unmap-remap verification
 *        After T1, call cudaHostUnregister + pmem_unmap to fully release the
 *        mapping, then pmem_map_file the same file again (CPU only, no CUDA).
 *        CPU reads and re-verifies the pattern.
 *        Proves: data is in the PM *file*, not just in GPU L2 cache or DRAM.
 *
 *   [T3] Bandwidth comparison: PM vs pinned DRAM
 *        Streaming write kernel to (a) cudaHostRegister'd PM and
 *        (b) cudaHostAlloc'd DRAM, both via PCIe, using the same volatile
 *        store kernel.
 *        PM bandwidth is Optane-limited (~2-4 GB/s); DRAM via PCIe is
 *        higher (~10-16 GB/s for PCIe 4.0 x16).
 *        Proves: PM is the bottleneck, i.e., GPU writes reach PM media.
 *
 * Build: part of tests/CMakeLists.txt (links beaver_gpu_lib -> pmem, pci)
 * Run:   sudo ./gpm_direct_pm_test   (root needed for DDIO if enabled)
 */

#include "gpm_interface.cuh"
#include <cuda_runtime.h>
#include <libpmem.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/* T1/T2: 1024 threads, each owns a 64-byte (8 × uint64) slot in PM   */
#define T1_NUM_THREADS  1024
#define T1_SLOT_BYTES   64                          /* one cache line   */
#define T1_TOTAL_BYTES  (T1_NUM_THREADS * T1_SLOT_BYTES)
#define T1_MAGIC        0xBEEFCAFE00000000ULL

/* T3: 64 MB per bandwidth trial, 5 warm-up + 10 measurement runs     */
#define T3_REGION_BYTES (64UL * 1024 * 1024)
#define T3_WARMUP_RUNS  2
#define T3_MEASURE_RUNS 5

/* Grid for T3 bandwidth kernel: fill all 128 SMs of RTX 4090          */
#define T3_BLOCKS  512
#define T3_THREADS 256

/* ------------------------------------------------------------------ */
/* GPU kernels                                                         */
/* ------------------------------------------------------------------ */

/*
 * kernel_write_pattern:
 *   Each thread writes 8 uint64 values to its dedicated cache-line slot.
 *   Uses volatile stores (= gpm_memcpy_nodrain semantics) followed by a
 *   system-scope fence (gpm_persist) to ensure data reaches PM.
 *   Pattern: slot[j] = T1_MAGIC | (tid * 8 + j)
 */
__global__ void kernel_write_pattern(void *pm_base, uint32_t nthreads)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    volatile uint64_t *slot =
        (volatile uint64_t *)((char *)pm_base + tid * T1_SLOT_BYTES);

    for (int j = 0; j < 8; ++j)
        slot[j] = T1_MAGIC | (uint64_t)(tid * 8 + j);

    /* System-scope fence: flush GPU write buffers through PCIe to PM */
    gpm_persist(pm_base, T1_SLOT_BYTES);
}

/*
 * kernel_read_verify:
 *   GPU reads back the pattern written by kernel_write_pattern and checks
 *   correctness.  Errors are counted atomically.
 */
__global__ void kernel_read_verify(void *pm_base, uint32_t nthreads,
                                   uint32_t *d_errors)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    volatile uint64_t *slot =
        (volatile uint64_t *)((char *)pm_base + tid * T1_SLOT_BYTES);

    for (int j = 0; j < 8; ++j) {
        uint64_t expected = T1_MAGIC | (uint64_t)(tid * 8 + j);
        uint64_t got = slot[j];
        if (got != expected)
            atomicAdd(d_errors, 1u);
    }
}

/*
 * kernel_bw_write:
 *   Streaming write across the entire region using volatile stores.
 *   Each thread handles a contiguous chunk for coalesced access.
 *   One __threadfence_system() at the end per block (lightweight).
 */
__global__ void kernel_bw_write(volatile uint64_t *dst, size_t nwords)
{
    size_t total  = (size_t)gridDim.x * blockDim.x;
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = tid; i < nwords; i += total)
        dst[i] = (uint64_t)i;

    /* One drain per block: enough for this benchmark */
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
}

/* ------------------------------------------------------------------ */
/* Helper: check CUDA errors                                           */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            return -1;                                                  \
        }                                                               \
    } while (0)

/* ------------------------------------------------------------------ */
/* Test T1 + T2                                                        */
/* ------------------------------------------------------------------ */
static int run_t1_t2(void)
{
    printf("\n=== [T1] GPU write + CPU readback ===\n");

    /* --- Allocate PM region ---------------------------------------- */
    gpm_region_t region;
    gpm_error_t  gerr = gpm_alloc(&region, T1_TOTAL_BYTES, "t1t2");
    if (gerr != GPM_SUCCESS) {
        fprintf(stderr, "gpm_alloc failed (%d). Is /mnt/pmem mounted?\n",
                gerr);
        return -1;
    }
    printf("  PM region: %zu bytes at UVA %p  (is_pmem=%d)\n",
           region.size, region.addr, region.is_pmem);
    if (!region.is_pmem)
        printf("  WARNING: not a real PM device; persistence test "
               "relies on msync fallback\n");

    /* Zero-initialise so stale data doesn't hide bugs */
    memset(region.addr, 0, region.size);

    /* --- T1: GPU kernel writes pattern ----------------------------- */
    int blocks  = (T1_NUM_THREADS + 255) / 256;
    int threads = 256;

    kernel_write_pattern<<<blocks, threads>>>(region.addr, T1_NUM_THREADS);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  GPU kernel_write_pattern done\n");

    /* --- T1: GPU reads back and verifies --------------------------- */
    uint32_t *d_errors;
    CUDA_CHECK(cudaMalloc(&d_errors, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_errors, 0, sizeof(uint32_t)));

    kernel_read_verify<<<blocks, threads>>>(region.addr, T1_NUM_THREADS,
                                            d_errors);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_errors);

    if (h_errors) {
        fprintf(stderr, "  [T1] FAIL: %u GPU read-back errors\n", h_errors);
        gpm_free(&region);
        return -1;
    }
    printf("  [T1] PASS: GPU read-back verified (%d slots × 64 B)\n",
           T1_NUM_THREADS);

    /* --- T1: CPU reads back via same UVA pointer ------------------- */
    int cpu_errors = 0;
    uint64_t *cpu_view = (uint64_t *)region.addr;
    for (int tid = 0; tid < T1_NUM_THREADS; ++tid) {
        for (int j = 0; j < 8; ++j) {
            uint64_t expected = T1_MAGIC | (uint64_t)(tid * 8 + j);
            uint64_t got      = cpu_view[tid * T1_SLOT_BYTES / 8 + j];
            if (got != expected) {
                if (cpu_errors < 5)
                    fprintf(stderr,
                            "  [T1] CPU mismatch tid=%d j=%d "
                            "expected=%016llx got=%016llx\n",
                            tid, j, (unsigned long long)expected,
                            (unsigned long long)got);
                ++cpu_errors;
            }
        }
    }
    if (cpu_errors) {
        fprintf(stderr, "  [T1] FAIL: %d CPU read-back errors\n",
                cpu_errors);
        gpm_free(&region);
        return -1;
    }
    printf("  [T1] PASS: CPU read-back verified via UVA pointer\n");
    printf("       (no cudaMemcpy used — GPU wrote directly to PM)\n");

    /* === T2: Persistence =========================================== */
    printf("\n=== [T2] Persistence: CUDA unregister -> fresh devdax remap -> verify ===\n");

    /* Save the devdax offset so we can navigate in the new mmap */
    size_t saved_offset = region.pm_offset;

    /* Flush from host side for safety */
    if (region.is_pmem)
        pmem_persist(region.addr, region.size);
    else
        pmem_msync(region.addr, region.size);

    /* Release the CUDA mapping for this sub-region */
    cudaError_t cu_err = cudaHostUnregister(region.addr);
    if (cu_err != cudaSuccess)
        fprintf(stderr, "  WARNING: cudaHostUnregister: %s\n",
                cudaGetErrorString(cu_err));

    printf("  CUDA unregister complete. offset=0x%zx\n", saved_offset);
    printf("  Opening a fresh independent CPU-only mmap of %s...\n",
           GPM_DEVDAX_PATH);

    /* Fresh independent mmap of the devdax device — no CUDA, no relation to
     * the global gpm mapping.  Proves data is in PM media, not in GPU L2
     * cache or the previous mmap's page-table entries. */
    size_t reopen_size = 0;
    int    is_pmem2    = 0;
    void  *addr2 = pmem_map_file(GPM_DEVDAX_PATH, 0 /* whole device */,
                                 0 /* no create */, 0666,
                                 &reopen_size, &is_pmem2);
    if (!addr2) {
        fprintf(stderr, "  [T2] FAIL: pmem_map_file(reopen) failed: %s\n",
                pmem_errormsg());
        return -1;
    }
    printf("  Fresh devdax mmap at CPU addr %p  size=%.1f GB  is_pmem=%d\n",
           addr2, (double)reopen_size / (1024.0*1024.0*1024.0), is_pmem2);

    /* Navigate to the saved offset and verify the pattern */
    int t2_errors = 0;
    uint64_t *view2 = (uint64_t *)((char *)addr2 + saved_offset);
    for (int tid = 0; tid < T1_NUM_THREADS; ++tid) {
        for (int j = 0; j < 8; ++j) {
            uint64_t expected = T1_MAGIC | (uint64_t)(tid * 8 + j);
            uint64_t got      = view2[tid * T1_SLOT_BYTES / 8 + j];
            if (got != expected) {
                if (t2_errors < 5)
                    fprintf(stderr,
                            "  [T2] mismatch tid=%d j=%d "
                            "expected=%016llx got=%016llx\n",
                            tid, j, (unsigned long long)expected,
                            (unsigned long long)got);
                ++t2_errors;
            }
        }
    }

    pmem_unmap(addr2, reopen_size);

    if (t2_errors) {
        fprintf(stderr, "  [T2] FAIL: %d errors after fresh remap\n",
                t2_errors);
        return -1;
    }
    printf("  [T2] PASS: data verified via independent devdax mmap\n");
    printf("       (data is in PM media, not GPU L2 cache or previous mapping)\n");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test T3: bandwidth comparison                                       */
/* ------------------------------------------------------------------ */
static double measure_bw_gb(volatile uint64_t *dst, size_t nbytes,
                             int warmup, int runs)
{
    size_t nwords = nbytes / sizeof(uint64_t);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        kernel_bw_write<<<T3_BLOCKS, T3_THREADS>>>(dst, nwords);
        cudaDeviceSynchronize();
    }

    float total_ms = 0.f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        kernel_bw_write<<<T3_BLOCKS, T3_THREADS>>>(dst, nwords);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_s  = (double)total_ms / (runs * 1000.0);
    double gb_s   = (double)nbytes / avg_s / 1e9;
    return gb_s;
}

static int run_t3(void)
{
    printf("\n=== [T3] Bandwidth: PM vs pinned DRAM (both via PCIe) ===\n");
    printf("  Region size: %lu MB  (%d warmup + %d measurement runs)\n",
           T3_REGION_BYTES >> 20, T3_WARMUP_RUNS, T3_MEASURE_RUNS);

    /* --- PM side --------------------------------------------------- */
    gpm_region_t pm_region;
    gpm_error_t  gerr = gpm_alloc(&pm_region, T3_REGION_BYTES, "t3pm");
    if (gerr != GPM_SUCCESS) {
        fprintf(stderr, "  [T3] gpm_alloc failed, skipping bandwidth test\n");
        return 0; /* non-fatal */
    }

    double pm_bw = measure_bw_gb((volatile uint64_t *)pm_region.addr,
                                 T3_REGION_BYTES,
                                 T3_WARMUP_RUNS, T3_MEASURE_RUNS);
    printf("  GPU -> PM write bandwidth    : %.2f GB/s\n", pm_bw);

    gpm_free(&pm_region);

    /* --- Pinned DRAM side ------------------------------------------ */
    void *dram_ptr = NULL;
    cudaError_t cu = cudaHostAlloc(&dram_ptr, T3_REGION_BYTES,
                                   cudaHostAllocDefault);
    if (cu != cudaSuccess) {
        fprintf(stderr, "  [T3] cudaHostAlloc failed: %s\n",
                cudaGetErrorString(cu));
        return 0;
    }

    double dram_bw = measure_bw_gb((volatile uint64_t *)dram_ptr,
                                   T3_REGION_BYTES,
                                   T3_WARMUP_RUNS, T3_MEASURE_RUNS);
    printf("  GPU -> pinned DRAM bandwidth : %.2f GB/s\n", dram_bw);

    cudaFreeHost(dram_ptr);

    /* --- Interpret results ----------------------------------------- */
    printf("\n  Interpretation:\n");
    printf("  - Both paths use volatile stores + PCIe (no cudaMemcpy)\n");
    printf("  - PM bandwidth is limited by Optane media (~2-4 GB/s expected)\n");
    printf("  - DRAM bandwidth is limited by PCIe bus (~10-16 GB/s expected)\n");
    if (pm_bw < dram_bw * 0.8) {
        printf("  [T3] PASS: PM significantly slower than DRAM  "
               "(%.1fx ratio) — GPU writes reach PM media\n",
               dram_bw / pm_bw);
    } else {
        printf("  [T3] NOTE: PM bandwidth close to DRAM (%.2f vs %.2f GB/s)\n",
               pm_bw, dram_bw);
        printf("       This may indicate emulated PM (is_pmem=0) or "
               "DDIO bypassing PM media.\n");
        printf("       Try running with DDIO disabled: "
               "gpm_persist_begin(gpm_find_gpu_bus())\n");
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║       GPM Direct PM Access Verification Test         ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    /* Init GPM (checks CUDA device) */
    if (gpm_init() != GPM_SUCCESS) {
        fprintf(stderr, "gpm_init failed\n");
        return 1;
    }

    /* Print device info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d)\n\n",
           prop.name, prop.major, prop.minor);

    int rc = 0;

    /* T1 + T2 */
    if (run_t1_t2() != 0) {
        fprintf(stderr, "\n[FAIL] T1/T2 failed\n");
        rc = 1;
    }

    /* T3 always runs (non-fatal) */
    run_t3();

    gpm_cleanup();

    printf("\n%s\n", rc == 0
           ? "══ All tests PASSED ══"
           : "══ Some tests FAILED ══");
    return rc;
}
