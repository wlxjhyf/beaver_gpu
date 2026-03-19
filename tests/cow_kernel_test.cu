/*
 * cow_kernel_test.cu — Phase 3 COW kernel correctness tests.
 *
 *  T1 — Single-thread: write → flip → get_read
 *  T2 — Second write = COW
 *  T3 — Block-level concurrent read/write (256 threads × 200 iters)
 *  T4 — Persistence: independent devdax remap
 *
 * holders and hash_table now live in cudaMalloc device memory.
 * Holder allocation is done via gpu_holder_alloc (device function) called
 * from a small setup kernel — the CPU never touches device memory directly.
 *
 * Run: sudo ./cow_kernel_test   (root needed for DDIO control)
 */

#include "beaver_cow.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <libpmem.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>

extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

#define TEST_MAX_HOLDERS  64u
#define T3_BLOCK_THREADS  256
#define T3_NUM_ITERS      200

#define T1_MAGIC  0xDEADBEEFCAFE0001ULL
#define T2_MAGIC  0xDEADBEEFCAFE0002ULL

/* ------------------------------------------------------------------ */
/* Result struct (cudaMallocManaged — visible to GPU + CPU)           */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t t1_pass;
    uint32_t t2_pass;
    uint32_t t3_null_count;
    uint32_t t3_iters_done;
    uint32_t find_pass;
    uint32_t _pad[3];
} test_results_t;

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
/* Setup kernel: allocate a holder via gpu_holder_alloc and write the */
/* resulting device pointer into *out (a device-memory pointer cell). */
/* ------------------------------------------------------------------ */
__global__ void holder_alloc_kernel(beaver_cache_t       *cache,
                                    uint64_t              page_id,
                                    gpu_shadow_holder_t **out)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *out = gpu_holder_alloc(cache, page_id);
}

/*
 * do_holder_alloc: launch the setup kernel and return the allocated
 * device pointer via cudaMemcpy.  The returned pointer is a device
 * address valid for passing to subsequent GPU kernels.
 */
static gpu_shadow_holder_t *do_holder_alloc(beaver_cache_t *cache,
                                            uint64_t        page_id)
{
    /* Allocate a device cell to hold the returned pointer */
    gpu_shadow_holder_t **d_out;
    cudaMalloc((void **)&d_out, sizeof(gpu_shadow_holder_t *));
    cudaMemset(d_out, 0, sizeof(gpu_shadow_holder_t *));

    holder_alloc_kernel<<<1, 1>>>(cache, page_id, d_out);
    cudaDeviceSynchronize();

    gpu_shadow_holder_t *h = NULL;
    cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return h;  /* device pointer */
}

/* ------------------------------------------------------------------ */
/* T1 kernel                                                           */
/* ------------------------------------------------------------------ */
__global__ void test_cow_t1_kernel(gpu_shadow_holder_t *h,
                                   beaver_cache_t      *cache,
                                   test_results_t      *res)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    if (h->cur != -1 || h->read_ptr != NULL) { res->t1_pass = 0; return; }

    /* gpu_find_holder smoke test */
    gpu_shadow_holder_t *found = gpu_find_holder(cache, h->page_id);
    res->find_pass = (found == h) ? 1u : 0u;

    void *waddr = gpu_holder_write_addr(h);
    if (waddr != h->pm_addrs[0]) { res->t1_pass = 0; return; }

    uint64_t magic = T1_MAGIC;
    gpm_memcpy(waddr, &magic, sizeof(magic));
    gpu_holder_flip(h);
    gpu_holder_commit(h);

    void *rptr = gpu_holder_get_read(h);
    if (rptr != h->pm_addrs[0] || h->cur != 0) { res->t1_pass = 0; return; }

    volatile uint64_t *slot0 = (volatile uint64_t *)h->pm_addrs[0];
    if (*slot0 != T1_MAGIC) { res->t1_pass = 0; return; }

    res->t1_pass = 1;
}

/* ------------------------------------------------------------------ */
/* T2 kernel                                                           */
/* ------------------------------------------------------------------ */
__global__ void test_cow_t2_kernel(gpu_shadow_holder_t *h,
                                   test_results_t      *res)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    if (h->cur != 0) { res->t2_pass = 0; return; }

    void *waddr = gpu_holder_write_addr(h);
    if (waddr != h->pm_addrs[1]) { res->t2_pass = 0; return; }

    uint64_t magic2 = T2_MAGIC;
    gpm_memcpy(waddr, &magic2, sizeof(magic2));
    gpu_holder_flip(h);
    gpu_holder_commit(h);

    void *rptr = gpu_holder_get_read(h);
    if (rptr != h->pm_addrs[1] || h->cur != 1) { res->t2_pass = 0; return; }

    volatile uint64_t *slot0 = (volatile uint64_t *)h->pm_addrs[0];
    if (*slot0 != T1_MAGIC) { res->t2_pass = 0; return; }

    volatile uint64_t *slot1 = (volatile uint64_t *)h->pm_addrs[1];
    if (*slot1 != T2_MAGIC) { res->t2_pass = 0; return; }

    res->t2_pass = 1;
}

/* ------------------------------------------------------------------ */
/* T3 kernel                                                           */
/* ------------------------------------------------------------------ */
__global__ void test_cow_t3_kernel(gpu_shadow_holder_t *h,
                                   test_results_t      *res)
{
    uint32_t tid = threadIdx.x;

    if (tid == 0) {
        void    *waddr = gpu_holder_write_addr(h);
        uint32_t val   = 0xABCD0000u;
        gpm_memcpy(waddr, &val, sizeof(val));
        gpu_holder_flip(h);
    }
    __syncthreads();

    for (int iter = 0; iter < T3_NUM_ITERS; ++iter) {
        if (tid == 0) {
            void    *waddr = gpu_holder_write_addr(h);
            uint32_t val   = (uint32_t)iter;
            gpm_memcpy_nodrain(waddr, &val, sizeof(val));
            gpu_holder_flip(h);
            atomicAdd(&res->t3_iters_done, 1u);
        }
        __syncthreads();

        void *rptr = gpu_holder_get_read(h);
        if (rptr == NULL)
            atomicAdd(&res->t3_null_count, 1u);

        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/* T4: persistence check via independent devdax remap                 */
/* ------------------------------------------------------------------ */
static int run_t4(size_t pm_offset)
{
    printf("\n=== [T4] Persistence: independent devdax remap ===\n");

    /* No CPU-side pmem_persist needed:
     *   - DDIO is disabled → GPU PCIe writes bypass CPU L3 cache entirely
     *   - gpu_holder_commit already issued __threadfence_system() (gpm_persist)
     * CPU cache holds none of this data, so clwb/sfence would be no-ops.
     * The GPU drain is the only persist operation required. */
    printf("  pm_offset=0x%zx\n", pm_offset);
    printf("  Opening fresh CPU-only mmap of %s ...\n", GPM_DEVDAX_PATH);

    size_t map_len  = 0;
    int    is_pmem2 = 0;
    void  *addr2 = pmem_map_file(GPM_DEVDAX_PATH, 0, 0, 0666,
                                  &map_len, &is_pmem2);
    if (!addr2) {
        fprintf(stderr, "  [T4] FAIL: pmem_map_file: %s\n", pmem_errormsg());
        return -1;
    }
    printf("  Fresh mmap at %p  size=%.1f GB  is_pmem=%d\n",
           addr2, (double)map_len / (1024.0*1024.0*1024.0), is_pmem2);

    int errors = 0;
    uint64_t got0 = *(volatile uint64_t *)((char *)addr2 + pm_offset);
    uint64_t got1 = *(volatile uint64_t *)((char *)addr2 + pm_offset
                                           + BEAVER_PAGE_SIZE);

    if (got0 != T1_MAGIC) {
        fprintf(stderr, "  [T4] slot0 mismatch: expected %016llx  got %016llx\n",
                (unsigned long long)T1_MAGIC, (unsigned long long)got0);
        errors++;
    } else {
        printf("  slot0 OK: %016llx == T1_MAGIC\n", (unsigned long long)got0);
    }

    if (got1 != T2_MAGIC) {
        fprintf(stderr, "  [T4] slot1 mismatch: expected %016llx  got %016llx\n",
                (unsigned long long)T2_MAGIC, (unsigned long long)got1);
        errors++;
    } else {
        printf("  slot1 OK: %016llx == T2_MAGIC\n", (unsigned long long)got1);
    }

    pmem_unmap(addr2, map_len);

    if (errors) { fprintf(stderr, "  [T4] FAIL\n"); return -1; }
    printf("  [T4] PASS: GPU writes reached PM media\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("==========================================================\n");
    printf("   beaver_gpu  Phase 3 — COW Kernel Correctness Tests\n");
    printf("==========================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    uint8_t gpu_bus = ddio_get_gpu_bus();
    printf("GPU PCIe bus: 0x%02x — disabling DDIO\n", gpu_bus);
    ddio_disable(gpu_bus);

    /* ---- Cache init: cache struct in managed memory --------------- */
    /* holders and hash_table are allocated as cudaMalloc device      */
    /* memory inside beaver_cache_init — CPU never touches them again. */
    beaver_cache_t *cache;
    cudaError_t cu = cudaMallocManaged((void **)&cache, sizeof(beaver_cache_t));
    if (cu != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged(cache): %s\n", cudaGetErrorString(cu));
        ddio_enable(gpu_bus); return 1;
    }

    beaver_error_t berr = beaver_cache_init(cache, TEST_MAX_HOLDERS);
    if (berr != BEAVER_SUCCESS) {
        fprintf(stderr, "beaver_cache_init failed (%d)\n", berr);
        cudaFree(cache); ddio_enable(gpu_bus); return 1;
    }

    test_results_t *res;
    cudaMallocManaged((void **)&res, sizeof(test_results_t));
    memset(res, 0, sizeof(test_results_t));

    /* ---- Allocate holders via GPU kernel (no CPU scanning) -------- */
    gpu_shadow_holder_t *h_t1t2 = do_holder_alloc(cache, 42u);
    gpu_shadow_holder_t *h_t3   = do_holder_alloc(cache, 99u);

    if (!h_t1t2 || !h_t3) {
        fprintf(stderr, "holder alloc failed\n");
        goto cleanup;
    }

    /* ================================================================ */
    printf("\n=== [T1] Single-thread: write → flip → get_read ===\n");

    test_cow_t1_kernel<<<1, 1>>>(h_t1t2, cache, res);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[T1] kernel error\n"); goto cleanup;
    }

    {
        /* Read holder fields from device memory for display */
        gpu_shadow_holder_t hl;
        cudaMemcpy(&hl, h_t1t2, sizeof(hl), cudaMemcpyDeviceToHost);
        printf("  gpu_find_holder: %s\n",
               res->find_pass ? "PASS (found correct holder)" : "FAIL");
        printf("  T1: %s  cur=%d  read_ptr=%p  pm_addrs[0]=%p\n",
               res->t1_pass ? "PASS" : "FAIL",
               hl.cur, hl.read_ptr, hl.pm_addrs[0]);
    }

    if (!res->t1_pass) { fprintf(stderr, "[T1] FAIL\n"); goto cleanup; }

    /* ================================================================ */
    printf("\n=== [T2] Second write = COW ===\n");

    test_cow_t2_kernel<<<1, 1>>>(h_t1t2, res);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[T2] kernel error\n"); goto cleanup;
    }

    {
        gpu_shadow_holder_t hl;
        cudaMemcpy(&hl, h_t1t2, sizeof(hl), cudaMemcpyDeviceToHost);
        printf("  T2: %s  cur=%d  read_ptr=%p  pm_addrs[1]=%p\n",
               res->t2_pass ? "PASS" : "FAIL",
               hl.cur, hl.read_ptr, hl.pm_addrs[1]);
        if (res->t2_pass)
            printf("       slot0=T1_MAGIC preserved, slot1=T2_MAGIC written\n");
    }

    /* ================================================================ */
    printf("\n=== [T3] Concurrent read/write (%d threads × %d iters) ===\n",
           T3_BLOCK_THREADS, T3_NUM_ITERS);

    test_cow_t3_kernel<<<1, T3_BLOCK_THREADS>>>(h_t3, res);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "[T3] kernel error\n"); goto cleanup;
    }

    printf("  T3: %s  null_reads=%u  iters_done=%u\n",
           res->t3_null_count == 0u ? "PASS" : "FAIL",
           res->t3_null_count, res->t3_iters_done);

    /* ================================================================ */
    if (res->t1_pass && res->t2_pass) {
        /* h_t1t2 is holder index 0; pm_addrs are computed from pm_base.
         * cache->pm_base is accessible here because cache is managed.  */
        run_t4(cache->pm_region.pm_offset);
    } else {
        printf("\n[T4] Skipped\n");
    }

cleanup:
    printf("\n==========================================================\n");
    int all_pass = res->t1_pass && res->t2_pass &&
                   (res->t3_null_count == 0) && res->find_pass;
    printf("  find_holder : %s\n", res->find_pass           ? "PASS" : "FAIL");
    printf("  T1          : %s\n", res->t1_pass             ? "PASS" : "FAIL");
    printf("  T2          : %s\n", res->t2_pass             ? "PASS" : "FAIL");
    printf("  T3          : %s\n", res->t3_null_count == 0  ? "PASS" : "FAIL");
    printf("==========================================================\n");
    printf("%s\n", all_pass ? "ALL PASS" : "SOME TESTS FAILED");

    cudaFree(res);
    beaver_cache_cleanup(cache);
    cudaFree(cache);
    ddio_enable(gpu_bus);
    printf("DDIO restored.\n");

    return all_pass ? 0 : 1;
}
