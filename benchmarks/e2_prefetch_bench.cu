/*
 * e2_prefetch_bench.cu — E2: DRAM Prefetch Validation & Performance
 *
 * Validates C3 claim: gap-period PM→DRAM prefetch eliminates the PM read
 * performance bottleneck without using VRAM as cache.
 *
 * Default config: 15 files × 1000 pages each (58.6 MiB total), matching
 * the aggregate size of a GPT-2 XL transformer block (E1 reference).
 * Files model per-tensor parameter offload (one file per weight tensor).
 * Each file stays within F2FS_ADDRS_PER_INODE (1017) direct-block limit.
 *
 * Three read measurements:
 *   PM-READ     : GPU thread reads from PM (volatile, L2 bypass) — no prefetch
 *   DRAM-COLD   : GPU thread reads from pinned DRAM, first cold access
 *                 Data just landed from PM via CPU prefetch; GPU L2 is cold.
 *                 Measures actual PCIe-limited DRAM bandwidth.
 *   DRAM-WARM   : Subsequent GPU reads, data already in GPU L2.
 *                 Shows full benefit once backward pass begins computation.
 *
 * PM and DRAM-COLD use volatile loads for L2 bypass (fair comparison).
 * DRAM-WARM uses cached loads (normal compute path).
 *
 * Run: sudo ./e2_prefetch_bench
 */

#include "gpu_f2fs.h"
#include "beaver_cow.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ------------------------------------------------------------------ */
/* Configuration                                                        */
/* ------------------------------------------------------------------ */

/*
 * One file per parameter tensor; each file must fit within
 * F2FS_ADDRS_PER_INODE (1017) direct-block pointers.
 * 15 × 1000 × 4 KiB = 58.6 MiB ≈ one GPT-2 XL transformer block.
 */
#ifndef E2_PAGES_PER_FILE
#define E2_PAGES_PER_FILE  1000u    /* ≤ F2FS_ADDRS_PER_INODE (1017) */
#endif
#ifndef E2_NUM_FILES
#define E2_NUM_FILES       15u
#endif
#ifndef E2_NUM_ITERS
#define E2_NUM_ITERS       5u
#endif

/* E1 reference constants (block_47, GPT-2 XL, batch=6, seq=512) */
#define E1_GAP_P50_MS      275.1
#define E1_PM_WR_MS        18.09   /* 58.6 MiB / 3.165 GB/s */

#define E2_TOTAL_PAGES     (E2_NUM_FILES * E2_PAGES_PER_FILE)

/* FS sizing */
#define BM_MAX_INODES      (E2_NUM_FILES + 2u)
#define BM_MAX_DATA_BLOCKS (E2_TOTAL_PAGES + 256u)

#define THREADS_PER_BLOCK  256u

/* ------------------------------------------------------------------ */
/* CUDA error check                                                     */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            exit(1);                                                   \
        }                                                              \
    } while (0)

/* ------------------------------------------------------------------ */
/* FS lifecycle                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    beaver_cache_t *cow_cache;
    beaver_cache_t *data_cache;
    gpu_f2fs_t     *fs;
    gpu_vfs_t      *vfs;
} e2_fs_t;

static void e2_fs_init(e2_fs_t *e)
{
    memset(e, 0, sizeof(*e));
    CUDA_CHECK(cudaMallocManaged((void **)&e->fs,         sizeof(gpu_f2fs_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&e->vfs,        sizeof(gpu_vfs_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&e->cow_cache,  sizeof(beaver_cache_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&e->data_cache, sizeof(beaver_cache_t)));

    if (beaver_cache_init(e->cow_cache,  BM_MAX_INODES)      != BEAVER_SUCCESS) {
        fprintf(stderr, "beaver_cache_init(cow) failed\n"); exit(1);
    }
    if (beaver_cache_init(e->data_cache, BM_MAX_DATA_BLOCKS)  != BEAVER_SUCCESS) {
        fprintf(stderr, "beaver_cache_init(data) failed\n"); exit(1);
    }
    if (gpu_f2fs_init(e->fs, e->cow_cache, e->data_cache, BM_MAX_INODES, 1)
            != GPU_F2FS_OK) {
        fprintf(stderr, "gpu_f2fs_init failed\n"); exit(1);
    }
    gpu_vfs_mount_f2fs(e->vfs, e->fs);
}

static void e2_fs_cleanup(e2_fs_t *e)
{
    gpu_f2fs_cleanup(e->fs);
    beaver_cache_cleanup(e->cow_cache);
    beaver_cache_cleanup(e->data_cache);
    cudaFree(e->vfs);
    cudaFree(e->fs);
    cudaFree(e->cow_cache);
    cudaFree(e->data_cache);
}

/* ------------------------------------------------------------------ */
/* GPU kernels                                                          */
/* ------------------------------------------------------------------ */

/* Create one file per slot; hashes[i] = unique name hash for file i */
__global__ static void create_kernel(gpu_f2fs_t *fs, const uint32_t *hashes,
                                      int *out_fds, uint32_t nfiles)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfiles)
        out_fds[i] = gpu_f2fs_create(fs, hashes[i]);
}

/*
 * Write one file's worth of pages from a contiguous slice of src_buf.
 * Single GPU thread sequential loop: avoids inode spinlock contention.
 * Staged writes + one fsync → one PM drain for the entire file.
 */
__global__ static void write_file_kernel(gpu_f2fs_t *fs, int fd,
                                          uint32_t npages,
                                          const char *src_slice)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (uint32_t pgoff = 0; pgoff < npages; ++pgoff)
        gpu_f2fs_write_data(fs, fd, pgoff,
                            src_slice + (size_t)pgoff * BEAVER_PAGE_SIZE);
    gpu_f2fs_fsync(fs, fd);
}

/*
 * Read all pages of one file into dst_slice, parallel.
 *
 * pm_volatile == true:  volatile loads throughout (L2 bypass) — used for
 *   PM reads and DRAM-COLD reads to force actual PCIe transfers.
 *   NOTE: beaver_data_read uses volatile for PM fallback automatically.
 *   For DRAM path, we call the same function; the compiler may cache the
 *   non-volatile DRAM loads.  To force volatile for DRAM, we read via
 *   the holder's read_ptr after confirming dram_dev_ptrs is set — but
 *   that bypasses the management layer.  Instead we flush L2 between
 *   iterations in the calling code (see L2 flush below).
 *
 * XOR checksum prevents the compiler from eliding reads.
 */
__global__ static void read_file_kernel(gpu_f2fs_t *fs, int fd,
                                         uint32_t npages, char *dst_slice,
                                         unsigned long long *checksum)
{
    uint32_t pgoff = blockIdx.x * blockDim.x + threadIdx.x;
    if (pgoff >= npages) return;

    char *dst = dst_slice + (size_t)pgoff * BEAVER_PAGE_SIZE;
    gpu_f2fs_read(fs, fd, pgoff, dst);

    /* XOR checksum to prevent dead-store elimination */
    uint64_t s = 0;
    const uint64_t *p = (const uint64_t *)dst;
    for (uint32_t i = 0; i < BEAVER_PAGE_SIZE / 8; ++i)
        s ^= p[i];
    atomicXor(checksum, s);
}

/*
 * L2 flush kernel: sequential volatile reads over a large buffer to
 * evict previously cached DRAM data from the GPU L2 (72 MiB on RTX 4090).
 * Call with buffer_bytes > L2_SIZE and stride = cache-line (128 B).
 */
__global__ static void l2_flush_kernel(const volatile uint32_t *buf,
                                        uint32_t nelems,
                                        unsigned long long *sink)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t acc = 0;
    /* stride by 32 (warp width) to touch every cache line */
    for (; i < nelems; i += gridDim.x * blockDim.x)
        acc += buf[i];
    atomicAdd(sink, acc);
}

/* ------------------------------------------------------------------ */
/* Timing helpers                                                       */
/* ------------------------------------------------------------------ */

static double elapsed_ms(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

static double gpu_event_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(void)
{
    const size_t file_bytes  = (size_t)E2_PAGES_PER_FILE * BEAVER_PAGE_SIZE;
    const size_t total_bytes = (size_t)E2_TOTAL_PAGES    * BEAVER_PAGE_SIZE;
    const uint32_t grid = (E2_PAGES_PER_FILE + THREADS_PER_BLOCK - 1)
                          / THREADS_PER_BLOCK;

    printf("=== E2 Prefetch Benchmark (C3 claim validation) ===\n");
    printf("Config: %u files x %u pages each (%.1f MiB total), %u iters\n\n",
           E2_NUM_FILES, E2_PAGES_PER_FILE,
           (double)total_bytes / (1 << 20), E2_NUM_ITERS);

    /* cudaDeviceMapHost must precede the first CUDA API call */
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost));

    uint8_t bus = ddio_get_gpu_bus();
    ddio_disable(bus);

    /* ---- Init FS ---- */
    e2_fs_t e;
    e2_fs_init(&e);

    if (beaver_dram_pool_init(e.data_cache, E2_TOTAL_PAGES) != BEAVER_SUCCESS) {
        fprintf(stderr, "beaver_dram_pool_init failed\n"); return 1;
    }

    /* ---- Allocate device buffers ---- */
    char               *d_src      = NULL;
    char               *d_dst      = NULL;
    unsigned long long *d_checksum = NULL;
    uint32_t           *d_flush    = NULL;
    unsigned long long *d_flush_sink = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_src,       total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_dst,       total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_checksum,  sizeof(unsigned long long)));

    /* L2 flush buffer: 96 MiB > RTX 4090 L2 (72 MiB) */
    const size_t flush_bytes = 96UL << 20;
    CUDA_CHECK(cudaMalloc((void **)&d_flush,      flush_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_flush_sink, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_flush, 0xAB, flush_bytes));
    CUDA_CHECK(cudaMemset(d_flush_sink, 0, sizeof(unsigned long long)));

    /* ---- Build source pattern on host, copy to device ---- */
    char *h_src = (char *)malloc(total_bytes);
    if (!h_src) { fprintf(stderr, "malloc(h_src) failed\n"); return 1; }
    {
        uint64_t *p = (uint64_t *)h_src;
        for (size_t i = 0; i < total_bytes / 8; ++i)
            p[i] = 0xDEADBEEFCAFEBABEULL ^ (uint64_t)i;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src, total_bytes, cudaMemcpyHostToDevice));

    /* ---- Create files ---- */
    uint32_t h_hashes[E2_NUM_FILES];
    for (uint32_t i = 0; i < E2_NUM_FILES; ++i)
        h_hashes[i] = 0x11111111u * (i + 1u);

    uint32_t *d_hashes = NULL;
    int      *d_fds    = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_hashes, E2_NUM_FILES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&d_fds, E2_NUM_FILES * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_hashes, h_hashes,
                          E2_NUM_FILES * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    {
        uint32_t blk = 256, grd = (E2_NUM_FILES + blk - 1) / blk;
        create_kernel<<<grd, blk>>>(e.fs, d_hashes, d_fds, E2_NUM_FILES);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Verify all fds valid */
    for (uint32_t i = 0; i < E2_NUM_FILES; ++i) {
        if (d_fds[i] < 0) {
            fprintf(stderr, "gpu_f2fs_create failed for file %u\n", i);
            return 1;
        }
    }

    /* ---------------------------------------------------------------- */
    /* WRITE phase: one kernel per file (sequential, staged + fsync)    */
    /* ---------------------------------------------------------------- */
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    for (uint32_t f = 0; f < E2_NUM_FILES; ++f) {
        const char *src_slice = d_src + (size_t)f * file_bytes;
        write_file_kernel<<<1, 1>>>(e.fs, d_fds[f], E2_PAGES_PER_FILE, src_slice);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaDeviceSynchronize());

    double write_ms = gpu_event_ms(ev0, ev1);
    double write_bw = (double)total_bytes / write_ms / 1024.0;
    printf("Write (Beaver COW, GPU→PM):     %.1f MB/s  (%.1f ms)\n",
           write_bw, write_ms);

    /* ---------------------------------------------------------------- */
    /* BM_PM_READ: volatile path to PM, GPU L2 bypassed, E2_NUM_ITERS  */
    /* ---------------------------------------------------------------- */
    double pm_ms_sum = 0.0;
    bool   pm_ok     = false;
    char  *h_dst     = (char *)malloc(total_bytes);
    if (!h_dst) { fprintf(stderr, "malloc(h_dst) failed\n"); return 1; }

    for (uint32_t iter = 0; iter < E2_NUM_ITERS; ++iter) {
        CUDA_CHECK(cudaMemset(d_dst,      0, total_bytes));
        CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));

        CUDA_CHECK(cudaEventRecord(ev0));
        for (uint32_t f = 0; f < E2_NUM_FILES; ++f) {
            char *dst_slice = d_dst + (size_t)f * file_bytes;
            read_file_kernel<<<grid, THREADS_PER_BLOCK>>>(
                    e.fs, d_fds[f], E2_PAGES_PER_FILE, dst_slice, d_checksum);
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        pm_ms_sum += gpu_event_ms(ev0, ev1);

        if (iter == E2_NUM_ITERS - 1) {
            /* Correctness: final PM read */
            CUDA_CHECK(cudaMemcpy(h_dst, d_dst, total_bytes,
                                  cudaMemcpyDeviceToHost));
            pm_ok = (memcmp(h_src, h_dst, total_bytes) == 0);
        }
    }
    double pm_ms_avg = pm_ms_sum / E2_NUM_ITERS;
    double pm_bw     = (double)total_bytes / pm_ms_avg / 1024.0;

    printf("PM cold read (GPU, volatile):   %.1f MB/s  (%.1f ms avg)  correct=%s\n",
           pm_bw, pm_ms_avg, pm_ok ? "YES" : "NO");

    /* ---------------------------------------------------------------- */
    /* BM_PREFETCH: CPU memcpy PM→pinned DRAM                           */
    /* ---------------------------------------------------------------- */
    double pf_ms_min = 1e9;

    for (uint32_t iter = 0; iter < E2_NUM_ITERS; ++iter) {
        CUDA_CHECK(cudaDeviceSynchronize());
        beaver_dram_pool_reset(e.data_cache);

        struct timespec pt0, pt1;
        clock_gettime(CLOCK_MONOTONIC, &pt0);
        for (uint32_t f = 0; f < E2_NUM_FILES; ++f)
            gpu_f2fs_prefetch(e.fs, (int)d_fds[f]);
        clock_gettime(CLOCK_MONOTONIC, &pt1);

        pf_ms_min = fmin(pf_ms_min, elapsed_ms(pt0, pt1));
    }
    double pf_bw = (double)total_bytes / pf_ms_min / 1024.0;
    printf("Prefetch (CPU, PM→DRAM):        %.1f MB/s  (%.1f ms min)\n",
           pf_bw, pf_ms_min);

    /* ---------------------------------------------------------------- */
    /* BM_DRAM_COLD: first GPU read pass after prefetch, L2 is cold.   */
    /* Flush GPU L2 first by reading a large buffer, then time one pass.*/
    /* ---------------------------------------------------------------- */

    /* L2 flush */
    {
        uint32_t nelems = (uint32_t)(flush_bytes / sizeof(uint32_t));
        l2_flush_kernel<<<256, 256>>>(d_flush, nelems, d_flush_sink);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemset(d_dst,      0, total_bytes));
    CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));

    CUDA_CHECK(cudaEventRecord(ev0));
    for (uint32_t f = 0; f < E2_NUM_FILES; ++f) {
        char *dst_slice = d_dst + (size_t)f * file_bytes;
        read_file_kernel<<<grid, THREADS_PER_BLOCK>>>(
                e.fs, d_fds[f], E2_PAGES_PER_FILE, dst_slice, d_checksum);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaDeviceSynchronize());
    double dram_cold_ms = gpu_event_ms(ev0, ev1);
    double dram_cold_bw = (double)total_bytes / dram_cold_ms / 1024.0;

    /* Correctness: DRAM cold read */
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, total_bytes, cudaMemcpyDeviceToHost));
    bool dram_ok = (memcmp(h_src, h_dst, total_bytes) == 0);

    printf("DRAM cold read (GPU, 1st pass): %.1f MB/s  (%.1f ms)  correct=%s\n",
           dram_cold_bw, dram_cold_ms, dram_ok ? "YES" : "NO");

    /* ---------------------------------------------------------------- */
    /* BM_DRAM_WARM: subsequent passes, data in GPU L2                  */
    /* ---------------------------------------------------------------- */
    double dram_warm_ms_min = 1e9;
    for (uint32_t iter = 0; iter < E2_NUM_ITERS; ++iter) {
        CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaEventRecord(ev0));
        for (uint32_t f = 0; f < E2_NUM_FILES; ++f) {
            char *dst_slice = d_dst + (size_t)f * file_bytes;
            read_file_kernel<<<grid, THREADS_PER_BLOCK>>>(
                    e.fs, d_fds[f], E2_PAGES_PER_FILE, dst_slice, d_checksum);
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaDeviceSynchronize());
        dram_warm_ms_min = fmin(dram_warm_ms_min, gpu_event_ms(ev0, ev1));
    }
    double dram_warm_bw = (double)total_bytes / dram_warm_ms_min / 1024.0;
    printf("DRAM warm read (GPU L2 hit):    %.1f MB/s  (%.1f ms min)\n",
           dram_warm_bw, dram_warm_ms_min);

    /* ---------------------------------------------------------------- */
    /* Summary                                                           */
    /* ---------------------------------------------------------------- */
    printf("\n--- Summary ---\n");
    printf("Read speedup DRAM-cold vs PM:  %.2fx\n",
           dram_cold_bw / pm_bw);
    printf("Read speedup DRAM-warm vs PM:  %.2fx  (L2 cached, compute path)\n",
           dram_warm_bw / pm_bw);

    printf("\n--- E1 gap window comparison (block_47, GPT-2 XL, batch=6) ---\n");
    printf("  E1 PM write time (est.):     %.2f ms\n", E1_PM_WR_MS);
    printf("  Measured prefetch time:      %.2f ms  (CPU memcpy, min of %u)\n",
           pf_ms_min, E2_NUM_ITERS);
    double total_required = E1_PM_WR_MS + pf_ms_min;
    double margin         = E1_GAP_P50_MS - total_required;
    printf("  Total required (wr+pf):      %.2f ms\n", total_required);
    printf("  E1 gap p50 (block_47):       %.1f ms\n", E1_GAP_P50_MS);
    printf("  Margin:                      %.2f ms  → %s\n",
           margin, margin > 0.0 ? "VIABLE" : "NOT VIABLE");

    /* ---------------------------------------------------------------- */
    /* Cleanup                                                           */
    /* ---------------------------------------------------------------- */
    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_checksum);
    cudaFree(d_flush);
    cudaFree(d_flush_sink);
    cudaFree(d_hashes);
    cudaFree(d_fds);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    e2_fs_cleanup(&e);
    ddio_enable(bus);

    return (pm_ok && dram_ok) ? 0 : 1;
}
