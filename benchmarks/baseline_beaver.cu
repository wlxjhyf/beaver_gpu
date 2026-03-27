/*
 * baseline_beaver.cu — beaver_gpu-COW microbenchmark.
 *
 * Data path (GPU-native, no CPU in write path):
 *   GPU kernel (vfs_write) → Beaver COW holder → gpm_memcpy → /dev/dax1.0 → PM
 *
 * Timing (global, host-side clock_gettime):
 *   START: before kernel launch
 *   STOP : after cudaDeviceSynchronize() (all PM writes durable)
 *
 * MB_SEQ  : pre-create N files (not timed), then N threads seq-write.
 * MB_RAND : pre-create N files (not timed), then N threads rand-write.
 * MB_MULTI: N threads create + seq-write in one kernel (all timed).
 *
 * FS sizing: each run allocates a fresh FS sized for nthreads:
 *   max_inodes    = nthreads
 *   max_data      = nthreads × MB_WRITES_PER_THREAD
 * This keeps PM allocation proportional and avoids over-allocating for
 * small thread counts.
 *
 * Run: sudo (DDIO disable + /dev/dax1.0 access, handled by microbench main).
 */

#include "bench_config.h"
#include "bench_utils.h"
#include "workload_kernels.cuh"
#include "gpu_f2fs.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── FS lifecycle (mirrors bm_fs_t from perf_benchmark.cu) ──────── */

//现在的做法把inode和data从上层（cache）开始就完全分离了
//但是beaver_cache_t是向下管理PM block的
typedef struct {
    beaver_cache_t *cow_cache;   /* inode COW holders  */
    beaver_cache_t *data_cache;  /* data page holders  */
    gpu_f2fs_t     *fs;
    gpu_vfs_t      *vfs;
} bv_fs_t;

static int bv_fs_init(bv_fs_t *bv, uint32_t max_inodes, uint32_t max_data)
{
    memset(bv, 0, sizeof *bv);
    if (cudaMallocManaged((void **)&bv->fs,         sizeof(gpu_f2fs_t))     != cudaSuccess) return -1;
    if (cudaMallocManaged((void **)&bv->vfs,        sizeof(gpu_vfs_t))      != cudaSuccess) return -1;
    if (cudaMallocManaged((void **)&bv->cow_cache,  sizeof(beaver_cache_t)) != cudaSuccess) return -1;
    if (cudaMallocManaged((void **)&bv->data_cache, sizeof(beaver_cache_t)) != cudaSuccess) return -1;

    if (beaver_cache_init(bv->data_cache, max_data)   != BEAVER_SUCCESS) return -1;
    if (beaver_cache_init(bv->cow_cache,  max_inodes)  != BEAVER_SUCCESS) return -1;

    if (gpu_f2fs_init(bv->fs, bv->cow_cache, bv->data_cache,
                      max_inodes, /*use_cow=*/1) != GPU_F2FS_OK) return -1;
    gpu_vfs_mount_f2fs(bv->vfs, bv->fs);
    return 0;
}

static void bv_fs_cleanup(bv_fs_t *bv)
{
    gpu_f2fs_cleanup(bv->fs);
    beaver_cache_cleanup(bv->cow_cache);
    beaver_cache_cleanup(bv->data_cache);
    cudaFree(bv->cow_cache);
    cudaFree(bv->data_cache);
    cudaFree(bv->vfs);
    cudaFree(bv->fs);
    memset(bv, 0, sizeof *bv);
}

/* ── Name hash array (device memory) ────────────────────────────── */

static int alloc_hashes(uint32_t base, uint32_t n, uint32_t **d_out)
{
    uint32_t *h = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!h) return -1;
    for (uint32_t i = 0; i < n; i++) h[i] = base + i * 7919u;
    uint32_t *d;
    if (cudaMalloc((void **)&d, n * sizeof(uint32_t)) != cudaSuccess)
        { free(h); return -1; }
    cudaMemcpy(d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    free(h);
    *d_out = d;
    return 0;
}

/* ── Shared write buffer (device, constant pattern) ─────────────── */

static uint8_t *g_wbuf = NULL;   /* MB_PAGE_SIZE bytes, device memory */

int beaver_bench_init(void)
{
    if (cudaMalloc((void **)&g_wbuf, MB_PAGE_SIZE) != cudaSuccess) {
        fprintf(stderr, "[Beaver] cudaMalloc for write buffer failed\n");
        return -1;
    }
    cudaMemset(g_wbuf, 0xBE, MB_PAGE_SIZE);
    return 0;
}

void beaver_bench_cleanup(void)
{
    if (g_wbuf) { cudaFree(g_wbuf); g_wbuf = NULL; }
}

/* ── MB_SEQ ──────────────────────────────────────────────────────── */

/*
 * Timing now covers create + write + fsync, matching the CPU rename path.
 * Uses mb_multi_write_kernel_loop (create+write in one kernel) for Seq order.
 * nthreads parameter is ignored — Beaver always uses MB_BEAVER_WARPS warps.
 */
double beaver_run_seq(uint32_t nthreads)
{
    (void)nthreads;
    uint32_t n       = MB_TOTAL_FILES;
    uint32_t n_warps = MB_BEAVER_WARPS;

    bv_fs_t bv;
    if (bv_fs_init(&bv, n, n * MB_WRITES_PER_THREAD) != 0) {
        fprintf(stderr, "[Beaver/Seq] FS init failed (MB_TOTAL_FILES=%u)\n", n);
        return -1.0;
    }

    uint32_t *d_hashes = NULL;
    if (alloc_hashes(0xA0000000u, n, &d_hashes) != 0)
        { bv_fs_cleanup(&bv); return -1.0; }

    /* warm-up (not timed) */
    mb_multi_write_kernel_loop<<<n_warps, 32u>>>(bv.vfs, d_hashes, g_wbuf, n);
    cudaDeviceSynchronize();
    bv_fs_cleanup(&bv);
    if (bv_fs_init(&bv, n, n * MB_WRITES_PER_THREAD) != 0) return -1.0;

    /* Timed: create + seq write + fsync */
    mb_timer_t t;
    mb_timer_start(&t);
    mb_multi_write_kernel_loop<<<n_warps, 32u>>>(bv.vfs, d_hashes, g_wbuf, n);
    double ms = mb_cuda_sync_elapsed_ms(&t);

    cudaFree(d_hashes);
    bv_fs_cleanup(&bv);
    return mb_throughput(ms);
}

/* ── MB_RAND ─────────────────────────────────────────────────────── */

double beaver_run_rand(uint32_t nthreads)
{
    (void)nthreads;
    uint32_t n       = MB_TOTAL_FILES;
    uint32_t n_warps = MB_BEAVER_WARPS;

    bv_fs_t bv;
    if (bv_fs_init(&bv, n, n * MB_WRITES_PER_THREAD) != 0) {
        fprintf(stderr, "[Beaver/Rand] FS init failed (MB_TOTAL_FILES=%u)\n", n);
        return -1.0;
    }

    uint32_t *d_hashes = NULL;
    if (alloc_hashes(0xA1000000u, n, &d_hashes) != 0)
        { bv_fs_cleanup(&bv); return -1.0; }

    /* warm-up (not timed) */
    mb_rand_multi_write_kernel_loop<<<n_warps, 32u>>>(bv.vfs, d_hashes, g_wbuf, n);
    cudaDeviceSynchronize();
    bv_fs_cleanup(&bv);
    if (bv_fs_init(&bv, n, n * MB_WRITES_PER_THREAD) != 0) return -1.0;

    /* Timed: create + rand write + fsync */
    mb_timer_t t;
    mb_timer_start(&t);
    mb_rand_multi_write_kernel_loop<<<n_warps, 32u>>>(bv.vfs, d_hashes, g_wbuf, n);
    double ms = mb_cuda_sync_elapsed_ms(&t);

    cudaFree(d_hashes);
    bv_fs_cleanup(&bv);
    return mb_throughput(ms);
}

/* ── MB_MULTI ────────────────────────────────────────────────────── */

double beaver_run_multi(uint32_t nthreads)
{
    (void)nthreads;
    uint32_t n       = MB_TOTAL_FILES;
    uint32_t n_warps = MB_BEAVER_WARPS;

    bv_fs_t bv;
    if (bv_fs_init(&bv, n, n * MB_WRITES_PER_THREAD) != 0) {
        fprintf(stderr, "[Beaver/Multi] FS init failed (MB_TOTAL_FILES=%u)\n", n);
        return -1.0;
    }

    uint32_t *d_hashes = NULL;
    if (alloc_hashes(0xA2000000u, n, &d_hashes) != 0)
        { bv_fs_cleanup(&bv); return -1.0; }

    /* MB_MULTI: create + write all timed together */
    mb_timer_t t;
    mb_timer_start(&t);
    mb_multi_write_kernel_loop<<<n_warps, 32u>>>(bv.vfs, d_hashes, g_wbuf, n);
    double ms = mb_cuda_sync_elapsed_ms(&t);

    cudaFree(d_hashes);
    bv_fs_cleanup(&bv);
    return mb_throughput(ms);
}

/* ── Dispatch ────────────────────────────────────────────────────── */

double beaver_run(mb_workload_t wl, uint32_t nthreads)
{
    switch (wl) {
    case MB_SEQ:   return beaver_run_seq  (nthreads);
    case MB_RAND:  return beaver_run_rand (nthreads);
    case MB_MULTI: return beaver_run_multi(nthreads);
    default:       return -1.0;
    }
}

/* ── Raw data write: no metadata, no holders, no log ────────────── */
/*
 * Allocates a flat PM slab of MB_TOTAL_FILES × MB_DATA_PER_THREAD bytes
 * and writes it using MB_BEAVER_WARPS warps with one __threadfence_system()
 * per file.  No VFS, no inode, no COW holder, no log entry.
 *
 * Isolates pure PM data-write throughput from Beaver metadata overhead.
 * The fence pattern (1 per file) is identical to Beaver's vfs_fsync.
 */
double beaver_run_raw_data(void)
{
    uint32_t n       = MB_TOTAL_FILES;
    uint32_t n_warps = MB_BEAVER_WARPS;
    size_t   total   = (size_t)n * MB_DATA_PER_THREAD;   /* 512 MiB */

    gpm_region_t pm_region;
    if (gpm_alloc(&pm_region, total, "beaver-rawdata") != GPM_SUCCESS) {
        fprintf(stderr, "[Beaver/RawData] gpm_alloc(%zu) failed\n", total);
        return -1.0;
    }
    void *pm_base = pm_region.addr;

    /* warm-up */
    mb_raw_write_kernel_loop<<<n_warps, 32u>>>(pm_base, g_wbuf, n);
    cudaDeviceSynchronize();

    /* timed run */
    mb_timer_t t;
    mb_timer_start(&t);
    mb_raw_write_kernel_loop<<<n_warps, 32u>>>(pm_base, g_wbuf, n);
    double ms = mb_cuda_sync_elapsed_ms(&t);

    gpm_free(&pm_region);
    return mb_throughput(ms);
}
