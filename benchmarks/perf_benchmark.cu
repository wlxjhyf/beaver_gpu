/*
 * perf_benchmark.cu — Phase 5: End-to-end Performance Benchmarks
 *
 * Quantifies the benefit of Beaver COW over F2FS checkpoint in a GPU F2FS.
 * All results are collected in memory and a summary analysis is printed after
 * all three benchmark suites complete.
 *
 *   BM1: Write throughput vs. thread count (COW vs. Checkpoint mode)
 *        Thread counts: 1, 4, 16, 64, 128, 256, 512, 1024
 *        Each thread writes BM1_WRITES_PER_THREAD 4-KiB pages to its own file.
 *        Metric: aggregate MB/s (cudaEvent timing).
 *
 *   BM2: Per-write latency distribution (COW vs. Checkpoint mode)
 *        BM2_NTHREADS × BM2_WRITES_PER_THREAD concurrent writes.
 *        Each write latency captured via clock64() on GPU.
 *        Metric: P50, P95, P99, P99.9, Max (ns).
 *
 *   BM3: Checkpoint stop-the-world stall analysis
 *        dirty_count ∈ {64, 128, 256, 512, 1024} dirty inodes,
 *        each file pre-written with BM3_WRITES_PER_FILE pages.
 *        Metric: cow_ms, ckpt_write_ms, ckpt_stall_ms, ckpt_total_ms.
 *
 *   Analysis: printed after all BMs — ratios, overhead, key findings.
 *
 * Write cost breakdown (per gpu_write call):
 *   COW mode:        data→PM (gpm_memcpy = volatile stores + drain)
 *                  + inode→PM (gpm_memcpy_nodrain + gpu_holder_flip drain)
 *                  → 2 PM writes + 2 __threadfence_system()
 *   Checkpoint mode: data→PM (gpm_memcpy = volatile stores + drain)
 *                  + dirty_flags[nid] = 1  (DRAM only)
 *                  → 1 PM write + 1 __threadfence_system()
 *   Checkpoint stall (do_checkpoint): flush N dirty inodes→PM (stop-the-world)
 *
 * Run: sudo ./perf_benchmark   (root required for DDIO + /dev/dax1.0)
 */

#include "gpu_f2fs.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <libpmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ------------------------------------------------------------------ */
/* Benchmark configuration                                             */
/* ------------------------------------------------------------------ */

#define BM_MAX_INODES       1024u
#define BM_MAX_DATA_BLOCKS  65536u   /* 64 K × 4 KiB = 256 MiB per FS init */

#define BM1_WRITES_PER_THREAD  32u
#define BM1_NT_COUNT           8u    /* number of thread-count configs */

#define BM2_NTHREADS           256u
#define BM2_WRITES_PER_THREAD  64u
#define BM2_TOTAL_SAMPLES      ((size_t)BM2_NTHREADS * BM2_WRITES_PER_THREAD)

#define BM3_WRITES_PER_FILE    32u   /* total writes per thread             */
#define BM3_BATCH_SIZE         4u    /* writes per thread per checkpoint    */
#define BM3_NUM_BATCHES        (BM3_WRITES_PER_FILE / BM3_BATCH_SIZE)
#define BM3_ND_COUNT           5u    /* number of dirty-inode-count configs */

/* ------------------------------------------------------------------ */
/* Result structs (filled by run_bm*, read by print_analysis)         */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t nt;            /* thread count                          */
    double   cow_ms;        /* COW mode kernel time                  */
    double   ckpt_ms;       /* Checkpoint mode kernel time           */
    double   cow_tput;      /* COW mode aggregate MB/s               */
    double   ckpt_tput;     /* Checkpoint mode aggregate MB/s        */
} bm1_row_t;

typedef struct {
    double p50, p95, p99, p999, pmax;   /* all in nanoseconds */
} bm2_stats_t;

typedef struct {
    uint32_t nd;               /* dirty inode count                    */
    double   cow_ms;           /* COW: write kernel time               */
    double   ckpt_write_ms;    /* Checkpoint: write kernel time        */
    double   ckpt_stall_ms;    /* Checkpoint: do_checkpoint() wall time */
    double   ckpt_total_ms;    /* ckpt_write_ms + ckpt_stall_ms        */
} bm3_row_t;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));         \
            return -1;                                                   \
        }                                                                \
    } while (0)

static void fill_page(void *buf, uint64_t pattern)
{
    uint64_t *p = (uint64_t *)buf;
    for (size_t i = 0; i < BEAVER_PAGE_SIZE / sizeof(uint64_t); ++i)
        p[i] = pattern;
}

static uint32_t bm_hash(uint32_t base, uint32_t i)
{
    return base + i * 7919u;   /* 7919 is prime; keeps hashes spread */
}

/* ------------------------------------------------------------------ */
/* FS lifecycle                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    beaver_cache_t *cache;       /* inode COW cache (COW mode only)     */
    beaver_cache_t *data_cache;  /* data page COW cache (both modes)    */
    gpu_f2fs_t     *fs;
    gpu_vfs_t      *vfs;
} bm_fs_t;

static int bm_fs_init(bm_fs_t *bm, uint32_t use_cow)
{
    memset(bm, 0, sizeof(*bm));
    if (cudaMallocManaged((void **)&bm->fs,         sizeof(gpu_f2fs_t))     != cudaSuccess) return -1;
    if (cudaMallocManaged((void **)&bm->vfs,        sizeof(gpu_vfs_t))      != cudaSuccess) return -1;
    if (cudaMallocManaged((void **)&bm->data_cache, sizeof(beaver_cache_t)) != cudaSuccess) return -1;

    /* data_cache is always needed (data goes through Beaver in both modes) */
    if (beaver_cache_init(bm->data_cache, BM_MAX_DATA_BLOCKS) != BEAVER_SUCCESS) return -1;

    /* cow_cache only needed in COW mode */
    if (use_cow) {
        if (cudaMallocManaged((void **)&bm->cache, sizeof(beaver_cache_t)) != cudaSuccess) return -1;
        if (beaver_cache_init(bm->cache, BM_MAX_INODES) != BEAVER_SUCCESS) return -1;
    }

    if (gpu_f2fs_init(bm->fs, bm->cache, bm->data_cache, BM_MAX_INODES, use_cow)
            != GPU_F2FS_OK) return -1;
    gpu_vfs_mount_f2fs(bm->vfs, bm->fs);
    return 0;
}

static void bm_fs_cleanup(bm_fs_t *bm)
{
    gpu_f2fs_cleanup(bm->fs);
    if (bm->cache) {
        beaver_cache_cleanup(bm->cache);
        cudaFree(bm->cache);
    }
    if (bm->data_cache) {
        beaver_cache_cleanup(bm->data_cache);
        cudaFree(bm->data_cache);
    }
    cudaFree(bm->vfs);
    cudaFree(bm->fs);
    memset(bm, 0, sizeof(bm_fs_t));
}

static int setup_hashes(uint32_t base, uint32_t n, uint32_t **d_out)
{
    uint32_t *h = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!h) return -1;
    for (uint32_t i = 0; i < n; ++i) h[i] = bm_hash(base, i);
    uint32_t *d;
    if (cudaMalloc((void **)&d, n * sizeof(uint32_t)) != cudaSuccess)
        { free(h); return -1; }
    cudaMemcpy(d, h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    free(h);
    *d_out = d;
    return 0;
}

static void grid_block(uint32_t n, uint32_t *grid, uint32_t *block)
{
    *block = (n < 128u) ? n : 128u;
    *grid  = (n + *block - 1) / *block;
}

/* ------------------------------------------------------------------ */
/* GPU kernels                                                         */
/* ------------------------------------------------------------------ */

__global__ void bm_create_kernel(gpu_vfs_t      *vfs,
                                  const uint32_t *hashes,
                                  uint32_t        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vfs_create(vfs, hashes[i]);
}

__global__ void bm_write_kernel(gpu_vfs_t      *vfs,
                                 const uint32_t *name_hashes,
                                 const uint8_t  *wbuf,
                                 uint32_t        nthreads,
                                 uint32_t        writes_per_thread,
                                 uint32_t       *errors)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    int fd = vfs_open(vfs, name_hashes[tid]);
    if (fd < 0) { atomicAdd(errors, 1u); return; }
    for (uint32_t i = 0; i < writes_per_thread; ++i)
        if (vfs_write_data(vfs, fd, i, wbuf) != 0)   /* stage, no fence */
            atomicAdd(errors, 1u);
    vfs_fsync(vfs, fd);   /* one fence + inode persist for entire batch */
    vfs_close(vfs, fd);
}

/*
 * bm3_batch_write_kernel: write batch_size pages per thread starting at
 * pgoff_start.  Used by BM3 to write in fixed-size batches between
 * checkpoints, modelling F2FS stop-the-world behaviour.
 */
__global__ void bm3_batch_write_kernel(gpu_vfs_t      *vfs,
                                        const uint32_t *name_hashes,
                                        const uint8_t  *wbuf,
                                        uint32_t        nthreads,
                                        uint32_t        pgoff_start,
                                        uint32_t        batch_size,
                                        uint32_t       *errors)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    int fd = vfs_open(vfs, name_hashes[tid]);
    if (fd < 0) { atomicAdd(errors, 1u); return; }
    for (uint32_t i = 0; i < batch_size; ++i)
        if (vfs_write_data(vfs, fd, pgoff_start + i, wbuf) != 0)
            atomicAdd(errors, 1u);
    vfs_fsync(vfs, fd);   /* one fence per batch before checkpoint */
    vfs_close(vfs, fd);
}

/*
 * Latency kernel: clock64() delta for every vfs_write.
 * Stored row-major: lat_samples[tid * writes_per_thread + i] = cycles.
 */
__global__ void bm_latency_kernel(gpu_vfs_t      *vfs,
                                   const uint32_t *name_hashes,
                                   const uint8_t  *wbuf,
                                   uint32_t        nthreads,
                                   uint32_t        writes_per_thread,
                                   uint64_t       *lat_samples)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads) return;

    int fd = vfs_open(vfs, name_hashes[tid]);
    if (fd < 0) return;
    uint64_t *my = lat_samples + (size_t)tid * writes_per_thread;
    for (uint32_t i = 0; i < writes_per_thread; ++i) {
        uint64_t t0 = clock64();
        vfs_write(vfs, fd, i, wbuf);
        my[i] = clock64() - t0;
    }
    vfs_close(vfs, fd);
}

/* ------------------------------------------------------------------ */
/* Percentile helpers                                                  */
/* ------------------------------------------------------------------ */

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static double cyc2ns(uint64_t cyc, int rate_khz)
{
    return (double)cyc * 1000.0 / (double)rate_khz;
}

static bm2_stats_t compute_stats(uint64_t *s, size_t n, int rate_khz)
{
    qsort(s, n, sizeof(uint64_t), cmp_u64);
    bm2_stats_t r;
    r.p50  = cyc2ns(s[(size_t)(n * 0.500)], rate_khz);
    r.p95  = cyc2ns(s[(size_t)(n * 0.950)], rate_khz);
    r.p99  = cyc2ns(s[(size_t)(n * 0.990)], rate_khz);
    r.p999 = cyc2ns(s[(size_t)(n * 0.999)], rate_khz);
    r.pmax = cyc2ns(s[n - 1],               rate_khz);
    return r;
}

static void print_latency_row(const bm2_stats_t *st, const char *label)
{
    printf("  %-12s  %9.0f  %9.0f  %9.0f  %10.0f  %10.0f\n",
           label, st->p50, st->p95, st->p99, st->p999, st->pmax);
}

/* ------------------------------------------------------------------ */
/* Timed write helper (cudaEvent, returns elapsed ms)                 */
/* ------------------------------------------------------------------ */

static float timed_write(bm_fs_t *bm, const uint32_t *d_h,
                          const uint8_t *wbuf,
                          uint32_t gr, uint32_t bl,
                          uint32_t nt, uint32_t wpt)
{
    uint32_t *d_err = NULL;
    cudaMallocManaged((void **)&d_err, sizeof(uint32_t)); *d_err = 0;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    bm_write_kernel<<<gr, bl>>>(bm->vfs, d_h, wbuf, nt, wpt, d_err);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    if (*d_err > 0)
        fprintf(stderr, "  [WARNING] %u write errors\n", *d_err);
    cudaFree(d_err);
    return ms;
}

/* ------------------------------------------------------------------ */
/* BM1: Write throughput vs. thread count                             */
/* ------------------------------------------------------------------ */

static const uint32_t BM1_TCOUNTS[BM1_NT_COUNT] = {1, 4, 16, 64, 128, 256, 512, 1024};

static int run_bm1(const uint8_t *wbuf, bm1_row_t *rows)
{
    printf("\n### BM1: Write Throughput vs. Thread Count ###\n");
    printf("    %u writes/thread × %u B/write\n",
           BM1_WRITES_PER_THREAD, BEAVER_PAGE_SIZE);
    printf("  %-10s  %-12s  %16s  %10s\n",
           "Threads", "Mode", "Throughput(MB/s)", "Time(ms)");

    for (uint32_t ti = 0; ti < BM1_NT_COUNT; ++ti) {
        uint32_t nt = BM1_TCOUNTS[ti];
        rows[ti].nt = nt;

        uint32_t gr, bl;
        grid_block(nt, &gr, &bl);

        /* ---- COW ---- */
        bm_fs_t bm;
        if (bm_fs_init(&bm, 1u) != 0) return -1;
        uint32_t *d_h = NULL;
        if (setup_hashes(0xB0000000u, nt, &d_h) != 0) return -1;
        bm_create_kernel<<<gr, bl>>>(bm.vfs, d_h, nt);
        cudaDeviceSynchronize();

        float ms_cow = timed_write(&bm, d_h, wbuf, gr, bl, nt, BM1_WRITES_PER_THREAD);
        size_t total = (size_t)nt * BM1_WRITES_PER_THREAD * BEAVER_PAGE_SIZE;
        rows[ti].cow_ms   = (double)ms_cow;
        rows[ti].cow_tput = (double)total / ((double)ms_cow / 1000.0) / (1024.0*1024.0);
        printf("  %-10u  %-12s  %16.2f  %10.3f\n",
               nt, "COW", rows[ti].cow_tput, rows[ti].cow_ms);
        cudaFree(d_h);
        bm_fs_cleanup(&bm);

        /* ---- Checkpoint ---- */
        if (bm_fs_init(&bm, 0u) != 0) return -1;
        if (setup_hashes(0xB1000000u, nt, &d_h) != 0) return -1;
        bm_create_kernel<<<gr, bl>>>(bm.vfs, d_h, nt);
        cudaDeviceSynchronize();

        float ms_ckpt = timed_write(&bm, d_h, wbuf, gr, bl, nt, BM1_WRITES_PER_THREAD);
        rows[ti].ckpt_ms   = (double)ms_ckpt;
        rows[ti].ckpt_tput = (double)total / ((double)ms_ckpt / 1000.0) / (1024.0*1024.0);
        printf("  %-10u  %-12s  %16.2f  %10.3f\n",
               nt, "Checkpoint", rows[ti].ckpt_tput, rows[ti].ckpt_ms);
        cudaFree(d_h);
        bm_fs_cleanup(&bm);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* BM2: Per-write latency distribution                                */
/* ------------------------------------------------------------------ */

static int run_bm2(const uint8_t *wbuf, int rate_khz,
                   bm2_stats_t *cow_out, bm2_stats_t *ckpt_out)
{
    printf("\n### BM2: Per-write Latency Distribution "
           "(%u threads × %u writes, %zu samples) ###\n",
           BM2_NTHREADS, BM2_WRITES_PER_THREAD, BM2_TOTAL_SAMPLES);
    printf("  %-12s  %9s  %9s  %9s  %10s  %10s  (ns)\n",
           "Mode", "P50", "P95", "P99", "P99.9", "Max");

    uint32_t gr, bl;
    grid_block(BM2_NTHREADS, &gr, &bl);

    static const char    *MNAMES[] = {"COW", "Checkpoint"};
    static const uint32_t UCOWS[]  = {1u, 0u};
    bm2_stats_t *outs[] = {cow_out, ckpt_out};

    for (uint32_t mi = 0; mi < 2u; ++mi) {
        bm_fs_t bm;
        if (bm_fs_init(&bm, UCOWS[mi]) != 0) return -1;

        uint32_t *d_h   = NULL;
        uint64_t *d_lat = NULL;
        if (setup_hashes(0xC0000000u + mi * 0x100000u, BM2_NTHREADS, &d_h) != 0) return -1;
        cudaMallocManaged((void **)&d_lat, BM2_TOTAL_SAMPLES * sizeof(uint64_t));
        memset(d_lat, 0, BM2_TOTAL_SAMPLES * sizeof(uint64_t));

        bm_create_kernel<<<gr, bl>>>(bm.vfs, d_h, BM2_NTHREADS);
        cudaDeviceSynchronize();
        bm_latency_kernel<<<gr, bl>>>(bm.vfs, d_h, wbuf,
                                       BM2_NTHREADS, BM2_WRITES_PER_THREAD, d_lat);
        cudaDeviceSynchronize();

        *outs[mi] = compute_stats(d_lat, BM2_TOTAL_SAMPLES, rate_khz);
        print_latency_row(outs[mi], MNAMES[mi]);

        cudaFree(d_h);
        cudaFree(d_lat);
        bm_fs_cleanup(&bm);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* BM3: Checkpoint stall analysis                                     */
/* ------------------------------------------------------------------ */

static const uint32_t BM3_DCOUNTS[BM3_ND_COUNT] = {64, 128, 256, 512, 1024};

/*
 * run_bm3: true stop-the-world model.
 *
 * Writes are split into BM3_NUM_BATCHES batches of BM3_BATCH_SIZE pages each.
 * After every batch, all GPU threads complete (cudaDeviceSynchronize), then
 * checkpoint runs — no writes can proceed until checkpoint finishes.
 * This mirrors F2FS's stop-the-world: ongoing writes are blocked for the
 * duration of checkpoint, repeated every BATCH_SIZE writes per thread.
 *
 * COW mode writes all pages in one shot (no checkpoint, no stall).
 *
 * Metrics:
 *   cow_ms       — total COW write time (all BM3_WRITES_PER_FILE pages)
 *   ckpt_write_ms — sum of batch write kernel times across all batches
 *   ckpt_stall_ms — sum of checkpoint wall-clock times across all batches
 *   ckpt_total_ms — ckpt_write_ms + ckpt_stall_ms
 */
static int run_bm3(const uint8_t *wbuf, bm3_row_t *rows)
{
    printf("\n### BM3: Checkpoint Stop-the-World Stall ###\n");
    printf("    %u writes/file split into %u batches of %u, "
           "checkpoint after each batch\n",
           BM3_WRITES_PER_FILE, BM3_NUM_BATCHES, BM3_BATCH_SIZE);
    printf("  %-12s  %10s  %14s  %14s  %14s  %10s\n",
           "DirtyInodes", "cow_ms", "ckpt_write_ms",
           "ckpt_stall_ms", "ckpt_total_ms", "stall/cow");

    for (uint32_t di = 0; di < BM3_ND_COUNT; ++di) {
        uint32_t nd = BM3_DCOUNTS[di];
        rows[di].nd = nd;

        uint32_t gr, bl;
        grid_block(nd, &gr, &bl);

        /* ---- COW mode: write all pages in one shot, no checkpoint ---- */
        {
            bm_fs_t cow;
            if (bm_fs_init(&cow, 1u) != 0) return -1;
            uint32_t *d_hc = NULL;
            if (setup_hashes(0xD0000000u, nd, &d_hc) != 0) return -1;
            bm_create_kernel<<<gr, bl>>>(cow.vfs, d_hc, nd);
            cudaDeviceSynchronize();
            rows[di].cow_ms = (double)timed_write(&cow, d_hc, wbuf, gr, bl,
                                                   nd, BM3_WRITES_PER_FILE);
            cudaFree(d_hc);
            bm_fs_cleanup(&cow);
        }

        /* ---- Checkpoint mode: batched writes + checkpoint per batch ---- */
        {
            bm_fs_t ckpt;
            if (bm_fs_init(&ckpt, 0u) != 0) return -1;
            uint32_t *d_hk = NULL;
            if (setup_hashes(0xD1000000u, nd, &d_hk) != 0) return -1;
            bm_create_kernel<<<gr, bl>>>(ckpt.vfs, d_hk, nd);
            cudaDeviceSynchronize();

            uint32_t *d_err = NULL;
            cudaMallocManaged((void **)&d_err, sizeof(uint32_t));

            double total_write_ms = 0.0, total_stall_ms = 0.0;

            for (uint32_t b = 0; b < BM3_NUM_BATCHES; ++b) {
                uint32_t pgoff_start = b * BM3_BATCH_SIZE;
                *d_err = 0;

                /* Write one batch — all nd threads write BM3_BATCH_SIZE pages. */
                cudaEvent_t ev0, ev1;
                cudaEventCreate(&ev0); cudaEventCreate(&ev1);
                cudaEventRecord(ev0);
                bm3_batch_write_kernel<<<gr, bl>>>(
                        ckpt.vfs, d_hk, wbuf, nd,
                        pgoff_start, BM3_BATCH_SIZE, d_err);
                cudaEventRecord(ev1);
                cudaDeviceSynchronize();   /* all threads finish this batch */

                float batch_ms = 0.0f;
                cudaEventElapsedTime(&batch_ms, ev0, ev1);
                cudaEventDestroy(ev0); cudaEventDestroy(ev1);
                total_write_ms += (double)batch_ms;

                if (*d_err > 0)
                    fprintf(stderr, "  [WARNING] batch %u: %u write errors\n",
                            b, *d_err);

                /*
                 * Stop-the-world: all write threads are now idle.
                 * Checkpoint runs — new writes are impossible until it
                 * completes, modelling F2FS's global write suspension.
                 */
                struct timespec ts0, ts1;
                clock_gettime(CLOCK_MONOTONIC, &ts0);
                gpu_f2fs_do_checkpoint(ckpt.fs);
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                total_stall_ms += (ts1.tv_sec  - ts0.tv_sec)  * 1000.0
                                + (ts1.tv_nsec - ts0.tv_nsec) / 1.0e6;
            }

            rows[di].ckpt_write_ms = total_write_ms;
            rows[di].ckpt_stall_ms = total_stall_ms;
            rows[di].ckpt_total_ms = total_write_ms + total_stall_ms;

            cudaFree(d_err);
            cudaFree(d_hk);
            bm_fs_cleanup(&ckpt);
        }

        printf("  %-12u  %10.3f  %14.3f  %14.3f  %14.3f  %9.2f×\n",
               nd,
               rows[di].cow_ms,
               rows[di].ckpt_write_ms,
               rows[di].ckpt_stall_ms,
               rows[di].ckpt_total_ms,
               rows[di].ckpt_stall_ms / rows[di].cow_ms);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Analysis: printed after all BMs complete                           */
/* ------------------------------------------------------------------ */

static void print_analysis(const bm1_row_t   *b1,
                            const bm2_stats_t *b2_cow,
                            const bm2_stats_t *b2_ckpt,
                            const bm3_row_t   *b3)
{
    printf("\n");
    printf("##########################################################\n");
    printf("## Result Analysis                                      ##\n");
    printf("##########################################################\n");

    /* ---- BM1 analysis: per-thread-count overhead ---- */
    printf("\n[BM1] Checkpoint vs. COW Throughput Ratio "
           "(ckpt_tput / cow_tput)\n");
    printf("  %-10s  %16s  %16s  %12s\n",
           "Threads", "COW(MB/s)", "Ckpt(MB/s)", "Ratio(ckpt/cow)");

    double sum_ratio = 0.0;
    uint32_t valid = 0;
    double peak_cow = 0.0, peak_ckpt = 0.0;
    uint32_t peak_cow_nt = 0, peak_ckpt_nt = 0;

    for (uint32_t ti = 0; ti < BM1_NT_COUNT; ++ti) {
        double ratio = b1[ti].ckpt_tput / b1[ti].cow_tput;
        printf("  %-10u  %16.2f  %16.2f  %11.2f×\n",
               b1[ti].nt, b1[ti].cow_tput, b1[ti].ckpt_tput, ratio);
        sum_ratio += ratio;
        valid++;
        if (b1[ti].cow_tput  > peak_cow)  { peak_cow  = b1[ti].cow_tput;  peak_cow_nt  = b1[ti].nt; }
        if (b1[ti].ckpt_tput > peak_ckpt) { peak_ckpt = b1[ti].ckpt_tput; peak_ckpt_nt = b1[ti].nt; }
    }
    double avg_ratio = (valid > 0) ? sum_ratio / valid : 1.0;

    printf("\n  Key findings:\n");
    printf("  · Average Checkpoint/COW throughput ratio: %.2f×\n", avg_ratio);
    printf("    (>1 means Checkpoint writes are faster due to 1 fewer PM write\n"
           "     and 1 fewer __threadfence_system() drain per vfs_write call)\n");
    printf("  · Peak COW       throughput: %.1f MB/s @ %u threads\n",
           peak_cow, peak_cow_nt);
    printf("  · Peak Checkpoint throughput: %.1f MB/s @ %u threads\n",
           peak_ckpt, peak_ckpt_nt);

    /* ---- BM2 analysis: latency overhead ---- */
    printf("\n[BM2] COW per-write Latency Overhead vs. Checkpoint\n");
    printf("  %-14s  %9s  %9s  %9s  %10s  %10s  (ns)\n",
           "Mode", "P50", "P95", "P99", "P99.9", "Max");
    print_latency_row(b2_cow,  "COW");
    print_latency_row(b2_ckpt, "Checkpoint");

    printf("\n  Overhead (COW - Checkpoint) due to extra PM write + drain:\n");
    printf("  %-14s  %9.0f  %9.0f  %9.0f  %10.0f  %10.0f  (ns)\n",
           "Overhead",
           b2_cow->p50  - b2_ckpt->p50,
           b2_cow->p95  - b2_ckpt->p95,
           b2_cow->p99  - b2_ckpt->p99,
           b2_cow->p999 - b2_ckpt->p999,
           b2_cow->pmax - b2_ckpt->pmax);

    if (b2_ckpt->p99 > 0.0) {
        printf("\n  P99 overhead ratio (COW/Ckpt): %.2f×\n",
               b2_cow->p99 / b2_ckpt->p99);
        printf("  Note: COW overhead is the per-write price to eliminate\n"
               "        stop-the-world checkpoint stalls entirely.\n");
    }

    /* ---- BM3 analysis: stall impact ---- */
    printf("\n[BM3] Checkpoint Stall (%u checkpoints per run, "
           "%u writes/thread/batch)\n",
           BM3_NUM_BATCHES, BM3_BATCH_SIZE);
    printf("  %-12s  %10s  %14s  %14s  %10s  %14s\n",
           "DirtyInodes", "cow_ms",
           "ckpt_total_ms", "ckpt_stall_ms",
           "stall/cow", "stall% of total");

    double max_stall_ratio = 0.0;
    uint32_t max_stall_nd = 0;
    int cow_always_better = 1;

    for (uint32_t di = 0; di < BM3_ND_COUNT; ++di) {
        double stall_ratio  = b3[di].ckpt_stall_ms / b3[di].cow_ms;
        double stall_pct    = 100.0 * b3[di].ckpt_stall_ms / b3[di].ckpt_total_ms;
        double total_ratio  = b3[di].ckpt_total_ms / b3[di].cow_ms;

        printf("  %-12u  %10.3f  %14.3f  %14.3f  %9.2f×  %13.1f%%\n",
               b3[di].nd, b3[di].cow_ms,
               b3[di].ckpt_total_ms, b3[di].ckpt_stall_ms,
               stall_ratio, stall_pct);

        if (stall_ratio > max_stall_ratio) {
            max_stall_ratio = stall_ratio;
            max_stall_nd    = b3[di].nd;
        }
        if (total_ratio < 1.0) cow_always_better = 0;
    }

    printf("\n  Key findings:\n");
    printf("  · Max stall/cow ratio: %.2f× at %u dirty inodes\n",
           max_stall_ratio, max_stall_nd);
    printf("    (The stall alone is %.2f× the total COW write time for\n"
           "     the same workload — every GPU thread is blocked during this.)\n",
           max_stall_ratio);
    printf("  · Checkpoint ckpt_total_ms = write_ms + stall_ms is %s\n"
           "    than COW write time across all dirty inode counts tested.\n",
           cow_always_better ? "HIGHER" : "not always higher");

    /*
     * Stall penalty per dirty inode (linear fit: slope = stall / nd).
     * Gives intuition for how stall scales with FS concurrency.
     */
    {
        double sum_stall_per_nd = 0.0;
        for (uint32_t di = 0; di < BM3_ND_COUNT; ++di)
            sum_stall_per_nd += b3[di].ckpt_stall_ms / (double)b3[di].nd;
        double avg_us_per_inode =
            (sum_stall_per_nd / BM3_ND_COUNT) * 1000.0; /* ms→µs */
        printf("  · Average checkpoint cost per dirty inode: %.2f µs\n",
               avg_us_per_inode);
    }

    /* ---- Overall conclusion ---- */
    printf("\n[Summary] COW vs. Checkpoint trade-off:\n");
    printf("  COW mode incurs ~%.2f× per-write latency overhead (P99)\n",
           (b2_ckpt->p99 > 0.0) ? b2_cow->p99 / b2_ckpt->p99 : 0.0);
    printf("  but eliminates stop-the-world stalls entirely.\n");
    printf("  Total checkpoint stall (~%.1f ms at 1024 dirty inodes,\n"
           "  %u stop-the-world events) equals %.2f× the COW write time\n",
           b3[BM3_ND_COUNT - 1].ckpt_stall_ms,
           BM3_NUM_BATCHES,
           b3[BM3_ND_COUNT - 1].ckpt_stall_ms /
               (b3[BM3_ND_COUNT - 1].cow_ms > 0 ?
                b3[BM3_ND_COUNT - 1].cow_ms : 1.0));
    printf("  for the same workload.  Each stall blocks ALL GPU threads;\n"
           "  COW eliminates these stalls by persisting every inode write\n"
           "  immediately, at the cost of higher per-write PM overhead.\n");
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("==========================================================\n");
    printf("   beaver_gpu  Phase 5 — Performance Benchmarks\n");
    printf("==========================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU  : %s  (compute %d.%d)\n", prop.name, prop.major, prop.minor);

    int rate_khz = 0;
    cudaDeviceGetAttribute(&rate_khz, cudaDevAttrClockRate, 0);
    printf("Clock: %d kHz  (used for ns conversion in BM2)\n\n", rate_khz);

    uint8_t gpu_bus = ddio_get_gpu_bus();
    printf("PCIe bus 0x%02x — disabling DDIO\n\n", gpu_bus);
    ddio_disable(gpu_bus);

    /* Shared write buffer */
    uint8_t *wbuf = NULL;
    CUDA_CHECK(cudaMallocManaged((void **)&wbuf, BEAVER_PAGE_SIZE));
    fill_page(wbuf, 0xDEADBEEFCAFEBABEULL);

    /* Result storage */
    bm1_row_t   bm1[BM1_NT_COUNT];
    bm2_stats_t bm2_cow, bm2_ckpt;
    bm3_row_t   bm3[BM3_ND_COUNT];
    memset(bm1, 0, sizeof(bm1));
    memset(&bm2_cow,  0, sizeof(bm2_cow));
    memset(&bm2_ckpt, 0, sizeof(bm2_ckpt));
    memset(bm3, 0, sizeof(bm3));

    int rc = 0;
    if (run_bm1(wbuf,              bm1)               != 0) { rc = 1; goto done; }
    if (run_bm2(wbuf, rate_khz, &bm2_cow, &bm2_ckpt) != 0) { rc = 1; goto done; }
    if (run_bm3(wbuf,              bm3)               != 0) { rc = 1; goto done; }

    print_analysis(bm1, &bm2_cow, &bm2_ckpt, bm3);

    printf("\n==========================================================\n");
    printf("   Benchmark + Analysis complete.\n");
    printf("==========================================================\n");

done:
    cudaFree(wbuf);
    ddio_enable(gpu_bus);
    printf("DDIO restored.\n");
    return rc;
}
