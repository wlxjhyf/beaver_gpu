/*
 * baseline_basic.cu — "Basic" baseline: CPU manages PM, GPU data staged via
 * cudaMemcpy then written to ext4-dax on /dev/pmem0 by a single CPU thread.
 *
 * Data path:
 *   GPU device memory  →  cudaMemcpy D→H  →  CPU pinned buffer
 *                      →  pwrite() to ext4-dax file  →  PM (/dev/pmem0)
 *
 * Timing (global, host-side clock_gettime):
 *   START: before cudaMemcpy (simulates end of GPU compute, start of persist)
 *   STOP : after last pwrite() returns
 *
 * Each of N logical "threads" corresponds to one pwrite() call of
 * MB_DATA_PER_THREAD bytes at its designated offset.  All writes are
 * issued sequentially from a single CPU thread — this is the fundamental
 * bottleneck that Basic exhibits at large N.
 *
 * MB_SEQ  : pre-created single large file, N writes at consecutive offsets.
 * MB_RAND : pre-created single large file, N writes at pseudo-random offsets.
 * MB_MULTI: N files, opened + written + closed sequentially (includes
 *           open/close metadata overhead, showing CPU metadata bottleneck).
 *
 * Run: sudo (inherits from microbench main).
 */

#include "bench_config.h"
#include "bench_utils.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Module state ────────────────────────────────────────────────── */

static uint8_t *g_staging = NULL;   /* pinned host buffer, MB_DATA_PER_THREAD */
static uint8_t *g_devbuf  = NULL;   /* device-side mirror (GPU fills this)    */

/* ── Init / cleanup ──────────────────────────────────────────────── */

int basic_init(void)
{
    if (cudaMallocHost((void **)&g_staging, MB_DATA_PER_THREAD) != cudaSuccess) {
        fprintf(stderr, "[Basic] cudaMallocHost failed\n");
        return -1;
    }
    if (cudaMalloc((void **)&g_devbuf, MB_DATA_PER_THREAD) != cudaSuccess) {
        fprintf(stderr, "[Basic] cudaMalloc failed\n");
        return -1;
    }
    /* Pre-fill with a constant pattern — content doesn't affect throughput */
    memset(g_staging, 0xAB, MB_DATA_PER_THREAD);
    cudaMemcpy(g_devbuf, g_staging, MB_DATA_PER_THREAD, cudaMemcpyHostToDevice);
    return 0;
}

void basic_cleanup(void)
{
    if (g_staging) { cudaFreeHost(g_staging); g_staging = NULL; }
    if (g_devbuf)  { cudaFree(g_devbuf);      g_devbuf  = NULL; }
}

/* ── Offset shuffle (LCG, same seed as beaver rand kernel) ──────── */

static void gen_rand_offsets(uint32_t nthreads, off_t *offsets)
{
    /* Use same LCG as mb_rand_write_kernel so patterns are comparable */
    uint32_t rng = 42u;
    off_t    file_blocks = (off_t)nthreads; /* N regions of MB_DATA_PER_THREAD */
    for (uint32_t i = 0; i < nthreads; i++) {
        rng = rng * 1664525u + 1013904223u;
        offsets[i] = (off_t)(rng % file_blocks) * MB_DATA_PER_THREAD;
    }
}

/* ── MB_SEQ ──────────────────────────────────────────────────────── */

double basic_run_seq(uint32_t nthreads)
{
    off_t total = (off_t)nthreads * MB_DATA_PER_THREAD;

    /* Pre-create and size the file (not timed) */
    int fd = open(MB_SEQ_FILE, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) { perror("[Basic/Seq] open"); return -1.0; }
    if (ftruncate(fd, total) < 0) { perror("[Basic/Seq] ftruncate"); close(fd); return -1.0; }

    mb_timer_t t;
    /* Simulate GPU→CPU handoff: copy one page-worth from device (negligible time) */
    cudaMemcpy(g_staging, g_devbuf, MB_PAGE_SIZE, cudaMemcpyDeviceToHost);
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        off_t off = (off_t)i * MB_DATA_PER_THREAD;
        if (pwrite(fd, g_staging, MB_DATA_PER_THREAD, off) < 0) {
            perror("[Basic/Seq] pwrite"); break;
        }
    }

    double ms = mb_timer_elapsed_ms(&t);
    close(fd);
    unlink(MB_SEQ_FILE);
    return ms;
}

/* ── MB_RAND ─────────────────────────────────────────────────────── */

double basic_run_rand(uint32_t nthreads)
{
    off_t total = (off_t)nthreads * MB_DATA_PER_THREAD;

    int fd = open(MB_RAND_FILE, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) { perror("[Basic/Rand] open"); return -1.0; }
    if (ftruncate(fd, total) < 0) { perror("[Basic/Rand] ftruncate"); close(fd); return -1.0; }

    off_t *offsets = (off_t *)malloc(nthreads * sizeof(off_t));
    if (!offsets) { close(fd); return -1.0; }
    gen_rand_offsets(nthreads, offsets);

    mb_timer_t t;
    cudaMemcpy(g_staging, g_devbuf, MB_PAGE_SIZE, cudaMemcpyDeviceToHost);
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        if (pwrite(fd, g_staging, MB_DATA_PER_THREAD, offsets[i]) < 0) {
            perror("[Basic/Rand] pwrite"); break;
        }
    }

    double ms = mb_timer_elapsed_ms(&t);
    free(offsets);
    close(fd);
    unlink(MB_RAND_FILE);
    return ms;
}

/* ── MB_MULTI ────────────────────────────────────────────────────── */
/*
 * For multi-file, timing includes open() + pwrite() + close() for each file,
 * making CPU metadata overhead (N syscall pairs) visible in the results.
 */
double basic_run_multi(uint32_t nthreads)
{
    char path[256];

    mb_timer_t t;
    cudaMemcpy(g_staging, g_devbuf, MB_PAGE_SIZE, cudaMemcpyDeviceToHost);
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        snprintf(path, sizeof path, MB_FILE_FMT, i);
        int fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (fd < 0) { perror("[Basic/Multi] open"); break; }
        if (pwrite(fd, g_staging, MB_DATA_PER_THREAD, 0) < 0)
            perror("[Basic/Multi] pwrite");
        close(fd);
    }

    double ms = mb_timer_elapsed_ms(&t);

    /* Cleanup files (not timed) */
    for (uint32_t i = 0; i < nthreads; i++) {
        snprintf(path, sizeof path, MB_FILE_FMT, i);
        unlink(path);
    }
    return ms;
}

/* ── Dispatch ────────────────────────────────────────────────────── */

double basic_run(mb_workload_t wl, uint32_t nthreads)
{
    double ms;
    switch (wl) {
    case MB_SEQ:   ms = basic_run_seq  (nthreads); break;
    case MB_RAND:  ms = basic_run_rand (nthreads); break;
    case MB_MULTI: ms = basic_run_multi(nthreads); break;
    default:       return -1.0;
    }
    if (ms < 0.0) return -1.0;
    return mb_throughput(nthreads, ms);
}
