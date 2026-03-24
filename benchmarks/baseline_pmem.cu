/*
 * baseline_pmem.cu — CPU pmem_memcpy_persist() baseline (strict GPU→CPU→PM).
 *
 * Data path (strict, timed end-to-end):
 *   GPU VRAM --(pageable cudaMemcpy)--> CPU DRAM --(pmem_memcpy_persist)--> PM
 *
 * The cudaMemcpy is included in timing (same as baseline_basic) so both
 * CPU-path baselines reflect the true cost of moving GPU data to PM.
 * pmem_memcpy_persist() = MOVNTI + SFENCE — no fsync, direct to PM.
 * Workers bind to NUMA-0 (Socket 2, co-located with /mnt/pmem).
 */

#include "bench_config.h"
#include "bench_utils.h"

#include <libpmem.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Module state ────────────────────────────────────────────────── */

#define PMEM_MAP_FILE  MB_PMEM_MOUNT "/pmem_bench_direct.dat"

/* Pageable staging buffer: MB_TOTAL_FILES × MB_DATA_PER_THREAD = 512 MiB. */
static uint8_t *g_src    = NULL;   /* pageable CPU staging, MB_TOTAL_FILES × 128 KiB */
static uint8_t *g_devbuf = NULL;   /* GPU source, same size                          */

/* ── NUMA-0 cpuset (/mnt/pmem is on Socket 2 = NUMA node 0) ─────── */

static cpu_set_t g_numa0_cpuset;
static int       g_numa0_valid = 0;

static void pmem_cpuset_init(void)
{
    CPU_ZERO(&g_numa0_cpuset);
    FILE *f = fopen("/sys/devices/system/node/node0/cpulist", "r");
    if (!f) return;
    char buf[512];
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return; }
    fclose(f);
    for (char *p = buf; *p && *p != '\n'; ) {
        int lo = (int)strtol(p, &p, 10);
        if (*p == '-') {
            ++p;
            int hi = (int)strtol(p, &p, 10);
            for (int c = lo; c <= hi; c++) CPU_SET(c, &g_numa0_cpuset);
        } else {
            CPU_SET(lo, &g_numa0_cpuset);
        }
        if (*p == ',') ++p;
    }
    g_numa0_valid = (CPU_COUNT(&g_numa0_cpuset) > 0);
    fprintf(stderr, "[pmem] NUMA-0 cpuset: %d CPUs  (binding workers to Socket 2)\n",
            g_numa0_valid ? CPU_COUNT(&g_numa0_cpuset) : 0);
}

static uint32_t pmem_nworkers(uint32_t nthreads)
{
    uint32_t hw = g_numa0_valid ? (uint32_t)CPU_COUNT(&g_numa0_cpuset) : 48u;
    return (nthreads < hw) ? nthreads : hw;
}

/* ── Init / cleanup ──────────────────────────────────────────────── */

int pmem_bench_init(void)
{
    pmem_cpuset_init();
    size_t total = (size_t)MB_TOTAL_FILES * MB_DATA_PER_THREAD;  /* 512 MiB */
    g_src = (uint8_t *)malloc(total);
    if (!g_src) { fprintf(stderr, "[pmem] malloc g_src failed\n"); return -1; }
    if (cudaMalloc((void **)&g_devbuf, total) != cudaSuccess) {
        fprintf(stderr, "[pmem] cudaMalloc devbuf failed\n");
        free(g_src); g_src = NULL; return -1;
    }
    memset(g_src, 0xCC, total);
    cudaMemcpy(g_devbuf, g_src, total, cudaMemcpyHostToDevice);
    return 0;
}

void pmem_bench_cleanup(void)
{
    if (g_src)    { free(g_src);       g_src    = NULL; }
    if (g_devbuf) { cudaFree(g_devbuf); g_devbuf = NULL; }
    remove(PMEM_MAP_FILE);
}

/* ── Worker ──────────────────────────────────────────────────────── */

typedef struct {
    void           *pm_base;
    const uint8_t  *src_base;  /* base of staging buffer; per-region offset = i * MB_DATA_PER_THREAD */
    uint32_t        start_idx;
    uint32_t        count;
} pmem_worker_arg_t;

static void *pmem_worker(void *a_)
{
    pmem_worker_arg_t *a = (pmem_worker_arg_t *)a_;
    if (g_numa0_valid)
        pthread_setaffinity_np(pthread_self(),
                               sizeof g_numa0_cpuset, &g_numa0_cpuset);
    for (uint32_t i = a->start_idx; i < a->start_idx + a->count; i++) {
        char       *dst = (char *)a->pm_base   + (size_t)i * MB_DATA_PER_THREAD;
        const void *src = a->src_base          + (size_t)i * MB_DATA_PER_THREAD;
        pmem_memcpy_persist(dst, src, MB_DATA_PER_THREAD);
    }
    return NULL;
}

/* ── Core parallel persist ───────────────────────────────────────── */

/* nthreads = CPU workers; total regions = MB_TOTAL_FILES (fixed). */
static double pmem_run_parallel(uint32_t nthreads, const char *tag)
{
    if (!g_src || !g_devbuf) return -1.0;

    /* Fixed 512 MiB transfer + persist regardless of worker count */
    size_t xfer = (size_t)MB_TOTAL_FILES * MB_DATA_PER_THREAD;
    size_t mapped_len = 0;
    int    is_pmem    = 0;
    void  *pm = pmem_map_file(PMEM_MAP_FILE, xfer, PMEM_FILE_CREATE, 0644,
                               &mapped_len, &is_pmem);
    if (!pm) { perror("[pmem] pmem_map_file"); return -1.0; }
    if (!is_pmem)
        fprintf(stderr, "[pmem] WARNING: is_pmem=0 (writes may not use MOVNTI)\n");

    uint32_t nw = pmem_nworkers(nthreads);
    if (getenv("VERBOSE"))
        fprintf(stderr, "[pmem/%s] workers=%-4u  regions=%u\n", tag, nw, MB_TOTAL_FILES);

    pmem_worker_arg_t *args   = (pmem_worker_arg_t *)malloc(nw * sizeof *args);
    uint32_t          *starts = (uint32_t *)malloc(nw * sizeof *starts);
    uint32_t          *counts = (uint32_t *)malloc(nw * sizeof *counts);
    pthread_t         *tids   = (pthread_t *)malloc(nw * sizeof(pthread_t));
    if (!args || !starts || !counts || !tids) {
        free(args); free(starts); free(counts); free(tids);
        pmem_unmap(pm, mapped_len); remove(PMEM_MAP_FILE);
        return -1.0;
    }

    uint32_t base = 0;
    for (uint32_t i = 0; i < nw; i++) {
        counts[i] = (MB_TOTAL_FILES - base) / (nw - i);
        starts[i] = base;
        base += counts[i];
    }
    for (uint32_t i = 0; i < nw; i++)
        args[i] = (pmem_worker_arg_t){ pm, g_src, starts[i], counts[i] };

    /* warm-up: full pipeline (GPU→CPU→PM) */
    cudaMemcpy(g_src, g_devbuf, xfer, cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < nw; i++)
        pthread_create(&tids[i], NULL, pmem_worker, &args[i]);
    for (uint32_t i = 0; i < nw; i++)
        pthread_join(tids[i], NULL);

    mb_timer_t t;
    mb_timer_start(&t);
    cudaMemcpy(g_src, g_devbuf, xfer, cudaMemcpyDeviceToHost);  /* GPU→CPU DRAM */
    for (uint32_t i = 0; i < nw; i++)
        pthread_create(&tids[i], NULL, pmem_worker, &args[i]);  /* CPU DRAM→PM  */
    for (uint32_t i = 0; i < nw; i++)
        pthread_join(tids[i], NULL);
    double ms = mb_timer_elapsed_ms(&t);

    free(tids); free(args); free(starts); free(counts);
    pmem_unmap(pm, mapped_len);
    remove(PMEM_MAP_FILE);
    return mb_throughput(ms);
}

/* ── Dispatch ────────────────────────────────────────────────────── */

double pmem_run(mb_workload_t wl, uint32_t nthreads)
{
    switch (wl) {
    case MB_SEQ:   return pmem_run_parallel(nthreads, "Seq");
    case MB_RAND:  return pmem_run_parallel(nthreads, "Rand");
    case MB_MULTI: return pmem_run_parallel(nthreads, "Multi");
    default:       return -1.0;
    }
}
