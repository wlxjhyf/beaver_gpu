/*
 * baseline_pmem.cu — CPU pmem_memcpy_persist() baseline.
 *
 * Data path:
 *   CPU buffer → pmem_map_file() → pmem_memcpy_persist() → PM (/mnt/pmem0)
 *
 * pmem_memcpy_persist() = MOVNTI (non-temporal stores) + SFENCE.
 * Data bypasses CPU L1/L2/L3 cache and goes directly to PM media.
 * NO fsync() needed: persistence is guaranteed by the MOVNTI+SFENCE path.
 *
 * This represents the "honest" CPU→PM write baseline:
 *   - Same physical PM device as Basic (pwrite+fsync)
 *   - No file system overhead (no inode update, no journal)
 *   - Shows raw CPU-side PM write bandwidth via libpmem
 *
 * Mapping strategy: per-run (map → write → unmap → delete).
 * Avoids pre-allocating a large file that would fill /dev/pmem0.
 *
 * Timing: covers only the pmem_memcpy_persist() calls (no FS overhead).
 *
 * Run: sudo (inherits from microbench main, same /mnt/pmem0 requirement).
 */

#include "bench_config.h"
#include "bench_utils.h"

#include <libpmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Module state ────────────────────────────────────────────────── */

#define PMEM_MAP_FILE  MB_PMEM_MOUNT "/pmem_bench_direct.dat"

/* CPU source buffer (constant pattern, heap-allocated once) */
static void *g_src = NULL;

/* ── Init / cleanup ──────────────────────────────────────────────── */

int pmem_bench_init(void)
{
    g_src = malloc(MB_DATA_PER_THREAD);
    if (!g_src) {
        fprintf(stderr, "[pmem] malloc g_src failed\n");
        return -1;
    }
    memset(g_src, 0xCC, MB_DATA_PER_THREAD);
    fprintf(stdout, "[pmem] pmem_persist baseline ready  src=%p\n", g_src);
    return 0;
}

void pmem_bench_cleanup(void)
{
    if (g_src) { free(g_src); g_src = NULL; }
    remove(PMEM_MAP_FILE);   /* best-effort cleanup of any leftover file */
}

/* ── Per-run mapping helper ──────────────────────────────────────── */

static void *pmem_map_run(uint32_t nthreads, size_t *out_len)
{
    size_t sz = (size_t)nthreads * MB_DATA_PER_THREAD;
    int is_pmem = 0;
    void *pm = pmem_map_file(PMEM_MAP_FILE, sz, PMEM_FILE_CREATE, 0644,
                              out_len, &is_pmem);
    if (!pm) {
        perror("[pmem] pmem_map_file");
        return NULL;
    }
    if (!is_pmem)
        fprintf(stderr, "[pmem] WARNING: is_pmem=0 (writes may not use MOVNTI)\n");
    return pm;
}

/* ── MB_SEQ ──────────────────────────────────────────────────────── */
/*
 * N sequential writes to consecutive offsets.
 * Timing covers only the pmem_memcpy_persist calls (no FS syscalls).
 * No fsync needed: MOVNTI+SFENCE inside pmem_memcpy_persist guarantees PM.
 */
double pmem_run_seq(uint32_t nthreads)
{
    if (!g_src) return -1.0;

    size_t mapped_len = 0;
    void *pm = pmem_map_run(nthreads, &mapped_len);
    if (!pm) return -1.0;

    mb_timer_t t;
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        char *dst = (char *)pm + (size_t)i * MB_DATA_PER_THREAD;
        pmem_memcpy_persist(dst, g_src, MB_DATA_PER_THREAD);
    }

    double ms = mb_timer_elapsed_ms(&t);
    pmem_unmap(pm, mapped_len);
    remove(PMEM_MAP_FILE);
    return mb_throughput(nthreads, ms);
}

/* ── MB_RAND ─────────────────────────────────────────────────────── */
/*
 * N writes to pseudo-random offsets within the mapped region.
 * Same LCG seed as Basic/Beaver rand kernels.
 */
double pmem_run_rand(uint32_t nthreads)
{
    if (!g_src) return -1.0;

    size_t *offsets = (size_t *)malloc(nthreads * sizeof(size_t));
    if (!offsets) return -1.0;

    uint32_t rng = 42u;
    for (uint32_t i = 0; i < nthreads; i++) {
        rng = rng * 1664525u + 1013904223u;
        offsets[i] = (size_t)(rng % nthreads) * MB_DATA_PER_THREAD;
    }

    size_t mapped_len = 0;
    void *pm = pmem_map_run(nthreads, &mapped_len);
    if (!pm) { free(offsets); return -1.0; }

    mb_timer_t t;
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        char *dst = (char *)pm + offsets[i];
        pmem_memcpy_persist(dst, g_src, MB_DATA_PER_THREAD);
    }

    double ms = mb_timer_elapsed_ms(&t);
    pmem_unmap(pm, mapped_len);
    remove(PMEM_MAP_FILE);
    free(offsets);
    return mb_throughput(nthreads, ms);
}

/* ── MB_MULTI ────────────────────────────────────────────────────── */
/*
 * Simulates multi-file: N writes to N separate regions.
 * No per-file open/close overhead (pmem bypasses FS entirely).
 * Sequential offsets — same data layout as MB_SEQ but timed to show
 * that pmem has no metadata overhead unlike pwrite+fsync.
 */
double pmem_run_multi(uint32_t nthreads)
{
    return pmem_run_seq(nthreads);
}

/* ── Dispatch ────────────────────────────────────────────────────── */

double pmem_run(mb_workload_t wl, uint32_t nthreads)
{
    switch (wl) {
    case MB_SEQ:   return pmem_run_seq  (nthreads);
    case MB_RAND:  return pmem_run_rand (nthreads);
    case MB_MULTI: return pmem_run_multi(nthreads);
    default:       return -1.0;
    }
}
