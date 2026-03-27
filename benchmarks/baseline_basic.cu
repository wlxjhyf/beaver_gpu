/*
 * baseline_basic.cu — "Basic" baseline: strict GPU→CPU DRAM→PM pipeline.
 *
 * Data path (strict, timed end-to-end):
 *   GPU VRAM --(pageable cudaMemcpy)--> CPU DRAM --(pwrite+fdatasync)--> PM
 *
 * The cudaMemcpy (nthreads × 128 KiB, pageable DeviceToHost) is included
 * in the timing window to represent the real cost of moving GPU data to CPU.
 * Each worker writes from its own distinct slice of the staging buffer so
 * the GPU-side data is unique per file (no token-copy tricks).
 *
 * Thread model: workers = min(nthreads, NUMA-0 CPU count).
 * All workers bind to NUMA-0 (Socket 2, where /mnt/pmem lives).
 */

#include "bench_config.h"
#include "bench_utils.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── Module state ────────────────────────────────────────────────── */

/* Pageable staging buffer: MB_TOTAL_FILES × MB_DATA_PER_THREAD = 512 MiB.
 * Pageable DeviceToHost cudaMemcpy (~1.3 GB/s) models the true cost of moving
 * GPU-generated data to CPU before writing to PM. */
static uint8_t *g_staging = NULL;   /* pageable CPU staging, MB_TOTAL_FILES × 128 KiB */
static uint8_t *g_devbuf  = NULL;   /* GPU source, same size                          */

/* ── NUMA-0 cpuset (/mnt/pmem is on Socket 2 = NUMA node 0) ─────── */

static cpu_set_t g_numa0_cpuset;
static int       g_numa0_valid = 0;

static void basic_cpuset_init(void)
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
    fprintf(stderr, "[Basic] NUMA-0 cpuset: %d CPUs  (binding workers to Socket 2)\n",
            g_numa0_valid ? CPU_COUNT(&g_numa0_cpuset) : 0);
}

/* Number of parallel workers: min(nthreads, NUMA-0 CPU count) */
static uint32_t basic_nworkers(uint32_t nthreads)
{
    uint32_t hw = g_numa0_valid ? (uint32_t)CPU_COUNT(&g_numa0_cpuset) : 48u;
    return (nthreads < hw) ? nthreads : hw;
}

/* ── File path helper ────────────────────────────────────────────── */

static inline void make_path(char *buf, size_t sz, uint32_t idx)
{
    snprintf(buf, sz, MB_FILE_FMT, idx);
}

/* ── Temp path helper (write-then-rename) ────────────────────────── */

static inline void make_tmp_path(char *buf, size_t sz, uint32_t idx)
{
    snprintf(buf, sz, MB_PMEM_MOUNT "/mb_%06u.tmp", idx);
}

static void unlink_files(uint32_t n)
{
    char path[256];
    for (uint32_t i = 0; i < n; i++) {
        make_path(path, sizeof path, i);
        unlink(path);
        make_tmp_path(path, sizeof path, i);
        unlink(path);  /* clean up any leftover tmp files */
    }
}

/* ── Work partition helper ───────────────────────────────────────── */

/* Divide nthreads files evenly among nw workers (fair residual distribution) */
static void divide_work(uint32_t nthreads, uint32_t nw,
                        uint32_t *starts, uint32_t *counts)
{
    uint32_t base = 0;
    for (uint32_t i = 0; i < nw; i++) {
        counts[i] = (nthreads - base) / (nw - i);
        starts[i] = base;
        base += counts[i];
    }
}

/*
 * Unified worker using write-then-rename for crash consistency.
 *
 * Pattern: open(tmp) → pwrite×32 (4KB each) → fsync(tmp) → rename→ fsync(dir)
 *
 * This matches Beaver's guarantee: after a crash, the final file either does
 * not exist or contains complete content — no torn writes possible.
 * rename() is a POSIX atomic operation; fsync(dir) persists the directory entry.
 *
 * Write granularity is aligned with Beaver: 32 × MB_PAGE_SIZE (4KB) per file.
 */

typedef struct {
    const uint8_t *src_base;
    uint32_t       file_start;
    uint32_t       file_count;
    int            rand_order;   /* 0 = sequential pages, 1 = random page order */
} basic_worker_arg_t;

static int g_dir_fd = -1;   /* fd for MB_PMEM_MOUNT directory, for fsync(dir) */

static void *basic_worker(void *a_)
{
    basic_worker_arg_t *a = (basic_worker_arg_t *)a_;
    if (g_numa0_valid)
        pthread_setaffinity_np(pthread_self(),
                               sizeof g_numa0_cpuset, &g_numa0_cpuset);

    char final_path[256], tmp_path[256];
    uint32_t order[MB_WRITES_PER_THREAD];

    for (uint32_t i = a->file_start; i < a->file_start + a->file_count; i++) {
        const uint8_t *src = a->src_base + (size_t)i * MB_DATA_PER_THREAD;
        make_path    (final_path, sizeof final_path, i);
        make_tmp_path(tmp_path,   sizeof tmp_path,   i);

        /* Build page order (sequential or random Fisher-Yates, same seed as Beaver) */
        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;
        if (a->rand_order) {
            uint32_t rng = i * 2654435761u + 1u;
            for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
                rng = rng * 1664525u + 1013904223u;
                uint32_t j = rng % (p + 1);
                uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
            }
        }

        /* write-then-rename: crash-consistent new-file creation */
        int fd = open(tmp_path, O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (fd < 0) { perror("[Basic] open tmp"); continue; }

        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
            pwrite(fd, src + (size_t)order[p] * MB_PAGE_SIZE, MB_PAGE_SIZE,
                   (off_t)order[p] * MB_PAGE_SIZE);

        fsync(fd);   /* persist file data + inode size */
        close(fd);

        rename(tmp_path, final_path);   /* atomic directory entry swap */

        if (g_dir_fd >= 0)
            fsync(g_dir_fd);   /* persist directory entry to PM */
    }
    return NULL;
}

/* ── Generic parallel launcher ───────────────────────────────────── */

static void launch_workers(void *(*fn)(void *), void *args, size_t arg_sz,
                            uint32_t nw)
{
    pthread_t *tids = (pthread_t *)malloc(nw * sizeof(pthread_t));
    for (uint32_t i = 0; i < nw; i++)
        pthread_create(&tids[i], NULL, fn, (char *)args + i * arg_sz);
    for (uint32_t i = 0; i < nw; i++)
        pthread_join(tids[i], NULL);
    free(tids);
}

/* ── Init / cleanup ──────────────────────────────────────────────── */

int basic_init(void)
{
    basic_cpuset_init();
    size_t total = (size_t)MB_TOTAL_FILES * MB_DATA_PER_THREAD;  /* 4096 × 128 KiB = 512 MiB */
    g_staging = (uint8_t *)malloc(total);
    if (!g_staging) { fprintf(stderr, "[Basic] malloc staging failed\n"); return -1; }
    if (cudaMalloc((void **)&g_devbuf, total) != cudaSuccess) {
        fprintf(stderr, "[Basic] cudaMalloc devbuf failed\n");
        free(g_staging); g_staging = NULL; return -1;
    }
    memset(g_staging, 0xAB, total);
    cudaMemcpy(g_devbuf, g_staging, total, cudaMemcpyHostToDevice);

    g_dir_fd = open(MB_PMEM_MOUNT, O_RDONLY | O_DIRECTORY);
    if (g_dir_fd < 0)
        fprintf(stderr, "[Basic] WARNING: cannot open dir fd for fsync(dir) — "
                        "directory entries may not be persisted\n");
    return 0;
}

void basic_cleanup(void)
{
    if (g_staging) { free(g_staging);    g_staging = NULL; }
    if (g_devbuf)  { cudaFree(g_devbuf); g_devbuf  = NULL; }
    if (g_dir_fd >= 0) { close(g_dir_fd); g_dir_fd = -1; }
}

/* ── Generic run helper ──────────────────────────────────────────── */
/*
 * All three workloads now use the same write-then-rename path.
 * Timing covers: cudaMemcpy(DevToHost) + create-tmp + pwrite×32 +
 *                fsync(file) + rename + fsync(dir).
 * This is the full end-to-end cost of crash-consistently writing a new file
 * from GPU data to PM — equivalent to Beaver's vfs_create+write+fsync path.
 */
static double basic_run_impl(uint32_t nthreads, int rand_order, const char *tag)
{
    uint32_t nw = basic_nworkers(nthreads);
    if (getenv("VERBOSE"))
        fprintf(stderr, "[Basic/%s] workers=%-4u  files=%u  rand=%d\n",
                tag, nw, MB_TOTAL_FILES, rand_order);

    size_t xfer = (size_t)MB_TOTAL_FILES * MB_DATA_PER_THREAD;

    basic_worker_arg_t *args   = (basic_worker_arg_t *)malloc(nw * sizeof *args);
    uint32_t           *starts = (uint32_t *)malloc(nw * sizeof *starts);
    uint32_t           *counts = (uint32_t *)malloc(nw * sizeof *counts);
    if (!args || !starts || !counts) {
        free(args); free(starts); free(counts); return -1.0;
    }
    divide_work(MB_TOTAL_FILES, nw, starts, counts);
    for (uint32_t i = 0; i < nw; i++)
        args[i] = (basic_worker_arg_t){ g_staging, starts[i], counts[i], rand_order };

    /* warm-up (not timed) */
    cudaMemcpy(g_staging, g_devbuf, xfer, cudaMemcpyDeviceToHost);
    launch_workers(basic_worker, args, sizeof *args, nw);
    unlink_files(MB_TOTAL_FILES);

    mb_timer_t t;
    mb_timer_start(&t);
    cudaMemcpy(g_staging, g_devbuf, xfer, cudaMemcpyDeviceToHost);  /* GPU→DRAM */
    launch_workers(basic_worker, args, sizeof *args, nw);            /* DRAM→PM  */
    double ms = mb_timer_elapsed_ms(&t);

    free(args); free(starts); free(counts);
    unlink_files(MB_TOTAL_FILES);
    return mb_throughput(ms);
}

/* ── MB_SEQ / MB_RAND / MB_MULTI ─────────────────────────────────── */

double basic_run_seq  (uint32_t nthreads) { return basic_run_impl(nthreads, 0, "Seq");   }
double basic_run_rand (uint32_t nthreads) { return basic_run_impl(nthreads, 1, "Rand");  }
double basic_run_multi(uint32_t nthreads) { return basic_run_impl(nthreads, 0, "Multi"); }

/* ── Dispatch ────────────────────────────────────────────────────── */

double basic_run(mb_workload_t wl, uint32_t nthreads)
{
    switch (wl) {
    case MB_SEQ:   return basic_run_seq  (nthreads);
    case MB_RAND:  return basic_run_rand (nthreads);
    case MB_MULTI: return basic_run_multi(nthreads);
    default:       return -1.0;
    }
}
