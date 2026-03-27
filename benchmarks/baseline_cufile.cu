/*
 * baseline_cufile.cu — cuFile (GDS) baseline: multi-threaded, crash-consistent.
 *
 * Data path:
 *   GPU VRAM --(cuFileWrite/GDS)--> ext4-dax --> PM (/mnt/pmem)
 *   No cudaMemcpy: data moves directly from GPU VRAM to PM via GDS.
 *
 * Design mirrors baseline_basic.cu exactly:
 *   - nthreads CPU workers, each handles MB_TOTAL_FILES/nthreads files
 *   - write-then-rename for crash consistency (same guarantee as pwrite+rename)
 *   - 32 × 4KB cuFileWrite per file (aligned with Beaver and pwrite+rename)
 *   - fsync(file) + rename + fsync(dir) for complete persistence
 *   - NUMA-0 binding (Socket 2, where /mnt/pmem lives)
 *
 * Per-file overhead (cuFileHandleRegister/Deregister) is included in timing
 * as it is inherent to GDS and a real cost of the API.
 *
 * All three workloads use 4096 separate files (not single large file), so
 * metadata overhead is equivalent to pwrite+rename.
 *
 * Note on GDS mode: if the nvidia-fs kernel module is not loaded or P2P DMA
 * to fsdax is unsupported, cuFile silently falls back to a bounce-buffer path
 * through CPU DRAM.  Check "max_device_cache" in the driver properties output:
 * a non-zero value indicates compat/legacy mode rather than true P2P DMA.
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

#if __has_include(<cufile.h>)
#  include <cufile.h>
#  define CUFILE_AVAILABLE 1
#else
#  define CUFILE_AVAILABLE 0
#endif

/* ── Module state ────────────────────────────────────────────────── */

/*
 * Shared registered GPU buffer (MB_DATA_PER_THREAD = 128 KiB).
 * All workers read from this buffer concurrently (read-only, no races).
 * buf_off selects which 4KB page to read for each cuFileWrite call.
 */
static uint8_t *g_devbuf  = NULL;
static int      g_cf_open = 0;
static int      g_dir_fd  = -1;

/* ── NUMA-0 cpuset (/mnt/pmem is on Socket 2 = NUMA node 0) ─────── */

static cpu_set_t g_numa0_cpuset;
static int       g_numa0_valid = 0;

static void cufile_cpuset_init(void)
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
}

static uint32_t cufile_nworkers(uint32_t nthreads)
{
    uint32_t hw = g_numa0_valid ? (uint32_t)CPU_COUNT(&g_numa0_cpuset) : 48u;
    return (nthreads < hw) ? nthreads : hw;
}

/* ── File path helpers ───────────────────────────────────────────── */

static inline void cf_make_path(char *buf, size_t sz, uint32_t idx)
{
    snprintf(buf, sz, MB_FILE_FMT, idx);
}

static inline void cf_make_tmp_path(char *buf, size_t sz, uint32_t idx)
{
    snprintf(buf, sz, MB_PMEM_MOUNT "/cf_%06u.tmp", idx);
}

static void cf_unlink_files(uint32_t n)
{
    char path[256];
    for (uint32_t i = 0; i < n; i++) {
        cf_make_path    (path, sizeof path, i); unlink(path);
        cf_make_tmp_path(path, sizeof path, i); unlink(path);
    }
}

/* ── Work partition helper ───────────────────────────────────────── */

static void cf_divide_work(uint32_t total, uint32_t nw,
                           uint32_t *starts, uint32_t *counts)
{
    uint32_t base = 0;
    for (uint32_t i = 0; i < nw; i++) {
        counts[i] = (total - base) / (nw - i);
        starts[i] = base;
        base += counts[i];
    }
}

/* ── Worker ──────────────────────────────────────────────────────── */

typedef struct {
    uint32_t file_start;
    uint32_t file_count;
    int      rand_order;   /* 0 = sequential pages, 1 = random page order */
} cufile_worker_arg_t;

#if CUFILE_AVAILABLE
static void *cufile_worker(void *a_)
{
    cufile_worker_arg_t *a = (cufile_worker_arg_t *)a_;
    if (g_numa0_valid)
        pthread_setaffinity_np(pthread_self(),
                               sizeof g_numa0_cpuset, &g_numa0_cpuset);

    char final_path[256], tmp_path[256];
    uint32_t order[MB_WRITES_PER_THREAD];

    for (uint32_t i = a->file_start; i < a->file_start + a->file_count; i++) {
        cf_make_path    (final_path, sizeof final_path, i);
        cf_make_tmp_path(tmp_path,   sizeof tmp_path,   i);

        /* Page order: sequential or random Fisher-Yates (same seed as Beaver) */
        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;
        if (a->rand_order) {
            uint32_t rng = i * 2654435761u + 1u;
            for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
                rng = rng * 1664525u + 1013904223u;
                uint32_t j   = rng % (p + 1);
                uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
            }
        }

        /* open tmp, pre-allocate full file size (required for random-order writes) */
        int fd = open(tmp_path, O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (fd < 0) { perror("[cuFile] open tmp"); continue; }
        if (ftruncate(fd, (off_t)MB_DATA_PER_THREAD) < 0) {
            perror("[cuFile] ftruncate");
            close(fd); unlink(tmp_path); continue;
        }

        /* Register file handle with GDS driver */
        CUfileDescr_t  descr  = {};
        CUfileHandle_t handle = {};
        descr.handle.fd = fd;
        descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        CUfileError_t err = cuFileHandleRegister(&handle, &descr);
        if (err.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "[cuFile] cuFileHandleRegister failed: %d\n",
                    (int)err.err);
            close(fd); unlink(tmp_path); continue;
        }

        /*
         * 32 × 4KB cuFileWrite — reads from the shared registered GPU buffer.
         * buf_off selects the corresponding source page within g_devbuf.
         * file_off is the destination offset in the tmp file.
         * Write granularity matches Beaver (MB_PAGE_SIZE = 4KB per call).
         */
        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) {
            off_t file_off = (off_t)order[p] * MB_PAGE_SIZE;
            off_t buf_off  = (off_t)order[p] * MB_PAGE_SIZE;
            cuFileWrite(handle, g_devbuf, MB_PAGE_SIZE, file_off, buf_off);
        }

        cuFileHandleDeregister(handle);

        fsync(fd);            /* persist file data + inode to PM */
        close(fd);

        rename(tmp_path, final_path);   /* atomic directory entry swap */
        if (g_dir_fd >= 0)
            fsync(g_dir_fd);            /* persist directory entry to PM */
    }
    return NULL;
}
#endif  /* CUFILE_AVAILABLE */

/* ── Parallel launcher ───────────────────────────────────────────── */

static void cf_launch_workers(uint32_t nw, cufile_worker_arg_t *args)
{
#if CUFILE_AVAILABLE
    pthread_t *tids = (pthread_t *)malloc(nw * sizeof(pthread_t));
    for (uint32_t i = 0; i < nw; i++)
        pthread_create(&tids[i], NULL, cufile_worker, &args[i]);
    for (uint32_t i = 0; i < nw; i++)
        pthread_join(tids[i], NULL);
    free(tids);
#else
    (void)nw; (void)args;
#endif
}

/* ── Init / cleanup ──────────────────────────────────────────────── */

int cufile_init(void)
{
#if !CUFILE_AVAILABLE
    fprintf(stderr, "[cuFile] cufile.h not found in CUDA toolkit include path.\n"
                    "         Install nvidia-fs / GDS package and re-build.\n");
    return -1;
#else
    cufile_cpuset_init();

    CUfileError_t err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[cuFile] cuFileDriverOpen() failed: err=%d\n"
                        "         Possible reasons: nvidia-fs kernel module not loaded,\n"
                        "         or GDS not supported on this kernel/driver version.\n",
                (int)err.err);
        return -1;
    }

    CUfileDrvProps_t props;
    memset(&props, 0, sizeof props);
    if (cuFileDriverGetProperties(&props).err == CU_FILE_SUCCESS)
        fprintf(stdout, "[cuFile] driver open OK  max_device_cache=%zu KB\n",
                (size_t)props.max_device_cache_size / 1024);

    if (cudaMalloc((void **)&g_devbuf, MB_DATA_PER_THREAD) != cudaSuccess) {
        fprintf(stderr, "[cuFile] cudaMalloc for device buffer failed\n");
        cuFileDriverClose();
        return -1;
    }
    cudaMemset(g_devbuf, 0xCD, MB_DATA_PER_THREAD);

    CUfileError_t berr = cuFileBufRegister(g_devbuf, MB_DATA_PER_THREAD, 0);
    if (berr.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[cuFile] cuFileBufRegister failed: err=%d\n"
                        "         GDS P2P DMA to fsdax may not be supported.\n",
                (int)berr.err);
        cudaFree(g_devbuf); g_devbuf = NULL;
        cuFileDriverClose();
        return -1;
    }

    g_dir_fd = open(MB_PMEM_MOUNT, O_RDONLY | O_DIRECTORY);
    if (g_dir_fd < 0)
        fprintf(stderr, "[cuFile] WARNING: cannot open dir fd for fsync(dir) — "
                        "directory entries may not be persisted\n");

    g_cf_open = 1;
    return 0;
#endif
}

void cufile_cleanup(void)
{
#if CUFILE_AVAILABLE
    if (g_devbuf) {
        cuFileBufDeregister(g_devbuf);
        cudaFree(g_devbuf);
        g_devbuf = NULL;
    }
    if (g_dir_fd >= 0) { close(g_dir_fd); g_dir_fd = -1; }
    if (g_cf_open)     { cuFileDriverClose(); g_cf_open = 0; }
#endif
}

/* ── Generic run helper ──────────────────────────────────────────── */

/*
 * All three workloads use write-then-rename for crash consistency.
 * Timing covers: cuFileHandleRegister + cuFileWrite×32 + cuFileHandleDeregister
 *                + fsync(file) + rename + fsync(dir).
 * This is the full end-to-end cost of crash-consistently writing a new file
 * from GPU data to PM via GDS — equivalent to Beaver's and pwrite+rename paths.
 */
static double cufile_run_impl(uint32_t nthreads, int rand_order, const char *tag)
{
#if !CUFILE_AVAILABLE
    (void)nthreads; (void)rand_order; (void)tag; return -1.0;
#else
    if (!g_cf_open) return -1.0;

    uint32_t nw = cufile_nworkers(nthreads);
    if (getenv("VERBOSE"))
        fprintf(stderr, "[cuFile/%s] workers=%-4u  files=%u  rand=%d\n",
                tag, nw, MB_TOTAL_FILES, rand_order);

    cufile_worker_arg_t *args   = (cufile_worker_arg_t *)malloc(nw * sizeof *args);
    uint32_t            *starts = (uint32_t *)malloc(nw * sizeof *starts);
    uint32_t            *counts = (uint32_t *)malloc(nw * sizeof *counts);
    if (!args || !starts || !counts) {
        free(args); free(starts); free(counts); return -1.0;
    }
    cf_divide_work(MB_TOTAL_FILES, nw, starts, counts);
    for (uint32_t i = 0; i < nw; i++)
        args[i] = (cufile_worker_arg_t){ starts[i], counts[i], rand_order };

    /* warm-up (not timed) */
    cf_launch_workers(nw, args);
    cf_unlink_files(MB_TOTAL_FILES);

    mb_timer_t t;
    mb_timer_start(&t);
    cf_launch_workers(nw, args);
    double ms = mb_timer_elapsed_ms(&t);

    free(args); free(starts); free(counts);
    cf_unlink_files(MB_TOTAL_FILES);
    return mb_throughput(ms);
#endif
}

/* ── MB_SEQ / MB_RAND / MB_MULTI ─────────────────────────────────── */

double cufile_run_seq  (uint32_t n) { return cufile_run_impl(n, 0, "Seq");   }
double cufile_run_rand (uint32_t n) { return cufile_run_impl(n, 1, "Rand");  }
double cufile_run_multi(uint32_t n) { return cufile_run_impl(n, 0, "Multi"); }

/* ── Dispatch ────────────────────────────────────────────────────── */

double cufile_run(mb_workload_t wl, uint32_t nthreads)
{
    switch (wl) {
    case MB_SEQ:   return cufile_run_seq  (nthreads);
    case MB_RAND:  return cufile_run_rand (nthreads);
    case MB_MULTI: return cufile_run_multi(nthreads);
    default:       return -1.0;
    }
}
