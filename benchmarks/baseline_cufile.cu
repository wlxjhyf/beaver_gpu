/*
 * baseline_cufile.cu — cuFile (GDS) baseline on /dev/pmem0 (fsdax / ext4-dax).
 *
 * Data path:
 *   GPU device memory  →  cuFileWrite()  →  PM (/dev/pmem0 via ext4-dax)
 *   (CPU calls cuFileWrite; GDS driver issues DMA if supported, else legacy)
 *
 * Timing (global, host-side clock_gettime):
 *   START: before first cuFileWrite
 *   STOP : after last cuFileWrite returns
 *
 * If cuFileDriverOpen() fails or the file cannot be opened with cuFileHandleRegister(),
 * the function prints the reason and returns -1.0 (displayed as "N/A").
 * The user decides based on the printed diagnostic whether cuFile is usable.
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

/* ── cuFile availability check ───────────────────────────────────── */
/*
 * cufile.h is part of the CUDA toolkit (libcufile).  If the header is
 * missing or the driver is not loaded, we fall back to a stub that always
 * returns -1.0 and prints the reason.
 */
#if __has_include(<cufile.h>)
#  include <cufile.h>
#  define CUFILE_AVAILABLE 1
#else
#  define CUFILE_AVAILABLE 0
#endif

/* ── Module state ────────────────────────────────────────────────── */

static uint8_t *g_devbuf    = NULL;   /* registered GPU device buffer       */
static int      g_cf_open   = 0;      /* cuFileDriverOpen() succeeded        */

/* ── Init / cleanup ──────────────────────────────────────────────── */

int cufile_init(void)
{
#if !CUFILE_AVAILABLE
    fprintf(stderr, "[cuFile] cufile.h not found in CUDA toolkit include path.\n"
                    "         Install nvidia-fs / GDS package and re-build.\n");
    return -1;
#else
    CUfileError_t err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[cuFile] cuFileDriverOpen() failed: err=%d\n"
                        "         Possible reasons: nvidia-fs kernel module not loaded,\n"
                        "         or GDS not supported on this kernel/driver version.\n",
                (int)err.err);
        return -1;
    }

    /* Print GDS mode so user knows which transfer path is active */
    CUfileDrvProps_t props;
    memset(&props, 0, sizeof props);
    CUfileError_t perr = cuFileDriverGetProperties(&props);
    if (perr.err == CU_FILE_SUCCESS) {
        fprintf(stdout, "[cuFile] driver open OK  max_device_cache=%zu KB\n",
                (size_t)props.max_device_cache_size / 1024);
    }

    if (cudaMalloc((void **)&g_devbuf, MB_DATA_PER_THREAD) != cudaSuccess) {
        fprintf(stderr, "[cuFile] cudaMalloc for device buffer failed\n");
        cuFileDriverClose();
        return -1;
    }
    /* Fill with constant pattern */
    cudaMemset(g_devbuf, 0xCD, MB_DATA_PER_THREAD);

    /* Register buffer with GDS */
    CUfileError_t berr = cuFileBufRegister(g_devbuf, MB_DATA_PER_THREAD, 0);
    if (berr.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[cuFile] cuFileBufRegister failed: err=%d\n"
                        "         GDS P2P DMA to fsdax may not be supported.\n",
                (int)berr.err);
        cudaFree(g_devbuf); g_devbuf = NULL;
        cuFileDriverClose();
        return -1;
    }

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
    if (g_cf_open) {
        cuFileDriverClose();
        g_cf_open = 0;
    }
#endif
}

/* ── LCG rand offsets (same seed as basic / beaver) ─────────────── */

static void gen_rand_offsets_cf(uint32_t nthreads, off_t *offsets)
{
    uint32_t rng = 42u;
    off_t    file_blocks = (off_t)nthreads;
    for (uint32_t i = 0; i < nthreads; i++) {
        rng = rng * 1664525u + 1013904223u;
        offsets[i] = (off_t)(rng % file_blocks) * MB_DATA_PER_THREAD;
    }
}

/* ── Single-file write helper ────────────────────────────────────── */

#if CUFILE_AVAILABLE
static double cufile_write_file(const char *path, off_t file_size,
                                 const off_t *offsets, uint32_t nthreads)
{
    int fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) { perror("[cuFile] open"); return -1.0; }
    if (ftruncate(fd, file_size) < 0) { perror("[cuFile] ftruncate"); close(fd); return -1.0; }

    CUfileDescr_t  descr  = {};
    CUfileHandle_t handle = {};
    descr.handle.fd = fd;
    descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileError_t err = cuFileHandleRegister(&handle, &descr);
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "[cuFile] cuFileHandleRegister failed: err=%d  path=%s\n"
                        "         fsdax (ext4-dax) may not support GDS O_DIRECT path.\n",
                (int)err.err, path);
        close(fd);
        return -1.0;
    }

    mb_timer_t t;
    mb_timer_start(&t);

    int failed = 0;
    for (uint32_t i = 0; i < nthreads; i++) {
        ssize_t r = cuFileWrite(handle, g_devbuf, MB_DATA_PER_THREAD,
                                offsets ? offsets[i] : (off_t)i * MB_DATA_PER_THREAD,
                                0 /* buf_off */);
        if (r < 0) {
            fprintf(stderr, "[cuFile] cuFileWrite failed at i=%u: %zd\n", i, r);
            failed = 1;
            break;
        }
    }

    double ms = failed ? -1.0 : mb_timer_elapsed_ms(&t);
    cuFileHandleDeregister(handle);
    close(fd);
    unlink(path);
    return ms;
}
#endif

/* ── MB_SEQ ──────────────────────────────────────────────────────── */

double cufile_run_seq(uint32_t nthreads)
{
#if !CUFILE_AVAILABLE
    return -1.0;
#else
    if (!g_cf_open) return -1.0;
    off_t total = (off_t)nthreads * MB_DATA_PER_THREAD;
    double ms = cufile_write_file(MB_SEQ_FILE, total, NULL, nthreads);
    if (ms < 0.0) return -1.0;
    return mb_throughput(nthreads, ms);
#endif
}

/* ── MB_RAND ─────────────────────────────────────────────────────── */

double cufile_run_rand(uint32_t nthreads)
{
#if !CUFILE_AVAILABLE
    return -1.0;
#else
    if (!g_cf_open) return -1.0;
    off_t total = (off_t)nthreads * MB_DATA_PER_THREAD;
    off_t *offsets = (off_t *)malloc(nthreads * sizeof(off_t));
    if (!offsets) return -1.0;
    gen_rand_offsets_cf(nthreads, offsets);

    /* Use same file name; rand offsets are within same total size */
    double ms = cufile_write_file(MB_RAND_FILE, total, offsets, nthreads);
    free(offsets);
    if (ms < 0.0) return -1.0;
    return mb_throughput(nthreads, ms);
#endif
}

/* ── MB_MULTI ────────────────────────────────────────────────────── */
/*
 * N files: open + cuFileHandleRegister + cuFileWrite + cuFileHandleDeregister + close per file.
 * Timing covers all of this (CPU metadata overhead per file is visible).
 */
double cufile_run_multi(uint32_t nthreads)
{
#if !CUFILE_AVAILABLE
    return -1.0;
#else
    if (!g_cf_open) return -1.0;

    char path[256];
    mb_timer_t t;
    mb_timer_start(&t);

    for (uint32_t i = 0; i < nthreads; i++) {
        snprintf(path, sizeof path, MB_FILE_FMT, i);

        int fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (fd < 0) { perror("[cuFile/Multi] open"); break; }
        if (ftruncate(fd, MB_DATA_PER_THREAD) < 0) {
            perror("[cuFile/Multi] ftruncate"); close(fd); break;
        }

        CUfileDescr_t  descr  = {};
        CUfileHandle_t handle = {};
        descr.handle.fd = fd;
        descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        CUfileError_t err = cuFileHandleRegister(&handle, &descr);
        if (err.err != CU_FILE_SUCCESS) {
            fprintf(stderr, "[cuFile/Multi] cuFileHandleRegister failed at i=%u: %d\n",
                    i, (int)err.err);
            close(fd); break;
        }

        cuFileWrite(handle, g_devbuf, MB_DATA_PER_THREAD, 0, 0);
        cuFileHandleDeregister(handle);
        close(fd);
    }

    double ms = mb_timer_elapsed_ms(&t);

    /* Cleanup (not timed) */
    for (uint32_t i = 0; i < nthreads; i++) {
        snprintf(path, sizeof path, MB_FILE_FMT, i);
        unlink(path);
    }
    return mb_throughput(nthreads, ms);
#endif
}

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
