/*
 * bench_utils.h — Timing helpers and print utilities for microbenchmarks.
 */
#pragma once
#include "bench_config.h"
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>

/* ── Host-side wall-clock timer ─────────────────────────────────── */

typedef struct { struct timespec t0; } mb_timer_t;

static inline void mb_timer_start(mb_timer_t *t)
{
    clock_gettime(CLOCK_MONOTONIC, &t->t0);
}

static inline double mb_timer_elapsed_ms(const mb_timer_t *t)
{
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec  - t->t0.tv_sec)  * 1000.0
         + (t1.tv_nsec - t->t0.tv_nsec) / 1.0e6;
}

/* For beaver: sync GPU then sample elapsed time. */
static inline double mb_cuda_sync_elapsed_ms(const mb_timer_t *t)
{
    cudaDeviceSynchronize();
    return mb_timer_elapsed_ms(t);
}

/* ── Throughput calculation ──────────────────────────────────────── */

/* Fixed-total throughput: always MB_TOTAL_FILES × MB_DATA_PER_THREAD bytes. */
static inline double mb_throughput(double elapsed_ms)
{
    if (elapsed_ms <= 0.0) return -1.0;
    double bytes = (double)MB_TOTAL_FILES * MB_DATA_PER_THREAD;
    return bytes / (elapsed_ms / 1000.0) / (1024.0 * 1024.0);  /* MB/s */
}

/* ── GPU launch helpers ──────────────────────────────────────────── */

static inline void mb_grid_block(uint32_t n, uint32_t *grid, uint32_t *blk)
{
    *blk  = (n < 128u) ? n : 128u;
    *grid = (n + *blk - 1) / *blk;
}

/*
 * mb_grid_block_warp: for warp-per-file kernels.
 * Each block = exactly 1 warp (32 threads) = 1 logical file.
 * Launch: <<<n_files, 32>>>.
 */
static inline void mb_grid_block_warp(uint32_t n_files, uint32_t *grid, uint32_t *blk)
{
    *blk  = 32u;
    *grid = n_files;
}

/* ── Error check macros ──────────────────────────────────────────── */

#define MB_CUDA_CHECK(call)                                              \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));         \
            return -1.0;                                                 \
        }                                                                \
    } while (0)

/* ── Result printing ─────────────────────────────────────────────── */

static inline void mb_print_table_header(void)
{
    printf("  %-12s  %14s  %12s  %12s\n",
           "CPU Workers", "pwrite+rename", "cuFile", "Beaver(GPU)");
    printf("  %-12s  %14s  %12s  %12s\n",
           "------------", "--------------", "------------", "------------");
}

static inline void mb_print_row(uint32_t nw,
                                 double basic,
                                 double cufile, double beaver)
{
    char sb[16], sc[16], sv[16];
#define FMT(buf, w, v) \
    do { if ((v) < 0.0) snprintf(buf,sizeof buf,"%*s",w,"N/A"); \
         else snprintf(buf,sizeof buf,"%*.1f",w,v); } while(0)
    FMT(sb, 14, basic);
    FMT(sc, 12, cufile);
    FMT(sv, 12, beaver);
#undef FMT
    printf("  %-12u  %s  %s  %s\n", nw, sb, sc, sv);
}
