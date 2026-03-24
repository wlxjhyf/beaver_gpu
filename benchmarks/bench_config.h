/*
 * bench_config.h — Shared configuration for microbenchmarks.
 *
 * Benchmark design: FIXED total data, varying CPU worker count.
 *   - MB_TOTAL_FILES × MB_DATA_PER_THREAD bytes are persisted each run.
 *   - CPU-path methods (pwrite+fsync, pmem_persist, cuFile) are swept over
 *     MB_THREAD_COUNTS CPU workers; all workers write the same MB_TOTAL_FILES.
 *   - Beaver always uses MB_TOTAL_FILES GPU warps ("GPU at full effort").
 *   - Throughput = MB_TOTAL_FILES × MB_DATA_PER_THREAD / elapsed_time.
 *
 * Storage:
 *   Basic + cuFile : /mnt/pmem  via ext4-dax  (block interface)
 *   beaver         : /dev/dax1.0 via GPM       (byte interface)
 */
#pragma once
#include <stdint.h>

/* ── Per-file workload ───────────────────────────────────────────── */
#define MB_PAGE_SIZE           4096u
#define MB_WRITES_PER_THREAD   32u
#define MB_DATA_PER_THREAD     (MB_PAGE_SIZE * MB_WRITES_PER_THREAD)  /* 128 KiB */

/* ── Fixed total files per run ───────────────────────────────────── */
/* 4096 files × 128 KiB = 512 MiB total data per run.
 * CPU methods divide these files among their worker pool.
 * Beaver uses MB_BEAVER_WARPS warps; each warp handles
 * ceil(MB_TOTAL_FILES / MB_BEAVER_WARPS) files serially.
 *
 * rawgpm data (b): 1 warp → 3165 MB/s, 128 warps → ~2000 MB/s.
 * Fewer warps → better PM WPQ write-combining, less PCIe fragmentation. */
#define MB_TOTAL_FILES   4096u

/* Number of GPU warps Beaver uses for the timed write phase.
 * 1  = single-warp serial (best raw PM bandwidth per rawgpm).
 * MB_TOTAL_FILES = one warp per file (max GPU parallelism, but WPQ contention).
 * Override at compile time: -DMB_BEAVER_WARPS=4 */
#ifndef MB_BEAVER_WARPS
#  define MB_BEAVER_WARPS  1u
#endif

/* ── CPU worker count sweep ──────────────────────────────────────── */
/* NUMA-0 (Socket 2, where /mnt/pmem lives) has 24 CPUs.
 * Sweep covers 1 → full NUMA node parallelism. */
#define MB_NT_COUNT  6u
static const uint32_t MB_THREAD_COUNTS[MB_NT_COUNT] = {
    1, 4, 8, 12, 16, 24
};

/* ── Storage paths ───────────────────────────────────────────────── */
#define MB_PMEM_MOUNT   "/mnt/pmem"
#define MB_FILE_FMT     MB_PMEM_MOUNT "/mb_%06u.dat"
#define MB_SEQ_FILE     MB_PMEM_MOUNT "/mb_seq.dat"
#define MB_RAND_FILE    MB_PMEM_MOUNT "/mb_rand.dat"

/* ── Workload types ──────────────────────────────────────────────── */
typedef enum {
    MB_SEQ   = 0,   /* seq write, pre-created files, time write only     */
    MB_RAND  = 1,   /* rand write, pre-created files, time write only     */
    MB_MULTI = 2,   /* multi-file: create + write all timed together      */
} mb_workload_t;

static const char *const MB_WORKLOAD_NAMES[] = { "SeqWrite", "RandWrite", "MultiWrite" };

/* ── Result row ──────────────────────────────────────────────────── */
typedef struct {
    uint32_t nworkers;
    double   mbps;    /* aggregate write throughput (MB/s); -1 = N/A */
} mb_row_t;
