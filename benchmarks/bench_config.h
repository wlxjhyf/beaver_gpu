/*
 * bench_config.h — Shared configuration for microbenchmarks.
 *
 * Three workloads (Seq / Rand / Multi-file write), three systems
 * (Basic / cuFile / beaver_gpu-COW), all on Intel Optane PM:
 *   Basic + cuFile : /dev/pmem0  via ext4-dax  (block interface)
 *   beaver         : /dev/dax1.0 via GPM        (byte interface)
 */
#pragma once
#include <stdint.h>

/* ── Workload parameters ─────────────────────────────────────────── */
#define MB_PAGE_SIZE           4096u
#define MB_WRITES_PER_THREAD   32u
#define MB_DATA_PER_THREAD     (MB_PAGE_SIZE * MB_WRITES_PER_THREAD)  /* 128 KiB */

/* Thread counts covering GPU scaling range */
#define MB_NT_COUNT  7u
static const uint32_t MB_THREAD_COUNTS[MB_NT_COUNT] = {
    1, 64, 256, 1024, 4096, 16384, 65536
};

/* ── Storage paths ───────────────────────────────────────────────── */
#define MB_PMEM_MOUNT   "/mnt/pmem"           /* ext4-dax on /dev/pmem0  */
#define MB_FILE_FMT     MB_PMEM_MOUNT "/mb_%06u.dat"  /* per-thread files */
#define MB_SEQ_FILE     MB_PMEM_MOUNT "/mb_seq.dat"   /* shared single file */
#define MB_RAND_FILE    MB_PMEM_MOUNT "/mb_rand.dat"

/* ── Workload types ──────────────────────────────────────────────── */
typedef enum {
    MB_SEQ   = 0,   /* seq write, pre-created files, time write only     */
    MB_RAND  = 1,   /* rand write, pre-created files, time write only     */
    MB_MULTI = 2,   /* multi-file: create + write all timed together      */
} mb_workload_t;

static const char *const MB_WORKLOAD_NAMES[] = { "SeqWrite", "RandWrite", "MultiWrite" };

/* ── Result: one row per (system, thread_count) ─────────────────── */
typedef struct {
    uint32_t nthreads;
    double   mbps;    /* aggregate write throughput (MB/s); -1 = N/A */
} mb_row_t;
