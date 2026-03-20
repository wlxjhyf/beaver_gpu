/*
 * workload_kernels.cuh — GPU write kernels for beaver_gpu-COW microbenchmark.
 *
 * Three kernels matching three workload types:
 *   mb_create_kernel     : create N files (used as setup before MB_SEQ / MB_RAND)
 *   mb_seq_write_kernel  : each thread writes MB_WRITES_PER_THREAD pages
 *                          sequentially (page 0, 1, …, 31) to its own file.
 *   mb_rand_write_kernel : each thread writes to MB_WRITES_PER_THREAD pseudo-random
 *                          page offsets (LCG) within its own file.
 *   mb_multi_write_kernel: each thread creates + writes sequentially (MB_MULTI).
 *
 * All kernels: one GPU thread = one logical client = one file.
 * No inter-thread file sharing → no inode spinlock contention.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include "gpu_f2fs.h"
#include "bench_config.h"

/* ── File creation ───────────────────────────────────────────────── */

__global__ void mb_create_kernel(gpu_vfs_t      *vfs,
                                  const uint32_t *hashes,
                                  uint32_t        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vfs_create(vfs, hashes[i]);
}

/* ── MB_SEQ: sequential page writes (pages 0 … 31) ─────────────── */

__global__ void mb_seq_write_kernel(gpu_vfs_t      *vfs,
                                     const uint32_t *hashes,
                                     const uint8_t  *wbuf,
                                     uint32_t        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int fd = vfs_open(vfs, hashes[i]);
    if (fd < 0) return;

    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write(vfs, fd, p, wbuf);

    vfs_close(vfs, fd);
}

/* ── MB_RAND: random-order page writes within own file ──────────── */
/*
 * LCG (Numerical Recipes): a=1664525, c=1013904223.
 * Each thread has a unique seed (tid * prime), so page visit orders differ
 * across threads while staying within [0, MB_WRITES_PER_THREAD).
 * Write to every page exactly once (shuffle via Fisher-Yates on a small
 * on-stack array — avoids repeated writes to the same pgoff).
 */
__global__ void mb_rand_write_kernel(gpu_vfs_t      *vfs,
                                      const uint32_t *hashes,
                                      const uint8_t  *wbuf,
                                      uint32_t        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    /* Build a per-thread page index shuffle on the stack */
    uint32_t order[MB_WRITES_PER_THREAD];
    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;

    uint32_t rng = i * 2654435761u + 1u;   /* unique per-thread seed */
    for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
        rng = rng * 1664525u + 1013904223u;
        uint32_t j  = rng % (p + 1);
        uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
    }

    int fd = vfs_open(vfs, hashes[i]);
    if (fd < 0) return;

    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write(vfs, fd, order[p], wbuf);

    vfs_close(vfs, fd);
}

/* ── MB_MULTI: create + sequential write in one kernel ──────────── */

__global__ void mb_multi_write_kernel(gpu_vfs_t      *vfs,
                                       const uint32_t *hashes,
                                       const uint8_t  *wbuf,
                                       uint32_t        n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    vfs_create(vfs, hashes[i]);

    int fd = vfs_open(vfs, hashes[i]);
    if (fd < 0) return;

    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write(vfs, fd, p, wbuf);

    vfs_close(vfs, fd);
}
