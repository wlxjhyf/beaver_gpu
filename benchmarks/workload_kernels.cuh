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
        vfs_write_data(vfs, fd, p, wbuf);   /* stage only, no fence */

    vfs_fsync(vfs, fd);                      /* one fence + inode persist */
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
        vfs_write_data(vfs, fd, order[p], wbuf);   /* stage only, no fence */

    vfs_fsync(vfs, fd);
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
        vfs_write_data(vfs, fd, p, wbuf);   /* stage only, no fence */

    vfs_fsync(vfs, fd);
    vfs_close(vfs, fd);
}

/* ================================================================== */
/* Warp-cooperative kernels: 1 block (32 threads) = 1 logical file   */
/*                                                                     */
/* All 32 threads call vfs_write_data_warp together for each page.   */
/* Adjacent threads write adjacent 8B words → coalesced PCIe writes. */
/* Only lane 0 handles metadata (open/close/fsync/create).            */
/*                                                                     */
/* Launch: <<<n_files, 32>>>  (one warp per file)                     */
/* ================================================================== */

/* ── Warp file creation (setup, not timed) ──────────────────────── */

__global__ void mb_create_kernel_warp(gpu_vfs_t      *vfs,
                                       const uint32_t *hashes,
                                       uint32_t        n)
{
    uint32_t file_id = blockIdx.x;
    if (file_id >= n) return;
    if (threadIdx.x == 0)
        vfs_create(vfs, hashes[file_id]);
}

/* ── MB_SEQ warp: sequential page writes, coalesced PM stores ───── */

__global__ void mb_seq_write_kernel_warp(gpu_vfs_t      *vfs,
                                          const uint32_t *hashes,
                                          const uint8_t  *wbuf,
                                          uint32_t        n)
{
    uint32_t file_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;   /* 0..31 */
    if (file_id >= n) return;

    /* Lane 0 opens the file; broadcast fd to all lanes */
    int fd = -1;
    if (lane == 0) fd = vfs_open(vfs, hashes[file_id]);
    fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
    if (fd < 0) return;

    /* All lanes cooperate on each page write */
    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write_data_warp(vfs, fd, p, wbuf);

    /* Lane 0 issues the single fence + inode persist */
    if (lane == 0) {
        vfs_fsync(vfs, fd);
        vfs_close(vfs, fd);
    }
}

/* ── MB_RAND warp: random-order page writes, coalesced PM stores ── */

__global__ void mb_rand_write_kernel_warp(gpu_vfs_t      *vfs,
                                           const uint32_t *hashes,
                                           const uint8_t  *wbuf,
                                           uint32_t        n)
{
    uint32_t file_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;

    /* Build shuffle order in shared memory (visible to all lanes) */
    __shared__ uint32_t order[MB_WRITES_PER_THREAD];

    if (file_id < n && lane == 0) {
        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;
        uint32_t rng = file_id * 2654435761u + 1u;
        for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
            rng = rng * 1664525u + 1013904223u;
            uint32_t j   = rng % (p + 1);
            uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
        }
    }
    __syncwarp(0xFFFFFFFFu);   /* ensure order[] visible before all lanes read */

    if (file_id >= n) return;

    int fd = -1;
    if (lane == 0) fd = vfs_open(vfs, hashes[file_id]);
    fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
    if (fd < 0) return;

    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write_data_warp(vfs, fd, order[p], wbuf);

    if (lane == 0) {
        vfs_fsync(vfs, fd);
        vfs_close(vfs, fd);
    }
}

/* ── MB_MULTI warp: create + sequential write, all timed ────────── */

__global__ void mb_multi_write_kernel_warp(gpu_vfs_t      *vfs,
                                            const uint32_t *hashes,
                                            const uint8_t  *wbuf,
                                            uint32_t        n)
{
    uint32_t file_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    if (file_id >= n) return;

    if (lane == 0) vfs_create(vfs, hashes[file_id]);

    int fd = -1;
    if (lane == 0) fd = vfs_open(vfs, hashes[file_id]);
    fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
    if (fd < 0) return;

    for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
        vfs_write_data_warp(vfs, fd, p, wbuf);

    if (lane == 0) {
        vfs_fsync(vfs, fd);
        vfs_close(vfs, fd);
    }
}

/* ================================================================== */
/* Raw PM write kernel: data only, no metadata, no holders, no log    */
/*                                                                     */
/* Used to isolate pure PM data-write throughput from Beaver overhead.*/
/* PM layout: flat contiguous slab, file fi occupies:                 */
/*   pm_base + fi * MB_WRITES_PER_THREAD * MB_PAGE_SIZE               */
/* One __threadfence_system() per file, same as Beaver's vfs_fsync.   */
/* ================================================================== */

__global__ void mb_raw_write_kernel_loop(void          *pm_base,
                                          const uint8_t *wbuf,
                                          uint32_t       n_files)
{
    uint32_t warp_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    uint32_t n_warps = gridDim.x;

    uint32_t per      = (n_files + n_warps - 1) / n_warps;
    uint32_t fi_start = warp_id * per;
    uint32_t fi_end   = (fi_start + per < n_files) ? fi_start + per : n_files;

    const uint32_t n_words = MB_PAGE_SIZE / sizeof(unsigned long long);
    const unsigned long long *src = (const unsigned long long *)wbuf;

    for (uint32_t fi = fi_start; fi < fi_end; fi++) {
        for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) {
            size_t offset = ((size_t)fi * MB_WRITES_PER_THREAD + p) * MB_PAGE_SIZE;
            volatile unsigned long long *dst =
                (volatile unsigned long long *)((uint8_t *)pm_base + offset);
            /* stride-32 coalesced write: same pattern as gpu_f2fs_write_data_warp */
            for (uint32_t i = lane; i < n_words; i += 32u)
                dst[i] = src[i];
        }
        /* one fence per file — matches Beaver vfs_fsync drain */
        __threadfence_system();
    }
}

/* ================================================================== */
/* Loop kernels: each warp handles ceil(n / gridDim.x) files serially */
/*                                                                     */
/* Launch: <<<MB_BEAVER_WARPS, 32>>>                                  */
/*                                                                     */
/* Fewer concurrent warps → fewer simultaneous PM writers →           */
/* less WPQ fragmentation → higher PM write-combining efficiency.     */
/* rawgpm (b): 1 warp ≈ 3165 MB/s  vs  128 warps ≈ 2000 MB/s.       */
/* ================================================================== */

/* ── MB_SEQ loop: sequential page writes, one warp per slice ─────── */

__global__ void mb_seq_write_kernel_loop(gpu_vfs_t      *vfs,
                                          const uint32_t *hashes,
                                          const uint8_t  *wbuf,
                                          uint32_t        n)
{
    uint32_t warp_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    uint32_t n_warps = gridDim.x;

    uint32_t per      = (n + n_warps - 1) / n_warps;
    uint32_t fi_start = warp_id * per;
    uint32_t fi_end   = (fi_start + per < n) ? fi_start + per : n;

    for (uint32_t fi = fi_start; fi < fi_end; fi++) {
        int fd = -1;
        if (lane == 0) fd = vfs_open(vfs, hashes[fi]);
        fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
        if (fd >= 0) {
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
                vfs_write_data_warp(vfs, fd, p, wbuf);
            if (lane == 0) { vfs_fsync(vfs, fd); vfs_close(vfs, fd); }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}

/* ── MB_RAND loop: random-order page writes, one warp per slice ──── */

__global__ void mb_rand_write_kernel_loop(gpu_vfs_t      *vfs,
                                           const uint32_t *hashes,
                                           const uint8_t  *wbuf,
                                           uint32_t        n)
{
    uint32_t warp_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    uint32_t n_warps = gridDim.x;

    __shared__ uint32_t order[MB_WRITES_PER_THREAD];

    uint32_t per      = (n + n_warps - 1) / n_warps;
    uint32_t fi_start = warp_id * per;
    uint32_t fi_end   = (fi_start + per < n) ? fi_start + per : n;

    for (uint32_t fi = fi_start; fi < fi_end; fi++) {
        /* Lane 0 builds shuffle order for this file */
        if (lane == 0) {
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;
            uint32_t rng = fi * 2654435761u + 1u;
            for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
                rng = rng * 1664525u + 1013904223u;
                uint32_t j   = rng % (p + 1);
                uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
            }
        }
        __syncwarp(0xFFFFFFFFu);

        int fd = -1;
        if (lane == 0) fd = vfs_open(vfs, hashes[fi]);
        fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
        if (fd >= 0) {
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
                vfs_write_data_warp(vfs, fd, order[p], wbuf);
            if (lane == 0) { vfs_fsync(vfs, fd); vfs_close(vfs, fd); }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}

/* ── MB_MULTI loop: create + sequential write, one warp per slice ── */

__global__ void mb_multi_write_kernel_loop(gpu_vfs_t      *vfs,
                                            const uint32_t *hashes,
                                            const uint8_t  *wbuf,
                                            uint32_t        n)
{
    uint32_t warp_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    uint32_t n_warps = gridDim.x;

    uint32_t per      = (n + n_warps - 1) / n_warps;
    uint32_t fi_start = warp_id * per;
    uint32_t fi_end   = (fi_start + per < n) ? fi_start + per : n;

    for (uint32_t fi = fi_start; fi < fi_end; fi++) {
        if (lane == 0) vfs_create(vfs, hashes[fi]);
        __syncwarp(0xFFFFFFFFu);

        int fd = -1;
        if (lane == 0) fd = vfs_open(vfs, hashes[fi]);
        fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
        if (fd >= 0) {
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
                vfs_write_data_warp(vfs, fd, p, wbuf);
            if (lane == 0) { vfs_fsync(vfs, fd); vfs_close(vfs, fd); }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}

/* ── MB_RAND_MULTI loop: create + random-order write, one warp per slice ──
 *
 * Used by MB_RAND when creation is included in timing (new fair design).
 * Same as mb_multi_write_kernel_loop but writes pages in random order
 * (Fisher-Yates shuffle, same seed as mb_rand_write_kernel_loop).
 */

__global__ void mb_rand_multi_write_kernel_loop(gpu_vfs_t      *vfs,
                                                 const uint32_t *hashes,
                                                 const uint8_t  *wbuf,
                                                 uint32_t        n)
{
    uint32_t warp_id = blockIdx.x;
    uint32_t lane    = threadIdx.x;
    uint32_t n_warps = gridDim.x;

    __shared__ uint32_t order[MB_WRITES_PER_THREAD];

    uint32_t per      = (n + n_warps - 1) / n_warps;
    uint32_t fi_start = warp_id * per;
    uint32_t fi_end   = (fi_start + per < n) ? fi_start + per : n;

    for (uint32_t fi = fi_start; fi < fi_end; fi++) {
        /* Lane 0: create file and build shuffle order */
        if (lane == 0) {
            vfs_create(vfs, hashes[fi]);
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++) order[p] = p;
            uint32_t rng = fi * 2654435761u + 1u;
            for (uint32_t p = MB_WRITES_PER_THREAD - 1; p > 0; p--) {
                rng = rng * 1664525u + 1013904223u;
                uint32_t j   = rng % (p + 1);
                uint32_t tmp = order[p]; order[p] = order[j]; order[j] = tmp;
            }
        }
        __syncwarp(0xFFFFFFFFu);

        int fd = -1;
        if (lane == 0) fd = vfs_open(vfs, hashes[fi]);
        fd = __shfl_sync(0xFFFFFFFFu, fd, 0);
        if (fd >= 0) {
            for (uint32_t p = 0; p < MB_WRITES_PER_THREAD; p++)
                vfs_write_data_warp(vfs, fd, order[p], wbuf);
            if (lane == 0) { vfs_fsync(vfs, fd); vfs_close(vfs, fd); }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}
