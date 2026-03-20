/*
 * gpu_f2fs_ckpt.h — Pure GPU F2FS baseline (no Beaver).
 *
 * This is a clean, standalone GPU implementation of F2FS semantics on PM.
 * It does NOT include beaver_cow.h and calls no Beaver functions.
 * Only gpm_interface.cuh is used for PM access primitives.
 *
 * Mirrors CPU-side F2FS logic:
 *   - Data: out-of-place writes to PM log area (atomicAdd seg_cursor).
 *     Each write allocates a new PM block — no COW, no double-buffer.
 *   - Inode: maintained in GPU DRAM (inode_array + lock array).
 *     dirty_flags[nid]=1 on each write.
 *   - Checkpoint: stop-the-world kernel flushes all dirty inodes to
 *     fixed PM slots (pm_ckpt_base + nid * PAGE_SIZE), then clears flags.
 *
 * This implementation is used as the F2FS checkpoint baseline in the
 * BM3 stop-the-world stall benchmark.
 *
 * inode.i_addr[pgoff] stores the physical PM data block address
 * (as a uint32_t block index into the PM data slab).
 */

#ifndef GPU_F2FS_CKPT_H
#define GPU_F2FS_CKPT_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "f2fs_types.h"
#include "gpm_interface.cuh"

/* ------------------------------------------------------------------ */
/* gpu_ckpt_fs_t — control block (cudaMallocManaged)                 */
/* ------------------------------------------------------------------ */
typedef struct {
    /* Inode DRAM state (no Beaver shadow — inodes stored flat) */
    f2fs_inode_t *inodes;          /* cudaMalloc, one F2FS inode per slot */
    uint32_t     *inode_locks;     /* cudaMalloc, per-inode GPU spinlock  */
    uint32_t     *dirty_flags;     /* cudaMalloc, 1=dirty                 */
    uint32_t      max_inodes;
    uint32_t      inode_cursor;    /* atomicAdd — allocate nid            */

    /* Name lookup */
    uint32_t     *name_table;      /* cudaMalloc, hash→nid                */
    uint32_t      name_table_size;

    /* Data: out-of-place log writes to PM */
    void         *pm_data_base;    /* UVA pointer to PM data slab         */
    gpm_region_t  pm_data_region;
    uint32_t      seg_cursor;      /* atomicAdd — next data block index   */
    uint32_t      max_data_blocks;

    /* Checkpoint pack: flat PM inode area (mirrors F2FS checkpoint pack) */
    void         *pm_ckpt_base;    /* UVA pointer: ckpt_base[nid] = inode */
    gpm_region_t  pm_ckpt_region;

    uint32_t      is_initialized;
} gpu_ckpt_fs_t;

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    GPU_CKPT_OK          =  0,
    GPU_CKPT_ERR_INIT    = -1,
    GPU_CKPT_ERR_NOMEM   = -2,
    GPU_CKPT_ERR_NOFILE  = -3,
    GPU_CKPT_ERR_NOSPACE = -4,
} gpu_ckpt_err_t;

/* ------------------------------------------------------------------ */
/* __device__ POSIX API                                               */
/* ------------------------------------------------------------------ */

/*
 * gpu_ckpt_create: allocate inode, insert into name_table.
 * Returns nid on success, -1 on full pool or full name_table.
 */
__device__ int  gpu_ckpt_create(gpu_ckpt_fs_t *fs, uint32_t name_hash);

/*
 * gpu_ckpt_open: lock-free name lookup.
 * Returns nid on success, -1 if not found.
 */
__device__ int  gpu_ckpt_open  (gpu_ckpt_fs_t *fs, uint32_t name_hash);

/*
 * gpu_ckpt_write: out-of-place write one 4 KiB page.
 * Allocates a new PM data block each call (no COW).
 * Marks inode dirty. Returns 0/-1/-2.
 */
__device__ int  gpu_ckpt_write (gpu_ckpt_fs_t *fs, int fd, uint32_t pgoff,
                                 const void *src);

/*
 * gpu_ckpt_read: read one 4 KiB page.
 * Lock-free volatile read from PM. Returns 0/-1.
 */
__device__ int  gpu_ckpt_read  (gpu_ckpt_fs_t *fs, int fd, uint32_t pgoff,
                                 void *dst);

/*
 * gpu_ckpt_close: no-op (stateless fd).
 */
__device__ void gpu_ckpt_close (gpu_ckpt_fs_t *fs, int fd);

/* ------------------------------------------------------------------ */
/* Host API                                                            */
/* ------------------------------------------------------------------ */

gpu_ckpt_err_t gpu_ckpt_fs_init   (gpu_ckpt_fs_t *fs,
                                    uint32_t max_inodes,
                                    uint32_t max_data_blocks);

gpu_ckpt_err_t gpu_ckpt_fs_cleanup(gpu_ckpt_fs_t *fs);

/*
 * gpu_ckpt_do_checkpoint: stop-the-world inode flush.
 * Flushes all dirty inodes to pm_ckpt_base, then clears dirty_flags.
 * Blocks until complete (cudaDeviceSynchronize).
 */
void gpu_ckpt_do_checkpoint(gpu_ckpt_fs_t *fs);

#endif /* GPU_F2FS_CKPT_H */
