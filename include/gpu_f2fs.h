/*
 * gpu_f2fs.h — GPU F2FS: structures, declarations, and VFS dispatch.
 *
 * This header contains:
 *   1. Struct definitions (f2fs_inode_shadow_t, gpu_f2fs_t, …)
 *   2. Error codes
 *   3. Inline VFS dispatch (vfs_* → gpu_f2fs_*)
 *   4. __device__ function declarations (implementations: gpu_f2fs.cu / beaver_f2fs.cu)
 *   5. Host API declarations (implementations: gpu_f2fs.cu)
 *
 * Architecture:
 *
 *   GPU Kernel
 *     │  vfs_open / vfs_write / vfs_read / vfs_close   (__forceinline__)
 *     ▼
 *   gpu_f2fs_write / gpu_f2fs_read  (gpu_f2fs.cu — pure F2FS logic)
 *     │
 *     ├─ DATA:  beaver_data_write / beaver_data_read   (beaver_f2fs.cu)
 *     │          → data_cache holders + PM log + holder_flip
 *     │
 *     └─ INODE: f2fs_inode_update (gpu_f2fs.cu)
 *                ├─ COW mode:  beaver_inode_persist     (beaver_f2fs.cu)
 *                │              → cow_cache holders + PM log + holder_flip
 *                └─ Ckpt mode: dirty_flags[nid] = 1   (batch flush later)
 *
 * Two persistence modes (selected at init via use_cow):
 *   COW mode  (use_cow=1): data + inode both go through Beaver holders → PM.
 *   Ckpt mode (use_cow=0): data still goes through Beaver holders → PM;
 *       inode update marks dirty_flags[nid]=1; gpu_f2fs_do_checkpoint()
 *       flushes dirty inodes stop-the-world to pm_inode_base.
 *
 * inode.i_addr[] semantics (Beaver F2FS):
 *   F2FS_NULL_ADDR (0xFFFFFFFF) → page not yet written.
 *   value < data_cache->max_holders → holder index in data_cache.
 *   Direct array access: &data_cache->holders[i_addr[pgoff]].
 *
 * VFS dispatch is __forceinline__ to avoid cross-TU device function
 * pointer issues on Ampere (cudaErrorIllegalInstruction).
 */

//我质疑F2FS的写策略中，对于数据块的写，是否能用Beaver的写函数？因为在F2FS里面没有强制要求马上就写入PM，只有在CP时才会强制刷盘
//现在的实现方法应该是只有inode这种元数据的写被改了

#ifndef GPU_F2FS_H
#define GPU_F2FS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "f2fs_types.h"
#include "beaver_cow.h"
#include "gpu_vfs.h"

/* ------------------------------------------------------------------ */
/* f2fs_inode_shadow_t — per-inode DRAM working copy (cudaMalloc)    */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t      lock;    /* GPU spinlock — serialises concurrent updates */
    f2fs_inode_t  inode;   /* current in-memory inode state               */
} f2fs_inode_shadow_t;

/* ------------------------------------------------------------------ */
/* gpu_f2fs_t — FS control block (cudaMallocManaged)                 */
/* ------------------------------------------------------------------ */
typedef struct {
    /* Inode DRAM state */
    f2fs_inode_shadow_t *inode_shadow;   /* cudaMalloc, GPU DRAM           */
    uint32_t            *dirty_flags;    /* cudaMalloc, 1=dirty (ckpt mode)*/
    uint32_t             max_inodes;
    uint32_t             inode_cursor;   /* atomicAdd — allocate nid       */

    /* Name lookup */
    uint32_t            *name_table;     /* cudaMalloc, hash→nid           */
    uint32_t             name_table_size;

    /*
     * Beaver COW caches:
     *   cow_cache  — one holder per inode (page_id = nid); COW mode only.
     *   data_cache — one holder per written data page (page_id = encode(nid,pgoff));
     *                used in BOTH COW and Checkpoint modes.
     * inode.i_addr[pgoff] stores the holder_idx in data_cache.
     */
    beaver_cache_t      *cow_cache;      /* inode COW holders              */
    beaver_cache_t      *data_cache;     /* data page COW holders          */

    /*
     * PM write-ahead log — shared by inode and data writes.
     * Embedded directly (not a pointer) so its DRAM counters (head,
     * log_seq) are device-accessible via the managed-memory gpu_f2fs_t.
     */
    beaver_log_t         log;

    /*
     * Checkpoint mode inode PM area — pure F2FS semantics, no Beaver.
     * Layout: pm_inode_base + nid * PAGE_SIZE = inode[nid] on PM.
     * Only allocated when use_cow == 0.  NULL in COW mode.
     */
    void                *pm_inode_base;   /* UVA, checkpoint mode only     */
    gpm_region_t         pm_inode_region;

    /* Mode */
    uint32_t             use_cow;        /* 1=COW, 0=Checkpoint            */
    uint32_t             is_initialized;
} gpu_f2fs_t;

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    GPU_F2FS_OK         =  0,
    GPU_F2FS_ERR_INIT   = -1,
    GPU_F2FS_ERR_NOMEM  = -2,
    GPU_F2FS_ERR_NOFILE = -3,
    GPU_F2FS_ERR_NOSPACE= -4,
    GPU_F2FS_ERR_BADFD  = -5,
} gpu_f2fs_err_t;

/* ------------------------------------------------------------------ */
/* __device__ core declarations (gpu_f2fs.cu)                         */
/* ------------------------------------------------------------------ */

__device__ void f2fs_inode_update(gpu_f2fs_t *fs, uint32_t nid);

/* ------------------------------------------------------------------ */
/* __device__ Beaver glue declarations (beaver_f2fs.cu)               */
/* ------------------------------------------------------------------ */

/*
 * beaver_inode_persist: copy inode_shadow[nid].inode → PM via cow_cache holder.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void beaver_inode_persist(gpu_f2fs_t *fs, uint32_t nid);

/*
 * beaver_inode_persist_stage: stage inode write WITHOUT fence/publish.
 * Caller must hold inode_shadow[nid].lock.
 * Follow up with __threadfence_system() + beaver_holder_publish().
 */
__device__ void beaver_inode_persist_stage(gpu_f2fs_t *fs, uint32_t nid);

/*
 * beaver_data_write: COW write of one 4 KiB page via data_cache holder.
 * existing_holder_idx: current i_addr[pgoff] value (F2FS_NULL_ADDR = new page).
 * Returns new holder_idx (store into i_addr[pgoff]) or F2FS_NULL_ADDR on full.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ uint32_t beaver_data_write(gpu_f2fs_t *fs, uint32_t nid,
                                      uint32_t pgoff,
                                      uint32_t existing_holder_idx,
                                      const void *src);

/*
 * beaver_data_write_stage: COW write of one 4 KiB page WITHOUT fence/publish.
 * Returns new holder_idx or F2FS_NULL_ADDR on full.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ uint32_t beaver_data_write_stage(gpu_f2fs_t *fs, uint32_t nid,
                                             uint32_t pgoff,
                                             uint32_t existing_holder_idx,
                                             const void *src);

/*
 * beaver_data_read: lock-free read of one 4 KiB page via data_cache holder.
 * holder_idx: the value in inode.i_addr[pgoff].
 * Returns 0 on success, -1 if not written.
 */
__device__ int beaver_data_read(gpu_f2fs_t *fs, uint32_t holder_idx, void *dst);

/* ------------------------------------------------------------------ */
/* __device__ POSIX API declarations (gpu_f2fs.cu)                   */
/* ------------------------------------------------------------------ */

__device__ int  gpu_f2fs_create(gpu_f2fs_t *fs, uint32_t name_hash);
__device__ int  gpu_f2fs_open  (gpu_f2fs_t *fs, uint32_t name_hash);
__device__ int  gpu_f2fs_write (gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                 const void *src);
__device__ int  gpu_f2fs_read  (gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                 void *dst);
__device__ void gpu_f2fs_close (gpu_f2fs_t *fs, int fd);

/*
 * gpu_f2fs_write_data: write one 4 KiB data page WITHOUT fence and WITHOUT
 * inode persistence.  Updates DRAM inode fields (i_addr, i_blocks, i_size)
 * but does NOT call f2fs_inode_update.  Call gpu_f2fs_fsync() after all
 * pages are written to issue ONE fence and persist the inode.
 */
__device__ int  gpu_f2fs_write_data(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                     const void *src);

/*
 * gpu_f2fs_fsync: flush all staged writes for fd with a single fence.
 *
 * COW mode:   stages inode write → __threadfence_system() → publishes
 *             all pending data and inode holder read_ptrs.
 * Ckpt mode:  __threadfence_system() → publishes pending data holders
 *             → marks inode dirty for the next do_checkpoint().
 *
 * Must be called after a batch of gpu_f2fs_write_data() calls.
 */
__device__ void gpu_f2fs_fsync(gpu_f2fs_t *fs, int fd);

/*
 * gpu_f2fs_write_data_warp: warp-cooperative data page write (no fence).
 *
 * ALL 32 threads in the warp must call this together with the same
 * (fs, fd, pgoff) and a src buffer accessible to all threads.
 * Lane 0 handles holder allocation/lock and inode bookkeeping.
 * All 32 lanes cooperate on the PM write via stride-32 volatile stores,
 * producing coalesced 256B PCIe write transactions per loop iteration.
 *
 * Returns 0 on success (same value broadcast to all 32 lanes).
 * Must call gpu_f2fs_fsync() (from lane 0) after all pages are staged.
 */
__device__ int gpu_f2fs_write_data_warp(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                         const void *src);

/* ------------------------------------------------------------------ */
/* Host API declarations (gpu_f2fs.cu)                                */
/* ------------------------------------------------------------------ */

/*
 * beaver_f2fs_init_holders: pre-allocate one inode COW holder per inode.
 * Called from gpu_f2fs_init (COW mode only).  Implemented in beaver_f2fs.cu.
 */
void beaver_f2fs_init_holders(beaver_cache_t *cow_cache, uint32_t max_inodes);

/*
 * gpu_f2fs_init: initialise a Beaver F2FS instance.
 *
 * cow_cache  — pre-initialised beaver_cache with max_holders >= max_inodes.
 *              Required in COW mode (use_cow=1); ignored (may be NULL) in Ckpt mode.
 * data_cache — pre-initialised beaver_cache with max_holders >= max_data_pages.
 *              Required in BOTH modes (data always goes through Beaver holders).
 */
gpu_f2fs_err_t gpu_f2fs_init(gpu_f2fs_t     *fs,
                              beaver_cache_t *cow_cache,
                              beaver_cache_t *data_cache,
                              uint32_t        max_inodes,
                              uint32_t        use_cow);

gpu_f2fs_err_t gpu_f2fs_cleanup(gpu_f2fs_t *fs);

/*
 * gpu_f2fs_do_checkpoint: host wrapper.
 * COW mode: no-op.
 * Checkpoint mode: stop-the-world flush of all dirty inodes to pm_inode_base.
 */
void gpu_f2fs_do_checkpoint(gpu_f2fs_t *fs);

/*
 * gpu_vfs_mount_f2fs: bind a VFS handle to this F2FS-PM instance.
 */
void gpu_vfs_mount_f2fs(gpu_vfs_t *vfs, gpu_f2fs_t *f2fs);

/* ------------------------------------------------------------------ */
/* VFS dispatch — __forceinline__ direct calls, no runtime ptrs      */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__ int
vfs_create(gpu_vfs_t *vfs, uint32_t name_hash)
{
    return gpu_f2fs_create((gpu_f2fs_t *)vfs->fs, name_hash);
}

static __device__ __forceinline__ int
vfs_open(gpu_vfs_t *vfs, uint32_t name_hash)
{
    return gpu_f2fs_open((gpu_f2fs_t *)vfs->fs, name_hash);
}

static __device__ __forceinline__ int
vfs_write(gpu_vfs_t *vfs, int fd, uint32_t pgoff, const void *src)
{
    return gpu_f2fs_write((gpu_f2fs_t *)vfs->fs, fd, pgoff, src);
}

static __device__ __forceinline__ int
vfs_read(gpu_vfs_t *vfs, int fd, uint32_t pgoff, void *dst)
{
    return gpu_f2fs_read((gpu_f2fs_t *)vfs->fs, fd, pgoff, dst);
}

static __device__ __forceinline__ void
vfs_close(gpu_vfs_t *vfs, int fd)
{
    gpu_f2fs_close((gpu_f2fs_t *)vfs->fs, fd);
}

static __device__ __forceinline__ int
vfs_write_data(gpu_vfs_t *vfs, int fd, uint32_t pgoff, const void *src)
{
    return gpu_f2fs_write_data((gpu_f2fs_t *)vfs->fs, fd, pgoff, src);
}

static __device__ __forceinline__ void
vfs_fsync(gpu_vfs_t *vfs, int fd)
{
    gpu_f2fs_fsync((gpu_f2fs_t *)vfs->fs, fd);
}

/*
 * vfs_write_data_warp: warp-cooperative staged page write.
 * ALL 32 threads must call this together. See gpu_f2fs_write_data_warp.
 */
static __device__ __forceinline__ int
vfs_write_data_warp(gpu_vfs_t *vfs, int fd, uint32_t pgoff, const void *src)
{
    return gpu_f2fs_write_data_warp((gpu_f2fs_t *)vfs->fs, fd, pgoff, src);
}

#endif /* GPU_F2FS_H */
