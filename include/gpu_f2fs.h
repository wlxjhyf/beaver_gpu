/*
 * gpu_f2fs.h — GPU F2FS: structures, declarations, and VFS dispatch.
 *
 * This header contains:
 *   1. Struct definitions (f2fs_inode_t, gpu_f2fs_t, …)
 *   2. Constants and error codes
 *   3. Inline helpers (_f2fs_name_slot; vfs_* dispatch)
 *   4. __device__ function declarations (implementations in gpu_f2fs.cu)
 *   5. Host API declarations (implementations in gpu_f2fs.cu)
 *
 * VFS dispatch (vfs_create / vfs_open / vfs_write / vfs_read / vfs_close)
 * is defined here as __forceinline__ direct calls to gpu_create etc.
 * This avoids CUDA separable-compilation limitations: cross-TU device
 * function pointer calls (runtime indirect dispatch) cause
 * cudaErrorIllegalInstruction on Ampere.  Direct calls are resolved at
 * device-link time and work correctly across translation units.
 *
 * Two persistence modes, selected at init time via use_cow:
 *   COW mode  (use_cow=1): every inode update → Beaver COW holder → PM.
 *   Checkpoint mode (use_cow=0): inode update → DRAM dirty flag;
 *       gpu_f2fs_do_checkpoint() flushes all dirty inodes (stop-the-world).
 *
 * Layering:
 *   gpu_vfs.h    — interface types only (gpu_vfs_t, gpu_fs_type_t)
 *   gpu_f2fs.h   — F2FS structures + vfs_* dispatch  ← here
 *   beaver_cow.h — COW holders + PM persistence primitives
 */

#ifndef GPU_F2FS_H
#define GPU_F2FS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "beaver_cow.h"
#include "gpu_vfs.h"

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/*
 * Direct block address entries in one inode page.
 * Header = 28 bytes; remaining = 4096 - 28 = 4068 = 1017 × 4.
 */
#define F2FS_ADDRS_PER_INODE   1017u

/* Sentinel: block not yet allocated (matches F2FS F2FS_NULL_ADDR). */
#define F2FS_NULL_ADDR         0xFFFFFFFFu

/* Sentinel: name_table slot is empty. */
#define F2FS_NAME_EMPTY        0xFFFFFFFFu

/* ------------------------------------------------------------------ */
/* f2fs_inode_t — on-disk inode page (exactly 4096 bytes)            */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t  name_hash;                   /* file name hash (open lookup)  */
    uint32_t  i_uid;
    uint32_t  i_gid;
    uint32_t  i_blocks;                    /* number of allocated data blks */
    uint64_t  i_size;                      /* file size in bytes            */
    uint16_t  i_mode;
    uint8_t   i_advise;
    uint8_t   _pad0;
    /* offset 28 — direct block addresses */
    uint32_t  i_addr[F2FS_ADDRS_PER_INODE];
} f2fs_inode_t;   /* sizeof must equal BEAVER_PAGE_SIZE = 4096 */

#ifdef __cplusplus
static_assert(sizeof(f2fs_inode_t) == BEAVER_PAGE_SIZE,
              "f2fs_inode_t must be exactly 4096 bytes");
#endif

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
    uint32_t            *dirty_flags;    /* cudaMalloc, 1 = dirty (ckpt)   */
    uint32_t             max_inodes;
    uint32_t             inode_cursor;   /* atomicAdd — allocate nid       */

    /* Name lookup */
    uint32_t            *name_table;     /* cudaMalloc, hash→nid           */
    uint32_t             name_table_size;

    /* Out-of-place data writes */
    void                *pm_data_base;   /* UVA ptr into PM data slab      */
    gpm_region_t         pm_data_region;
    uint32_t             seg_cursor;     /* atomicAdd — next data block    */
    uint32_t             max_data_blocks;

    /* Metadata persistence */
    beaver_cache_t      *cow_cache;      /* inode COW holders, page_id=nid */

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
/* Trivial host+device inline helper — stays in header                */
/*                                                                     */
/* _f2fs_name_slot: 6-line hash mix, called at every open/create.    */
/* Keeping it __forceinline__ avoids the call overhead on the device  */
/* and lets the host call it without a separate compilation unit.     */
/* ------------------------------------------------------------------ */
static __device__ __host__ __forceinline__ uint32_t
_f2fs_name_slot(uint32_t h, uint32_t sz)
{
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h % sz;
}

/* ------------------------------------------------------------------ */
/* __device__ core function declarations (implementations: gpu_f2fs.cu) */
/* ------------------------------------------------------------------ */

/*
 * _inode_cow_write: copy inode_shadow[nid].inode to PM via COW holder.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void _inode_cow_write(gpu_f2fs_t *fs, uint32_t nid);

/*
 * f2fs_inode_update: persist inode (COW path) or mark dirty (Checkpoint).
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void f2fs_inode_update(gpu_f2fs_t *fs, uint32_t nid);

/* f2fs_alloc_data_block: atomically claim next free data block. */
__device__ uint32_t f2fs_alloc_data_block(gpu_f2fs_t *fs);

/* f2fs_write_data_block: out-of-place volatile write to PM at block_addr. */
__device__ void f2fs_write_data_block(gpu_f2fs_t *fs, uint32_t block_addr,
                                      const void *src);

/* f2fs_read_data_block: volatile read from PM at block_addr into dst. */
__device__ void f2fs_read_data_block(gpu_f2fs_t *fs, uint32_t block_addr,
                                     void *dst);

/* ------------------------------------------------------------------ */
/* Host API declarations (implementations: gpu_f2fs.cu)              */
/* ------------------------------------------------------------------ */

gpu_f2fs_err_t gpu_f2fs_init(gpu_f2fs_t *fs, beaver_cache_t *cow_cache,
                              uint32_t max_inodes, uint32_t max_data_blocks,
                              uint32_t use_cow);

gpu_f2fs_err_t gpu_f2fs_cleanup(gpu_f2fs_t *fs);

/*
 * gpu_f2fs_do_checkpoint: host wrapper.
 * COW mode: no-op.  Checkpoint mode: stop-the-world flush of all dirty inodes.
 */
void gpu_f2fs_do_checkpoint(gpu_f2fs_t *fs);

/*
 * gpu_vfs_mount_f2fs: bind a VFS handle to this F2FS-PM instance.
 * Registers F2FS ops into vfs->ops (device memory).
 * Call after gpu_f2fs_init() succeeds.
 */
void gpu_vfs_mount_f2fs(gpu_vfs_t *vfs, gpu_f2fs_t *f2fs);

/* ------------------------------------------------------------------ */
/* __device__ POSIX API (implementations: gpu_f2fs.cu)               */
/* ------------------------------------------------------------------ */

/*
 * gpu_create: allocate a new inode for the file identified by name_hash.
 * Returns nid (fd) on success, -1 if inode pool or name_table is full.
 */
__device__ int gpu_create(gpu_f2fs_t *fs, uint32_t name_hash);

/*
 * gpu_open: look up file by name_hash.
 * Lock-free. Returns nid as fd on success, -1 if not found.
 */
__device__ int gpu_open(gpu_f2fs_t *fs, uint32_t name_hash);

/*
 * gpu_write: write one 4 KiB page (pgoff) from src.
 * Out-of-place: every call allocates a new data block.
 * Returns 0 on success, -1 bad fd/pgoff, -2 data slab full.
 */
__device__ int gpu_write(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                         const void *src);

/*
 * gpu_read: read one 4 KiB page (pgoff) into dst.
 * Lock-free. Returns 0 on success, -1 bad fd/pgoff or page not written.
 */
__device__ int gpu_read(gpu_f2fs_t *fs, int fd, uint32_t pgoff, void *dst);

/*
 * gpu_close: release fd. No-op: fd is a stateless nid index.
 */
__device__ void gpu_close(gpu_f2fs_t *fs, int fd);

/* ------------------------------------------------------------------ */
/* VFS dispatch — direct inline calls, no runtime function pointers   */
/*                                                                     */
/* Defined here (not in gpu_vfs.h) because they need gpu_f2fs_t and  */
/* the gpu_* POSIX declarations above.  All are __forceinline__ so   */
/* they compile to direct device function calls — resolved at         */
/* device-link time, safe across translation units.                  */
/*                                                                     */
/* To add a second FS type (e.g. GPU_FS_F2FS_SSD), add a switch on  */
/* vfs->type and call the SSD-backed equivalents.                    */
/* ------------------------------------------------------------------ */

static __device__ __forceinline__ int
vfs_create(gpu_vfs_t *vfs, uint32_t name_hash)
{
    return gpu_create((gpu_f2fs_t *)vfs->fs, name_hash);
}

static __device__ __forceinline__ int
vfs_open(gpu_vfs_t *vfs, uint32_t name_hash)
{
    return gpu_open((gpu_f2fs_t *)vfs->fs, name_hash);
}

static __device__ __forceinline__ int
vfs_write(gpu_vfs_t *vfs, int fd, uint32_t pgoff, const void *src)
{
    return gpu_write((gpu_f2fs_t *)vfs->fs, fd, pgoff, src);
}

static __device__ __forceinline__ int
vfs_read(gpu_vfs_t *vfs, int fd, uint32_t pgoff, void *dst)
{
    return gpu_read((gpu_f2fs_t *)vfs->fs, fd, pgoff, dst);
}

static __device__ __forceinline__ void
vfs_close(gpu_vfs_t *vfs, int fd)
{
    gpu_close((gpu_f2fs_t *)vfs->fs, fd);
}

#endif /* GPU_F2FS_H */
