/*
 * gpu_f2fs_ckpt.cu — Pure GPU F2FS baseline (no Beaver).
 *
 * Implements CPU-side F2FS semantics directly on GPU with PM as storage:
 *   - Data: out-of-place log writes (atomicAdd seg_cursor → PM block).
 *   - Inode: DRAM working copy, dirty_flags for checkpoint tracking.
 *   - Checkpoint: stop-the-world gpm_memcpy of dirty inodes to fixed PM area.
 *
 * This file includes ONLY f2fs_types.h and gpm_interface.cuh.
 * No beaver_cow.h, no Beaver functions.
 */

#include "gpu_f2fs_ckpt.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* ================================================================== */
/* GPU spinlock helpers (local, no beaver_cow.h dependency)           */
/* ================================================================== */

static __device__ __forceinline__ void
ckpt_spin_lock(uint32_t *lock)
{
    while (atomicCAS(lock, 0u, 1u) != 0u)
        __nanosleep(32);
}

static __device__ __forceinline__ void
ckpt_spin_unlock(uint32_t *lock)
{
    __threadfence();
    atomicExch(lock, 0u);
}

/* ================================================================== */
/* GPU F2FS POSIX ops                                                  */
/* ================================================================== */

__device__ int gpu_ckpt_create(gpu_ckpt_fs_t *fs, uint32_t name_hash)
{
    uint32_t nid = atomicAdd(&fs->inode_cursor, 1u);
    if (nid >= fs->max_inodes) {
        atomicSub(&fs->inode_cursor, 1u);
        return -1;
    }

    /* Zero-init inode in DRAM */
    f2fs_inode_t *in = &fs->inodes[nid];
    in->name_hash  = name_hash;
    in->i_uid      = 0;
    in->i_gid      = 0;
    in->i_blocks   = 0;
    in->i_size     = 0;
    in->i_mode     = 0100644u;
    in->i_advise   = 0;
    in->_pad0      = 0;
    for (uint32_t i = 0; i < F2FS_ADDRS_PER_INODE; ++i)
        in->i_addr[i] = F2FS_NULL_ADDR;

    fs->inode_locks[nid] = 0;
    fs->dirty_flags[nid] = 0;

    /* CAS linear-probe insert into name_table */
    uint32_t slot = f2fs_name_slot(name_hash, fs->name_table_size);
    for (uint32_t i = 0; i < fs->name_table_size; ++i) {
        uint32_t old = atomicCAS(&fs->name_table[slot], F2FS_NAME_EMPTY, nid);
        if (old == F2FS_NAME_EMPTY)
            return (int)nid;
        slot = (slot + 1) % fs->name_table_size;
    }
    return -1;
}

__device__ int gpu_ckpt_open(gpu_ckpt_fs_t *fs, uint32_t name_hash)
{
    uint32_t slot = f2fs_name_slot(name_hash, fs->name_table_size);
    for (uint32_t i = 0; i < fs->name_table_size; ++i) {
        uint32_t nid = fs->name_table[slot];
        if (nid == F2FS_NAME_EMPTY)
            return -1;
        if (fs->inodes[nid].name_hash == name_hash)
            return (int)nid;
        slot = (slot + 1) % fs->name_table_size;
    }
    return -1;
}

__device__ int gpu_ckpt_write(gpu_ckpt_fs_t *fs, int fd, uint32_t pgoff,
                               const void *src)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    /* Allocate next out-of-place PM data block */
    uint32_t block_addr = atomicAdd(&fs->seg_cursor, 1u);
    if (block_addr >= fs->max_data_blocks) {
        atomicSub(&fs->seg_cursor, 1u);
        return -2;
    }

    /* Write data directly to PM (no COW, no double-buffer) */
    void *dst_pm = (char *)fs->pm_data_base
                   + (size_t)block_addr * F2FS_PAGE_SIZE;
    gpm_memcpy(dst_pm, src, F2FS_PAGE_SIZE);

    /* Update inode under spinlock */
    ckpt_spin_lock(&fs->inode_locks[(uint32_t)fd]);

    f2fs_inode_t *in = &fs->inodes[(uint32_t)fd];
    in->i_addr[pgoff] = block_addr;
    in->i_blocks++;
    uint64_t new_end = ((uint64_t)pgoff + 1) * F2FS_PAGE_SIZE;
    if (new_end > in->i_size)
        in->i_size = new_end;

    /* Mark inode dirty — will be flushed at next checkpoint */
    fs->dirty_flags[(uint32_t)fd] = 1;

    ckpt_spin_unlock(&fs->inode_locks[(uint32_t)fd]);
    return 0;
}

__device__ int gpu_ckpt_read(gpu_ckpt_fs_t *fs, int fd, uint32_t pgoff,
                              void *dst)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    uint32_t block_addr = fs->inodes[(uint32_t)fd].i_addr[pgoff];
    if (block_addr == F2FS_NULL_ADDR)
        return -1;

    const char *src_pm = (const char *)fs->pm_data_base
                         + (size_t)block_addr * F2FS_PAGE_SIZE;
    const volatile unsigned long long *s =
        (const volatile unsigned long long *)src_pm;
    unsigned long long *d = (unsigned long long *)dst;
    for (uint32_t i = 0; i < F2FS_PAGE_SIZE / 8; ++i)
        d[i] = s[i];

    return 0;
}

__device__ void gpu_ckpt_close(gpu_ckpt_fs_t *fs, int fd)
{
    (void)fs; (void)fd;
}

/* ================================================================== */
/* __global__ kernels                                                  */
/* ================================================================== */

__global__ static void gpu_ckpt_init_kernel(
        f2fs_inode_t *inodes,
        uint32_t     *inode_locks,
        uint32_t     *dirty_flags,
        uint32_t     *name_table,
        uint32_t      max_inodes,
        uint32_t      name_table_size)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < max_inodes) {
        f2fs_inode_t *in = &inodes[i];
        in->name_hash  = 0;
        in->i_uid      = 0;
        in->i_gid      = 0;
        in->i_blocks   = 0;
        in->i_size     = 0;
        in->i_mode     = 0;
        in->i_advise   = 0;
        in->_pad0      = 0;
        for (uint32_t j = 0; j < F2FS_ADDRS_PER_INODE; ++j)
            in->i_addr[j] = F2FS_NULL_ADDR;
        inode_locks[i] = 0;
        dirty_flags[i] = 0;
    }

    if (i < name_table_size)
        name_table[i] = F2FS_NAME_EMPTY;
}

/*
 * gpu_ckpt_checkpoint_kernel: stop-the-world inode flush.
 *
 * Each thread writes one dirty inode from DRAM to its fixed PM slot.
 * Mirrors F2FS checkpoint pack: dirty node pages flushed to fixed area.
 */
__global__ static void gpu_ckpt_checkpoint_kernel(gpu_ckpt_fs_t *fs)
{
    uint32_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= fs->inode_cursor)
        return;
    if (!fs->dirty_flags[nid])
        return;

    void *dst = (char *)fs->pm_ckpt_base + (size_t)nid * F2FS_PAGE_SIZE;
    gpm_memcpy(dst, &fs->inodes[nid], F2FS_PAGE_SIZE);
    fs->dirty_flags[nid] = 0;
}

/* ================================================================== */
/* Host functions                                                      */
/* ================================================================== */

gpu_ckpt_err_t gpu_ckpt_fs_init(gpu_ckpt_fs_t *fs,
                                 uint32_t max_inodes,
                                 uint32_t max_data_blocks)
{
    if (!fs || max_inodes == 0 || max_data_blocks == 0)
        return GPU_CKPT_ERR_INIT;

    memset(fs, 0, sizeof(*fs));
    fs->max_inodes      = max_inodes;
    fs->max_data_blocks = max_data_blocks;
    fs->name_table_size = max_inodes * 2;
    fs->inode_cursor    = 0;
    fs->seg_cursor      = 0;

    cudaError_t cu;

    cu = cudaMalloc((void **)&fs->inodes,
                    max_inodes * sizeof(f2fs_inode_t));
    if (cu != cudaSuccess) goto err_inodes;

    cu = cudaMalloc((void **)&fs->inode_locks,
                    max_inodes * sizeof(uint32_t));
    if (cu != cudaSuccess) goto err_locks;

    cu = cudaMalloc((void **)&fs->dirty_flags,
                    max_inodes * sizeof(uint32_t));
    if (cu != cudaSuccess) goto err_dirty;

    cu = cudaMalloc((void **)&fs->name_table,
                    fs->name_table_size * sizeof(uint32_t));
    if (cu != cudaSuccess) goto err_nametable;

    /* PM data slab: one block per write (out-of-place) */
    {
        size_t data_sz = (size_t)max_data_blocks * F2FS_PAGE_SIZE;
        if (gpm_alloc(&fs->pm_data_region, data_sz, "ckpt_data") != GPM_SUCCESS) {
            fprintf(stderr,
                    "gpu_ckpt_fs_init: gpm_alloc(data %zu MiB) failed\n",
                    data_sz >> 20);
            goto err_pmdata;
        }
        fs->pm_data_base = fs->pm_data_region.addr;
    }

    /* PM checkpoint pack: fixed inode area */
    {
        size_t ckpt_sz = (size_t)max_inodes * F2FS_PAGE_SIZE;
        if (gpm_alloc(&fs->pm_ckpt_region, ckpt_sz, "ckpt_inodes") != GPM_SUCCESS) {
            fprintf(stderr,
                    "gpu_ckpt_fs_init: gpm_alloc(ckpt %zu MiB) failed\n",
                    ckpt_sz >> 20);
            goto err_pmckpt;
        }
        fs->pm_ckpt_base = fs->pm_ckpt_region.addr;
    }

    /* Parallel zero-init on GPU */
    {
        uint32_t cover   = fs->name_table_size > max_inodes
                           ? fs->name_table_size : max_inodes;
        uint32_t threads = 256;
        uint32_t blocks  = (cover + threads - 1) / threads;

        gpu_ckpt_init_kernel<<<blocks, threads>>>(
                fs->inodes, fs->inode_locks, fs->dirty_flags,
                fs->name_table, max_inodes, fs->name_table_size);

        cu = cudaDeviceSynchronize();
        if (cu != cudaSuccess) {
            fprintf(stderr, "gpu_ckpt_fs_init: init kernel failed: %s\n",
                    cudaGetErrorString(cu));
            goto err_kernel;
        }
    }

    fs->is_initialized = 1;
    printf("gpu_ckpt_fs_init: %u inodes  %u data blocks\n"
           "  inodes=%p  pm_data=%p (%.1f MiB)  pm_ckpt=%p\n",
           max_inodes, max_data_blocks,
           (void *)fs->inodes,
           fs->pm_data_base,
           (double)((size_t)max_data_blocks * F2FS_PAGE_SIZE) / (1024.0*1024.0),
           fs->pm_ckpt_base);
    return GPU_CKPT_OK;

err_kernel:
    gpm_free(&fs->pm_ckpt_region);
err_pmckpt:
    gpm_free(&fs->pm_data_region);
err_pmdata:
    cudaFree(fs->name_table);
err_nametable:
    cudaFree(fs->dirty_flags);
err_dirty:
    cudaFree(fs->inode_locks);
err_locks:
    cudaFree(fs->inodes);
err_inodes:
    fprintf(stderr, "gpu_ckpt_fs_init: allocation failed: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return GPU_CKPT_ERR_NOMEM;
}

gpu_ckpt_err_t gpu_ckpt_fs_cleanup(gpu_ckpt_fs_t *fs)
{
    if (!fs || !fs->is_initialized)
        return GPU_CKPT_ERR_INIT;

    gpm_free(&fs->pm_ckpt_region);
    gpm_free(&fs->pm_data_region);
    cudaFree(fs->name_table);
    cudaFree(fs->dirty_flags);
    cudaFree(fs->inode_locks);
    cudaFree(fs->inodes);
    memset(fs, 0, sizeof(*fs));
    return GPU_CKPT_OK;
}

void gpu_ckpt_do_checkpoint(gpu_ckpt_fs_t *fs)
{
    if (!fs || !fs->is_initialized) return;

    uint32_t nids = fs->inode_cursor;
    if (nids == 0) return;

    uint32_t threads = 256;
    uint32_t blocks  = (nids + threads - 1) / threads;
    gpu_ckpt_checkpoint_kernel<<<blocks, threads>>>(fs);
    cudaDeviceSynchronize();
}
