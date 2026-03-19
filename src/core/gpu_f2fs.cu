/*
 * gpu_f2fs.cu — GPU F2FS: all __device__ and host implementations.
 *
 * This file contains every function body for the F2FS layer:
 *
 *   __device__ core ops  (_inode_cow_write, f2fs_inode_update,
 *                          f2fs_alloc_data_block, f2fs_write/read_data_block)
 *   __device__ POSIX ops (gpu_create, gpu_open, gpu_write, gpu_read,
 *                          gpu_close)
 *   __global__ kernels   (gpu_f2fs_init_kernel, gpu_f2fs_checkpoint_kernel,
 *                          _init_f2fs_ops_kernel)
 *   host functions       (gpu_f2fs_init, gpu_f2fs_cleanup,
 *                          gpu_f2fs_do_checkpoint, gpu_vfs_mount_f2fs)
 *
 * Header files included here:
 *   gpu_f2fs.h        — struct definitions + all declarations (no impl)
 *   beaver_cow.h      — COW inline primitives (__forceinline__, stay in .h)
 *   gpm_interface.cuh — GPM inline primitives (__forceinline__, stay in .cuh)
 */

#include "gpu_f2fs.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* ================================================================== */
/* __device__ core ops                                                 */
/* ================================================================== */

/*
 * _inode_cow_write: copy inode_shadow[nid].inode to PM via COW holder.
 *
 * Uses gpu_holder_write_and_flip (gpm_memcpy_nodrain + gpu_holder_flip):
 * exactly one __threadfence_system() drain total.
 *
 * Caller must hold inode_shadow[nid].lock.
 * This function acquires/releases h->gpu_lock (serialises COW slot access).
 */
__device__ void _inode_cow_write(gpu_f2fs_t *fs, uint32_t nid)
{
    gpu_shadow_holder_t *h = gpu_find_holder(fs->cow_cache, (uint64_t)nid);
    if (!h) return;   /* pre-allocated at init; should never be NULL */

    gpu_spin_lock(&h->gpu_lock);
    gpu_holder_write_and_flip(h, &fs->inode_shadow[nid].inode);
    gpu_spin_unlock(&h->gpu_lock);
}

/*
 * f2fs_inode_update: persist the inode (COW) or mark dirty (Checkpoint).
 * Caller must hold inode_shadow[nid].lock.
 */
 //这里存在数据浪费，dirty_flags是每个inode一个uint32_t，但实际上只用1 bit就够了。可以改成位图来节省空间。
__device__ void f2fs_inode_update(gpu_f2fs_t *fs, uint32_t nid)
{
    if (fs->use_cow)
        _inode_cow_write(fs, nid);
    else
        fs->dirty_flags[nid] = 1;
}

/*
 * f2fs_alloc_data_block: atomically claim the next free data block.
 * Returns block_addr on success, F2FS_NULL_ADDR if the data slab is full.
 */
 //相比于F2FS的2MB segment，PM的优势在于不用大粒度擦除，这其实是相对于F2FS的优化（不用修改，但备注保留）
__device__ uint32_t f2fs_alloc_data_block(gpu_f2fs_t *fs)
{
    uint32_t addr = atomicAdd(&fs->seg_cursor, 1u);
    if (addr >= fs->max_data_blocks) {
        atomicSub(&fs->seg_cursor, 1u);
        return F2FS_NULL_ADDR;
    }
    return addr;
}

/*
 * f2fs_write_data_block: out-of-place write src to PM at block_addr.
 * Uses gpm_memcpy (volatile stores + drain) for PM durability.
 */
__device__ void f2fs_write_data_block(gpu_f2fs_t *fs, uint32_t block_addr,
                                      const void *src)
{
    void *dst = (char *)fs->pm_data_base
                + (size_t)block_addr * BEAVER_PAGE_SIZE;
    gpm_memcpy(dst, src, BEAVER_PAGE_SIZE);
}

/*
 * f2fs_read_data_block: read PM page at block_addr into dst.
 * Uses volatile loads to bypass GPU L2 cache (DDIO is disabled).
 */
__device__ void f2fs_read_data_block(gpu_f2fs_t *fs, uint32_t block_addr,
                                     void *dst)
{
    const char *src = (const char *)fs->pm_data_base
                      + (size_t)block_addr * BEAVER_PAGE_SIZE;
    const volatile unsigned long long *s =
        (const volatile unsigned long long *)src;
    unsigned long long *d = (unsigned long long *)dst;
    for (uint32_t i = 0; i < BEAVER_PAGE_SIZE / sizeof(unsigned long long); ++i)
        d[i] = s[i];
}

/* ================================================================== */
/* __device__ POSIX ops                                               */
/* ================================================================== */

/*
 * gpu_create: allocate a new inode for file identified by name_hash.
 * Returns nid (fd) on success, -1 on full pool or full name_table.
 */
 //看上去inode（fs->inode_shadow）这些元数据是保持在显存上的，是提前划分好的，但理论上来说我们需要按需分配
__device__ int gpu_create(gpu_f2fs_t *fs, uint32_t name_hash)
{
    uint32_t nid = atomicAdd(&fs->inode_cursor, 1u);
    if (nid >= fs->max_inodes) {
        atomicSub(&fs->inode_cursor, 1u);
        return -1;
    }

    f2fs_inode_shadow_t *s = &fs->inode_shadow[nid];
    s->lock              = 0;
    s->inode.name_hash   = name_hash;
    s->inode.i_uid       = 0;
    s->inode.i_gid       = 0;
    s->inode.i_blocks    = 0;
    s->inode.i_size      = 0;
    s->inode.i_mode      = 0100644u;   /* regular file, rw-r--r-- */
    s->inode.i_advise    = 0;
    s->inode._pad0       = 0;
    for (uint32_t i = 0; i < F2FS_ADDRS_PER_INODE; ++i)
        s->inode.i_addr[i] = F2FS_NULL_ADDR;

    fs->dirty_flags[nid] = 0;

    /* Insert into name_table (CAS linear-probe) */
    uint32_t slot = _f2fs_name_slot(name_hash, fs->name_table_size);
    for (uint32_t i = 0; i < fs->name_table_size; ++i) {
        uint32_t old = atomicCAS(&fs->name_table[slot],
                                 F2FS_NAME_EMPTY, nid);
        if (old == F2FS_NAME_EMPTY)
            return (int)nid;
        slot = (slot + 1) % fs->name_table_size;
    }
    return -1;   /* name_table full */
}

/*
 * gpu_open: look up file by name_hash.
 * Lock-free: reads name_table and inode_shadow.name_hash only.
 * Returns nid as fd on success, -1 if not found.
 */
__device__ int gpu_open(gpu_f2fs_t *fs, uint32_t name_hash)
{
    uint32_t slot = _f2fs_name_slot(name_hash, fs->name_table_size);
    for (uint32_t i = 0; i < fs->name_table_size; ++i) {
        uint32_t nid = fs->name_table[slot];
        if (nid == F2FS_NAME_EMPTY)
            return -1;
        if (fs->inode_shadow[nid].inode.name_hash == name_hash)
            return (int)nid;
        slot = (slot + 1) % fs->name_table_size;
    }
    return -1;
}

/*
 * gpu_write: write one 4 KiB page (pgoff) from src.
 * Out-of-place: every call allocates a new data block.
 * Returns 0 on success, -1 bad fd/pgoff, -2 data slab full.
 */
 //数据本就是out-of-place的，cow只对元数据作用，这样都在PM上对比性能可能不明显
__device__ int gpu_write(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                         const void *src)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    /* 1+2: allocate data block and write to PM */
    uint32_t block_addr = f2fs_alloc_data_block(fs);
    if (block_addr == F2FS_NULL_ADDR)
        return -2;
    f2fs_write_data_block(fs, block_addr, src);

    /* 3-6: update inode under spinlock */
    f2fs_inode_shadow_t *s = &fs->inode_shadow[(uint32_t)fd];
    gpu_spin_lock(&s->lock);

    s->inode.i_addr[pgoff] = block_addr;
    s->inode.i_blocks++;
    uint64_t new_end = ((uint64_t)pgoff + 1) * BEAVER_PAGE_SIZE;
    if (new_end > s->inode.i_size)
        s->inode.i_size = new_end;

    f2fs_inode_update(fs, (uint32_t)fd);   /* COW→PM or mark dirty */

    gpu_spin_unlock(&s->lock);
    return 0;
}

/*
 * gpu_read: read one 4 KiB page (pgoff) into dst.
 * Lock-free: reads block_addr from DRAM inode shadow, then fetches from PM.
 * Returns 0 on success, -1 bad fd/pgoff or page not yet written.
 */
__device__ int gpu_read(gpu_f2fs_t *fs, int fd, uint32_t pgoff, void *dst)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    uint32_t block_addr = fs->inode_shadow[(uint32_t)fd].inode.i_addr[pgoff];
    if (block_addr == F2FS_NULL_ADDR)
        return -1;

    f2fs_read_data_block(fs, block_addr, dst);
    return 0;
}

/*
 * gpu_close: release fd.
 * No-op: fd is a stateless nid index; no ref-counting required.
 */
__device__ void gpu_close(gpu_f2fs_t *fs, int fd)
{
    (void)fs; (void)fd;
}

/* ================================================================== */
/* __global__ kernels                                                  */
/* ================================================================== */

/*
 * gpu_f2fs_init_kernel: parallel zero-init of inode_shadow, dirty_flags,
 * name_table; pre-allocates one COW holder per inode (page_id = nid).
 * Launch with enough threads to cover max(max_inodes, name_table_size).
 */
 //现在的做法是一个thread给一个inode，这似乎不合理。
 //1. 会导致warp内部的thread分散执行吗？
 //2. 这样文件的数目不就被规划好了吗？
__global__ static void gpu_f2fs_init_kernel(
        f2fs_inode_shadow_t *inode_shadow,
        uint32_t            *dirty_flags,
        uint32_t            *name_table,
        beaver_cache_t      *cow_cache,
        uint32_t             max_inodes,
        uint32_t             name_table_size)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < max_inodes) {
        f2fs_inode_shadow_t *s = &inode_shadow[i];
        s->lock              = 0;
        s->inode.name_hash   = 0;
        s->inode.i_uid       = 0;
        s->inode.i_gid       = 0;
        s->inode.i_blocks    = 0;
        s->inode.i_size      = 0;
        s->inode.i_mode      = 0;
        s->inode.i_advise    = 0;
        s->inode._pad0       = 0;
        for (uint32_t j = 0; j < F2FS_ADDRS_PER_INODE; ++j)
            s->inode.i_addr[j] = F2FS_NULL_ADDR;

        dirty_flags[i] = 0;

        /* Pre-allocate COW holder for inode i (page_id = i). */
        gpu_holder_alloc(cow_cache, (uint64_t)i);
    }

    if (i < name_table_size)
        name_table[i] = F2FS_NAME_EMPTY;
}

/*
 * gpu_f2fs_checkpoint_kernel: one thread per inode; flushes dirty inodes
 * to their PM COW holders.  Equivalent to GoFS checkpoint stop-the-world.
 */
__global__ static void gpu_f2fs_checkpoint_kernel(gpu_f2fs_t *fs)
{
    uint32_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= fs->inode_cursor)
        return;
    if (!fs->dirty_flags[nid])
        return;

    gpu_shadow_holder_t *h = gpu_find_holder(fs->cow_cache, (uint64_t)nid);
    if (!h) return;

    gpu_spin_lock(&h->gpu_lock);
    gpu_holder_write_and_flip(h, &fs->inode_shadow[nid].inode);
    gpu_spin_unlock(&h->gpu_lock);

    fs->dirty_flags[nid] = 0;
}

/* ================================================================== */
/* Host functions                                                      */
/* ================================================================== */

gpu_f2fs_err_t gpu_f2fs_init(gpu_f2fs_t *fs, beaver_cache_t *cow_cache,
                              uint32_t max_inodes, uint32_t max_data_blocks,
                              uint32_t use_cow)
{
    if (!fs || !cow_cache || max_inodes == 0 || max_data_blocks == 0)
        return GPU_F2FS_ERR_INIT;
    if (cow_cache->max_holders < max_inodes) {
        fprintf(stderr,
                "gpu_f2fs_init: cow_cache max_holders (%u) < max_inodes (%u)\n",
                cow_cache->max_holders, max_inodes);
        return GPU_F2FS_ERR_INIT;
    }

    memset(fs, 0, sizeof(*fs));
    fs->cow_cache        = cow_cache;
    fs->max_inodes       = max_inodes;
    fs->max_data_blocks  = max_data_blocks;
    fs->name_table_size  = max_inodes * 2;   /* ~50% load factor */
    fs->use_cow          = use_cow;
    fs->inode_cursor     = 0;
    fs->seg_cursor       = 0;

    cudaError_t cu;

    cu = cudaMalloc((void **)&fs->inode_shadow,
                    max_inodes * sizeof(f2fs_inode_shadow_t));
    if (cu != cudaSuccess) goto err_shadow;

    cu = cudaMalloc((void **)&fs->dirty_flags,
                    max_inodes * sizeof(uint32_t));
    if (cu != cudaSuccess) goto err_dirty;

    cu = cudaMalloc((void **)&fs->name_table,
                    fs->name_table_size * sizeof(uint32_t));
    if (cu != cudaSuccess) goto err_nametable;

    {
        size_t data_sz = (size_t)max_data_blocks * BEAVER_PAGE_SIZE;
        if (gpm_alloc(&fs->pm_data_region, data_sz, "f2fs_data") != GPM_SUCCESS) {
            fprintf(stderr, "gpu_f2fs_init: gpm_alloc(%zu MiB) failed\n",
                    data_sz >> 20);
            goto err_pmdata;
        }
        fs->pm_data_base = fs->pm_data_region.addr;
    }

    {
        uint32_t cover   = fs->name_table_size > max_inodes
                           ? fs->name_table_size : max_inodes;
        uint32_t threads = 256;
        uint32_t blocks  = (cover + threads - 1) / threads;

        gpu_f2fs_init_kernel<<<blocks, threads>>>(
                fs->inode_shadow, fs->dirty_flags, fs->name_table,
                cow_cache, max_inodes, fs->name_table_size);

        cu = cudaDeviceSynchronize();
        if (cu != cudaSuccess) {
            fprintf(stderr, "gpu_f2fs_init: init kernel failed: %s\n",
                    cudaGetErrorString(cu));
            goto err_kernel;
        }
    }

    fs->is_initialized = 1;
    printf("gpu_f2fs_init: %u inodes  %u data blocks  mode=%s\n"
           "  inode_shadow=%p  pm_data=%p (%.1f MiB)\n",
           max_inodes, max_data_blocks,
           use_cow ? "COW" : "Checkpoint",
           (void *)fs->inode_shadow,
           fs->pm_data_base,
           (double)((size_t)max_data_blocks * BEAVER_PAGE_SIZE) / (1024.0*1024.0));
    return GPU_F2FS_OK;

err_kernel:
    gpm_free(&fs->pm_data_region);
err_pmdata:
    cudaFree(fs->name_table);
err_nametable:
    cudaFree(fs->dirty_flags);
err_dirty:
    cudaFree(fs->inode_shadow);
err_shadow:
    fprintf(stderr, "gpu_f2fs_init: allocation failed: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    return GPU_F2FS_ERR_NOMEM;
}

gpu_f2fs_err_t gpu_f2fs_cleanup(gpu_f2fs_t *fs)
{
    if (!fs || !fs->is_initialized)
        return GPU_F2FS_ERR_INIT;

    gpm_free(&fs->pm_data_region);
    cudaFree(fs->name_table);
    cudaFree(fs->dirty_flags);
    cudaFree(fs->inode_shadow);
    memset(fs, 0, sizeof(*fs));
    return GPU_F2FS_OK;
}

void gpu_f2fs_do_checkpoint(gpu_f2fs_t *fs)
{
    if (!fs || !fs->is_initialized) return;
    if (fs->use_cow) return;

    uint32_t nids = fs->inode_cursor;
    if (nids == 0) return;

    uint32_t threads = 256;
    uint32_t blocks  = (nids + threads - 1) / threads;
    gpu_f2fs_checkpoint_kernel<<<blocks, threads>>>(fs);
    cudaDeviceSynchronize();
}

/* ================================================================== */
/* VFS mount                                                           */
/* ================================================================== */

/*
 * gpu_vfs_mount_f2fs: bind a VFS handle to this F2FS-PM instance.
 *
 * vfs_* dispatch functions (gpu_f2fs.h) are __forceinline__ direct calls
 * resolved at device-link time — no runtime function pointer setup needed.
 */
void gpu_vfs_mount_f2fs(gpu_vfs_t *vfs, gpu_f2fs_t *f2fs)
{
    vfs->fs   = f2fs;
    vfs->type = GPU_FS_F2FS_PM;
}
