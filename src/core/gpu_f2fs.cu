/*
 * gpu_f2fs.cu — GPU F2FS: pure file-system operations.
 *
 * This file contains only F2FS layer code — no Beaver mechanics.
 * Beaver operations (beaver_inode_persist, beaver_data_write/read,
 * holder init) live in beaver_f2fs.cu.
 *
 * Data path: every write goes through beaver_data_write (COW holder).
 * Inode path: COW mode → beaver_inode_persist; Ckpt mode → dirty_flags.
 *
 * inode.i_addr[pgoff] stores the holder_idx in data_cache, NOT a raw
 * PM block address.  F2FS_NULL_ADDR means the page has not been written.
 */

#include "gpu_f2fs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* ================================================================== */
/* F2FS inode ops                                                      */
/* ================================================================== */

/*
 * f2fs_inode_update: persist the inode (Beaver COW) or mark dirty (Checkpoint).
 * This is the seam between the F2FS layer and the consistency layer.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void f2fs_inode_update(gpu_f2fs_t *fs, uint32_t nid)
{
    if (fs->use_cow)
        beaver_inode_persist(fs, nid);
    else
        fs->dirty_flags[nid] = 1;
}

/* ================================================================== */
/* GPU F2FS POSIX ops                                                  */
/* ================================================================== */

/*
 * gpu_f2fs_create: allocate a new inode, zero-init, insert into name_table.
 * Returns nid (fd) on success, -1 on full pool or full name_table.
 */
// 合理，只是多线程加速create
__device__ int gpu_f2fs_create(gpu_f2fs_t *fs, uint32_t name_hash)
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
    s->inode.i_mode      = 0100644u;
    s->inode.i_advise    = 0;
    s->inode._pad0       = 0;
    for (uint32_t i = 0; i < F2FS_ADDRS_PER_INODE; ++i)
        s->inode.i_addr[i] = F2FS_NULL_ADDR;

    fs->dirty_flags[nid] = 0;

    /* CAS linear-probe insert into name_table */
    uint32_t slot = f2fs_name_slot(name_hash, fs->name_table_size);
    for (uint32_t i = 0; i < fs->name_table_size; ++i) {
        uint32_t old = atomicCAS(&fs->name_table[slot],
                                 F2FS_NAME_EMPTY, nid);
        if (old == F2FS_NAME_EMPTY)
            return (int)nid;
        slot = (slot + 1) % fs->name_table_size;
    }
    return -1;
}

/*
 * gpu_f2fs_open: look up file by name_hash.
 * Lock-free. Returns nid as fd on success, -1 if not found.
 */
__device__ int gpu_f2fs_open(gpu_f2fs_t *fs, uint32_t name_hash)
{
    uint32_t slot = f2fs_name_slot(name_hash, fs->name_table_size);
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
 * gpu_f2fs_write: write one 4 KiB page (pgoff) from src.
 *
 * Data goes through beaver_data_write (COW holder, PM log, single drain).
 * i_addr[pgoff] is updated to hold the holder_idx in data_cache.
 * Inode (i_size, i_blocks, i_addr) is persisted via f2fs_inode_update.
 *
 * Returns 0 on success, -1 bad fd/pgoff, -2 data_cache full.
 */
 //代码合理，应该是核函数本身不合理导致的
__device__ int gpu_f2fs_write(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                               const void *src)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    f2fs_inode_shadow_t *s = &fs->inode_shadow[(uint32_t)fd];
    beaver_spin_lock(&s->lock);

    uint32_t existing = s->inode.i_addr[pgoff];   /* F2FS_NULL_ADDR or holder_idx */

    uint32_t new_idx = beaver_data_write(fs, (uint32_t)fd, pgoff, existing, src);
    if (new_idx == (uint32_t)F2FS_NULL_ADDR) {
        beaver_spin_unlock(&s->lock);
        return -2;   /* data_cache full */
    }

    /* Update inode metadata */
    if (existing == (uint32_t)F2FS_NULL_ADDR) {
        s->inode.i_addr[pgoff] = new_idx;
        s->inode.i_blocks++;
    }
    uint64_t new_end = ((uint64_t)pgoff + 1) * BEAVER_PAGE_SIZE;
    if (new_end > s->inode.i_size)
        s->inode.i_size = new_end;

    f2fs_inode_update(fs, (uint32_t)fd);

    beaver_spin_unlock(&s->lock);
    return 0;
}

/*
 * gpu_f2fs_write_data: write one data page to PM (no fence, no inode persist).
 *
 * Stages the data write via beaver_data_write_stage and updates DRAM inode
 * fields (i_addr, i_blocks, i_size).  No __threadfence_system() is issued.
 * Call gpu_f2fs_fsync() after all pages in the batch are written.
 */
__device__ int gpu_f2fs_write_data(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                    const void *src)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes) return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)            return -1;

    f2fs_inode_shadow_t *s = &fs->inode_shadow[(uint32_t)fd];
    beaver_spin_lock(&s->lock);

    uint32_t existing = s->inode.i_addr[pgoff];
    uint32_t new_idx  = beaver_data_write_stage(fs, (uint32_t)fd, pgoff,
                                                 existing, src);
    if (new_idx == (uint32_t)F2FS_NULL_ADDR) {
        beaver_spin_unlock(&s->lock);
        return -2;
    }

    if (existing == (uint32_t)F2FS_NULL_ADDR) {
        s->inode.i_addr[pgoff] = new_idx;
        s->inode.i_blocks++;
    }
    uint64_t new_end = ((uint64_t)pgoff + 1) * BEAVER_PAGE_SIZE;
    if (new_end > s->inode.i_size)
        s->inode.i_size = new_end;

    beaver_spin_unlock(&s->lock);
    return 0;
}

/*
 * gpu_f2fs_write_data_warp: warp-cooperative data page write (no fence).
 *
 * ALL 32 threads in the warp must call this function together with the
 * same (fs, fd, pgoff) and a src buffer accessible by all 32 threads.
 *
 * Lane 0 handles holder allocation/lock and inode bookkeeping.
 * All 32 lanes cooperate on the PM write via stride-32 volatile stores,
 * producing coalesced 256B PCIe write transactions per loop iteration
 * (vs. 512 independent 8B transactions in the scalar gpu_f2fs_write_data).
 *
 * The PM write address is broadcast from lane 0 to all lanes via warp shuffle.
 * s->lock and h->gpu_lock are held across the cooperative PM write phase and
 * released by lane 0 in Phase 3.
 *
 * Returns 0 on success (all lanes receive the same value).
 * Must call gpu_f2fs_fsync() (from lane 0) after all pages are written.
 */
__device__ int gpu_f2fs_write_data_warp(gpu_f2fs_t *fs, int fd, uint32_t pgoff,
                                         const void *src)
{
    uint32_t lane = threadIdx.x & 31u;

    /* ── Phase 1 : lane 0 allocates/finds the holder, acquires write addr ─ */
    int       phase1_ok = 0;
    uintptr_t waddr_int = 0;                  /* PM write address as integer  */
    f2fs_inode_shadow_t *p1_s = NULL;         /* inode shadow (lane 0 only)   */
    beaver_holder_t     *p1_h = NULL;         /* data holder  (lane 0 only)   */

    if (lane == 0) {
        if (fd < 0 || (uint32_t)fd >= fs->max_inodes ||
            pgoff >= F2FS_ADDRS_PER_INODE) {
            phase1_ok = -1;
        } else {
            p1_s = &fs->inode_shadow[(uint32_t)fd];
            beaver_spin_lock(&p1_s->lock);

            uint32_t existing = p1_s->inode.i_addr[pgoff];

            if (existing != (uint32_t)F2FS_NULL_ADDR) {
                /* Reuse existing holder: COW flips to the inactive slot */
                p1_h = &fs->data_cache->holders[existing];
            } else {
                /* First write to this page: claim a new holder */
                uint32_t new_idx = atomicAdd(&fs->data_cache->alloc_cursor, 1u);
                if (new_idx >= fs->data_cache->max_holders) {
                    atomicSub(&fs->data_cache->alloc_cursor, 1u);
                    beaver_spin_unlock(&p1_s->lock);
                    phase1_ok = -2;         /* data_cache full */
                } else {
                    p1_h = &fs->data_cache->holders[new_idx];
                    p1_h->gpu_lock = 0;
                    p1_h->cur      = -1;
                    p1_h->state    = HOLDER_INIT;
                    p1_h->_pad     = 0;
                    p1_h->read_ptr = NULL;
                    p1_h->page_id  = beaver_data_page_id((uint32_t)fd, pgoff);
                    beaver_hash_insert(fs->data_cache->hash_table,
                                       fs->data_cache->hash_size,
                                       p1_h->page_id, new_idx);
                    p1_s->inode.i_addr[pgoff] = new_idx;
                    p1_s->inode.i_blocks++;
                }
            }

            if (phase1_ok == 0) {
                beaver_spin_lock(&p1_h->gpu_lock);
                waddr_int = (uintptr_t)beaver_holder_write_addr(p1_h);
                uint64_t new_end = ((uint64_t)pgoff + 1) * BEAVER_PAGE_SIZE;
                if (new_end > p1_s->inode.i_size)
                    p1_s->inode.i_size = new_end;
                /* p1_s->lock and p1_h->gpu_lock remain held through Phase 3 */
            }
        }
    }

    /* ── Phase 2 : broadcast PM dest, all 32 lanes write cooperatively ───── */
    uint32_t w_lo = __shfl_sync(0xFFFFFFFFu, (uint32_t)(waddr_int & 0xFFFFFFFFu), 0);
    uint32_t w_hi = __shfl_sync(0xFFFFFFFFu, (uint32_t)(waddr_int >> 32),          0);
    void *pm_dst  = (void *)((uintptr_t)w_hi << 32 | w_lo);

    if (pm_dst != NULL) {
        /* stride-32 stores: adjacent lanes → adjacent PM addresses → coalesced */
        const uint32_t n_words = BEAVER_PAGE_SIZE / sizeof(unsigned long long);
        volatile unsigned long long       *d    = (volatile unsigned long long *)pm_dst;
        const          unsigned long long *sbuf = (const unsigned long long *)src;
        for (uint32_t i = lane; i < n_words; i += 32u)
            d[i] = sbuf[i];
    }

    /* ── Phase 3 : lane 0 updates holder state, writes log, releases locks ── */
    if (lane == 0 && phase1_ok == 0) {
        int next   = (p1_h->cur < 0) ? 0 : (p1_h->cur + 1) % 2;
        p1_h->cur  = next;
        beaver_log_write(&fs->log, BLOG_DATA_FLIP,
                         (uint32_t)fd, pgoff, (uint32_t)next);
        beaver_spin_unlock(&p1_h->gpu_lock);
        beaver_spin_unlock(&p1_s->lock);
    }

    /* Broadcast return code so all lanes get the same value */
    phase1_ok = __shfl_sync(0xFFFFFFFFu, phase1_ok, 0);
    return phase1_ok;
}

/*
 * gpu_f2fs_fsync: flush all staged writes with a single __threadfence_system().
 *
 * COW mode:   stage inode → fence → publish all data + inode holders.
 * Ckpt mode:  fence → publish all data holders → mark inode dirty.
 */
__device__ void gpu_f2fs_fsync(gpu_f2fs_t *fs, int fd)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes) return;

    f2fs_inode_shadow_t *s = &fs->inode_shadow[(uint32_t)fd];
    beaver_spin_lock(&s->lock);

    if (fs->use_cow)
        beaver_inode_persist_stage(fs, (uint32_t)fd);

    __threadfence_system();   /* single drain: all staged PM writes durable */

    /* Publish data holders up to last written page */
    uint32_t max_pgoff = (uint32_t)
        ((s->inode.i_size + BEAVER_PAGE_SIZE - 1) / BEAVER_PAGE_SIZE);
    if (max_pgoff > F2FS_ADDRS_PER_INODE)
        max_pgoff = F2FS_ADDRS_PER_INODE;
    for (uint32_t p = 0; p < max_pgoff; p++) {
        uint32_t idx = s->inode.i_addr[p];
        if (idx != (uint32_t)F2FS_NULL_ADDR)
            beaver_holder_publish(&fs->data_cache->holders[idx]);
    }

    if (fs->use_cow) {
        beaver_holder_t *ih = beaver_find_holder(fs->cow_cache, (uint64_t)fd);
        if (ih) beaver_holder_publish(ih);
    } else {
        fs->dirty_flags[(uint32_t)fd] = 1;
    }

    beaver_spin_unlock(&s->lock);
}

/*
 * gpu_f2fs_read: read one 4 KiB page (pgoff) into dst.
 * Lock-free: reads holder_idx from DRAM inode shadow, then fetches
 * from the Beaver data holder's current PM slot.
 * Returns 0 on success, -1 on bad fd/pgoff or page not written.
 */
__device__ int gpu_f2fs_read(gpu_f2fs_t *fs, int fd, uint32_t pgoff, void *dst)
{
    if (fd < 0 || (uint32_t)fd >= fs->max_inodes)
        return -1;
    if (pgoff >= F2FS_ADDRS_PER_INODE)
        return -1;

    uint32_t holder_idx = fs->inode_shadow[(uint32_t)fd].inode.i_addr[pgoff];
    if (holder_idx == (uint32_t)F2FS_NULL_ADDR)
        return -1;

    return beaver_data_read(fs, holder_idx, dst);
}

/*
 * gpu_f2fs_close: release fd.
 * No-op: fd is a stateless nid index; no ref-counting required.
 */
__device__ void gpu_f2fs_close(gpu_f2fs_t *fs, int fd)
{
    (void)fs; (void)fd;
}

/* ================================================================== */
/* __global__ kernels                                                  */
/* ================================================================== */

/*
 * gpu_f2fs_init_kernel: parallel zero-init of inode_shadow, dirty_flags,
 * name_table.  Pure F2FS — no Beaver.
 */
__global__ static void gpu_f2fs_init_kernel(
        f2fs_inode_shadow_t *inode_shadow,
        uint32_t            *dirty_flags,
        uint32_t            *name_table,
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
    }

    if (i < name_table_size)
        name_table[i] = F2FS_NAME_EMPTY;
}

/*
 * gpu_f2fs_checkpoint_kernel: Checkpoint mode inode flush.
 *
 * One thread per inode: writes dirty inode pages from DRAM (inode_shadow)
 * to fixed PM slots (pm_inode_base + nid * PAGE_SIZE).
 * Stop-the-world: all GPU threads stall until checkpoint completes.
 */
__global__ static void gpu_f2fs_checkpoint_kernel(gpu_f2fs_t *fs)
{
    uint32_t nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= fs->inode_cursor)
        return;
    if (!fs->dirty_flags[nid])
        return;

    void *dst = (char *)fs->pm_inode_base + (size_t)nid * BEAVER_PAGE_SIZE;
    gpm_memcpy(dst, &fs->inode_shadow[nid].inode, BEAVER_PAGE_SIZE);
    fs->dirty_flags[nid] = 0;
}

/* ================================================================== */
/* Host functions                                                      */
/* ================================================================== */

gpu_f2fs_err_t gpu_f2fs_init(gpu_f2fs_t     *fs,
                              beaver_cache_t *cow_cache,
                              beaver_cache_t *data_cache,
                              uint32_t        max_inodes,
                              uint32_t        use_cow)
{
    if (!fs || max_inodes == 0 || !data_cache)
        return GPU_F2FS_ERR_INIT;

    if (use_cow) {
        if (!cow_cache || cow_cache->max_holders < max_inodes) {
            fprintf(stderr,
                    "gpu_f2fs_init: COW mode requires cow_cache with "
                    "max_holders >= max_inodes (%u)\n", max_inodes);
            return GPU_F2FS_ERR_INIT;
        }
    }

    memset(fs, 0, sizeof(*fs));
    fs->cow_cache       = cow_cache;
    fs->data_cache      = data_cache;
    fs->max_inodes      = max_inodes;
    fs->name_table_size = max_inodes * 2;
    fs->use_cow         = use_cow;
    fs->inode_cursor    = 0;

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

    /* Checkpoint mode: allocate a flat PM inode area (no Beaver COW) */
    if (!use_cow) {
        size_t inode_pm_sz = (size_t)max_inodes * BEAVER_PAGE_SIZE;
        if (gpm_alloc(&fs->pm_inode_region, inode_pm_sz, "f2fs_inode") != GPM_SUCCESS) {
            fprintf(stderr,
                    "gpu_f2fs_init: gpm_alloc(inode PM, %zu MiB) failed\n",
                    inode_pm_sz >> 20);
            goto err_pminode;
        }
        fs->pm_inode_base = fs->pm_inode_region.dev_addr;
    }

    /* PM write-ahead log */
    if (beaver_log_init(&fs->log, BEAVER_LOG_CAP_DEFAULT) != 0) {
        fprintf(stderr, "gpu_f2fs_init: beaver_log_init failed\n");
        goto err_log;
    }

    {
        uint32_t cover   = fs->name_table_size > max_inodes
                           ? fs->name_table_size : max_inodes;
        uint32_t threads = 256;
        uint32_t blocks  = (cover + threads - 1) / threads;

        gpu_f2fs_init_kernel<<<blocks, threads>>>(
                fs->inode_shadow, fs->dirty_flags, fs->name_table,
                max_inodes, fs->name_table_size);

        if (use_cow)
            beaver_f2fs_init_holders(cow_cache, max_inodes);

        cu = cudaDeviceSynchronize();
        if (cu != cudaSuccess) {
            fprintf(stderr, "gpu_f2fs_init: init kernel failed: %s\n",
                    cudaGetErrorString(cu));
            goto err_kernel;
        }
    }

    fs->is_initialized = 1;
    if (getenv("VERBOSE"))
        printf("gpu_f2fs_init: %u inodes  data_cache=%u holders  mode=%s\n"
               "  inode_shadow=%p  log_entries=%p\n",
               max_inodes, data_cache->max_holders,
               use_cow ? "COW" : "Checkpoint",
               (void *)fs->inode_shadow, (void *)fs->log.entries);
    return GPU_F2FS_OK;

err_kernel:
    beaver_log_cleanup(&fs->log);
err_log:
    if (fs->pm_inode_base) gpm_free(&fs->pm_inode_region);
err_pminode:
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

    beaver_log_cleanup(&fs->log);
    if (fs->pm_inode_base) gpm_free(&fs->pm_inode_region);
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

void gpu_vfs_mount_f2fs(gpu_vfs_t *vfs, gpu_f2fs_t *f2fs)
{
    vfs->fs   = f2fs;
    vfs->type = GPU_FS_F2FS_PM;
}
