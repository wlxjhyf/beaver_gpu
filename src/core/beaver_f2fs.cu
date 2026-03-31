/*
 * beaver_f2fs.cu — Beaver COW operations on GPU F2FS metadata and data.
 *
 * This is the only file that imports both beaver_cow.h and gpu_f2fs.h
 * together.  It keeps Beaver mechanics fully separate from the pure F2FS
 * file-system operations in gpu_f2fs.cu.
 *
 * Contents:
 *   beaver_inode_persist      (__device__) — COW-persist one F2FS inode to PM
 *   beaver_data_write         (__device__) — COW-persist one data page to PM
 *   beaver_data_read          (__device__) — read from DRAM cache if available, else PM
 *   beaver_f2fs_init_holders  (host)       — pre-allocate one inode holder per nid
 *   gpu_f2fs_prefetch         (host)       — prefetch file pages from PM to pinned DRAM
 *
 * Write path (both inode and data):
 *   gpm_memcpy_nodrain → pm slot          (volatile stores, no drain yet)
 *   beaver_log_write   → PM log entry     (volatile stores, no drain yet)
 *   beaver_holder_flip                    (cur update + __threadfence_system() drain
 *                                          + read_ptr publish)
 *   DRAM invalidation: dram_dev_ptrs[idx] = NULL (managed, GPU-side)
 *   Total: exactly 1 drain per write.
 *
 * Read path:
 *   1. dram_dev_ptrs[holder_idx] != NULL → GPU thread load from pinned DRAM (PCIe)
 *   2. fallback → GPU thread load from PM (existing volatile-load path)
 *
 * DRAM coherency: dram_dev_ptrs is cudaMallocManaged. CPU writes it during
 *   prefetch (gap period, no GPU kernel running); GPU reads it during compute.
 *   Separated by CUDA stream synchronization at the application level.
 *   GPU write path invalidates the entry atomically so stale DRAM data is
 *   never served after a new COW flip.
 */

#include "gpu_f2fs.h"
#include "beaver_cow.h"
#include <cuda_runtime.h>
#include <string.h>  /* memcpy for host prefetch */
#include <stdio.h>

/* ================================================================== */
/* Device: inode COW persist                                           */
/* ================================================================== */

/*
 * beaver_inode_persist: copy inode_shadow[nid].inode to PM via Beaver COW.
 *
 * Pattern: gpm_memcpy_nodrain + log_write + holder_flip (1 drain total).
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void beaver_inode_persist(gpu_f2fs_t *fs, uint32_t nid)
{
    beaver_holder_t *h = beaver_find_holder(fs->cow_cache, (uint64_t)nid);
    if (!h) return;

    beaver_spin_lock(&h->gpu_lock);

    void *waddr = beaver_holder_write_addr(h);
    gpm_memcpy_nodrain(waddr, &fs->inode_shadow[nid].inode, BEAVER_PAGE_SIZE);

    int next = (h->cur < 0) ? 0 : (h->cur + 1) % 2;
    beaver_log_write(&fs->log, BLOG_INODE_FLIP, nid, (uint32_t)-1, (uint32_t)next);

    beaver_holder_flip(h);   /* __threadfence_system(): drains data + log */

    beaver_spin_unlock(&h->gpu_lock);
}

/* ================================================================== */
/* Device: data page COW write                                         */
/* ================================================================== */

/*
 * beaver_data_write: COW write of one 4 KiB data page.
 *
 * existing_holder_idx: value currently in inode.i_addr[pgoff].
 *   F2FS_NULL_ADDR (0xFFFFFFFF) → first write to this page; a new holder
 *     is allocated from data_cache (atomicAdd bump).
 *   < max_holders → re-use the existing holder; COW flips to the other slot.
 *
 * Returns the holder index to be stored in inode.i_addr[pgoff], or
 * F2FS_NULL_ADDR if data_cache is full.
 *
 * Caller must hold inode_shadow[nid].lock (serialises i_addr updates).
 * The holder's own gpu_lock is acquired inside this function.
 */
__device__ uint32_t beaver_data_write(
        gpu_f2fs_t *fs, uint32_t nid, uint32_t pgoff,
        uint32_t existing_holder_idx, const void *src)
{
    beaver_holder_t *h;
    uint32_t idx;

    if (existing_holder_idx != (uint32_t)F2FS_NULL_ADDR) {
        /* Re-use existing holder: COW to the inactive slot */
        idx = existing_holder_idx;
        h   = &fs->data_cache->holders[idx];
    } else {
        /* First write to this page: claim a fresh holder slot */
        idx = atomicAdd(&fs->data_cache->alloc_cursor, 1u);
        if (idx >= fs->data_cache->max_holders) {
            atomicSub(&fs->data_cache->alloc_cursor, 1u);
            return (uint32_t)F2FS_NULL_ADDR;   /* data_cache full */
        }
        h = &fs->data_cache->holders[idx];
        /* pm_addrs already wired by beaver_cache_init_kernel */
        h->gpu_lock = 0;
        h->cur      = -1;
        h->state    = HOLDER_INIT;
        h->_pad     = 0;
        h->read_ptr = NULL;
        h->page_id  = beaver_data_page_id(nid, pgoff);
        beaver_hash_insert(fs->data_cache->hash_table,
                           fs->data_cache->hash_size,
                           h->page_id, idx);
    }

    beaver_spin_lock(&h->gpu_lock);

    /* Invalidate DRAM cache before COW flip: stale copy must not be served */
    if (fs->data_cache->dram_dev_ptrs)
        fs->data_cache->dram_dev_ptrs[idx] = NULL;

    void *waddr = beaver_holder_write_addr(h);
    gpm_memcpy_nodrain(waddr, src, BEAVER_PAGE_SIZE);

    int next = (h->cur < 0) ? 0 : (h->cur + 1) % 2;
    beaver_log_write(&fs->log, BLOG_DATA_FLIP, nid, pgoff, (uint32_t)next);

    beaver_holder_flip(h);   /* __threadfence_system(): drains data + log */

    beaver_spin_unlock(&h->gpu_lock);
    return idx;
}

/* ================================================================== */
/* Device: staged writes (no fence, no read_ptr publish)              */
/* ================================================================== */

/*
 * beaver_data_write_stage: COW write of one data page WITHOUT fence.
 *
 * Same allocation / holder-lookup logic as beaver_data_write, but uses
 * beaver_holder_stage() instead of beaver_holder_flip(), so no
 * __threadfence_system() is issued.  The caller is responsible for
 * issuing ONE __threadfence_system() after all staged writes, then
 * calling beaver_holder_publish() on every staged holder.
 *
 * Returns holder_idx (store into i_addr[pgoff]), or F2FS_NULL_ADDR on full.
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ uint32_t beaver_data_write_stage(
        gpu_f2fs_t *fs, uint32_t nid, uint32_t pgoff,
        uint32_t existing_holder_idx, const void *src)
{
    beaver_holder_t *h;
    uint32_t idx;

    if (existing_holder_idx != (uint32_t)F2FS_NULL_ADDR) {
        idx = existing_holder_idx;
        h   = &fs->data_cache->holders[idx];
    } else {
        idx = atomicAdd(&fs->data_cache->alloc_cursor, 1u);
        if (idx >= fs->data_cache->max_holders) {
            atomicSub(&fs->data_cache->alloc_cursor, 1u);
            return (uint32_t)F2FS_NULL_ADDR;
        }
        h = &fs->data_cache->holders[idx];
        h->gpu_lock = 0;
        h->cur      = -1;
        h->state    = HOLDER_INIT;
        h->_pad     = 0;
        h->read_ptr = NULL;
        h->page_id  = beaver_data_page_id(nid, pgoff);
        beaver_hash_insert(fs->data_cache->hash_table,
                           fs->data_cache->hash_size,
                           h->page_id, idx);
    }

    beaver_spin_lock(&h->gpu_lock);

    /* Invalidate DRAM cache before staging the new write */
    if (fs->data_cache->dram_dev_ptrs)
        fs->data_cache->dram_dev_ptrs[idx] = NULL;

    beaver_holder_stage(h, src);          /* write + cur update, no fence */
    int next = h->cur;
    beaver_log_write(&fs->log, BLOG_DATA_FLIP, nid, pgoff, (uint32_t)next);
    beaver_spin_unlock(&h->gpu_lock);
    return idx;
}

/*
 * beaver_inode_persist_stage: stage inode write to PM WITHOUT fence.
 *
 * Like beaver_inode_persist but uses beaver_holder_stage() so no
 * __threadfence_system() is issued.  Caller issues the fence separately
 * (in gpu_f2fs_fsync) then calls beaver_holder_publish() on the inode holder.
 *
 * Caller must hold inode_shadow[nid].lock.
 */
__device__ void beaver_inode_persist_stage(gpu_f2fs_t *fs, uint32_t nid)
{
    beaver_holder_t *h = beaver_find_holder(fs->cow_cache, (uint64_t)nid);
    if (!h) return;

    beaver_spin_lock(&h->gpu_lock);
    void *waddr = beaver_holder_write_addr(h);
    gpm_memcpy_nodrain(waddr, &fs->inode_shadow[nid].inode, BEAVER_PAGE_SIZE);
    int next = (h->cur < 0) ? 0 : (h->cur + 1) % 2;
    h->cur = next;
    beaver_log_write(&fs->log, BLOG_INODE_FLIP, nid, (uint32_t)-1, (uint32_t)next);
    beaver_spin_unlock(&h->gpu_lock);
}

/* ================================================================== */
/* Device: data page lock-free read                                    */
/* ================================================================== */

/*
 * beaver_data_read: read one 4 KiB data page, preferring the DRAM cache.
 *
 * holder_idx: the value stored in inode.i_addr[pgoff] by a prior write.
 * Returns 0 on success, -1 if the page has not been written.
 *
 * Read priority:
 *   1. DRAM cache hit (dram_dev_ptrs[holder_idx] != NULL):
 *      GPU thread loads from pinned CPU DRAM via PCIe.
 *      Faster than PM reads: lower DRAM latency even over PCIe.
 *   2. PM fallback: volatile loads from PM (DDIO disabled), existing path.
 */
__device__ int beaver_data_read(
        gpu_f2fs_t *fs, uint32_t holder_idx, void *dst)
{
    if (holder_idx >= fs->data_cache->max_holders)
        return -1;

    /* DRAM cache hit: GPU thread reads pinned DRAM via device pointer */
    void **dram_dev_ptrs = fs->data_cache->dram_dev_ptrs;
    if (dram_dev_ptrs) {
        void *dram_addr = dram_dev_ptrs[holder_idx];
        if (dram_addr) {
            unsigned long long *s = (unsigned long long *)dram_addr;
            unsigned long long *d = (unsigned long long *)dst;
            for (uint32_t i = 0; i < BEAVER_PAGE_SIZE / 8; ++i)
                d[i] = s[i];
            return 0;
        }
    }

    /* PM fallback: volatile loads bypass GPU L2 (DDIO disabled) */
    beaver_holder_t *h = &fs->data_cache->holders[holder_idx];
    void *addr = beaver_holder_get_read(h);
    if (!addr)
        return -1;

    const volatile unsigned long long *s =
        (const volatile unsigned long long *)addr;
    unsigned long long *d = (unsigned long long *)dst;
    for (uint32_t i = 0; i < BEAVER_PAGE_SIZE / 8; ++i)
        d[i] = s[i];

    return 0;
}

/* ================================================================== */
/* Inode holder initialisation (COW mode only)                        */
/* ================================================================== */

/*
 * beaver_f2fs_init_holders_kernel: one thread per inode; pre-allocates a
 * Beaver holder for each inode using page_id = nid (identity mapping).
 */
__global__ static void beaver_f2fs_init_holders_kernel(
        beaver_cache_t *cow_cache, uint32_t max_inodes)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < max_inodes)
        beaver_holder_alloc(cow_cache, (uint64_t)i);
}

/*
 * beaver_f2fs_init_holders: launch the holder-allocation kernel.
 * Called from gpu_f2fs_init when use_cow == 1, before cudaDeviceSynchronize.
 */
void beaver_f2fs_init_holders(beaver_cache_t *cow_cache, uint32_t max_inodes)
{
    uint32_t threads = 256;
    uint32_t blocks  = (max_inodes + threads - 1) / threads;
    beaver_f2fs_init_holders_kernel<<<blocks, threads>>>(cow_cache, max_inodes);
}

/* ================================================================== */
/* Host: DRAM prefetch                                                 */
/* ================================================================== */

/*
 * gpu_f2fs_prefetch: prefetch all written pages of fd from PM to pinned DRAM.
 *
 * Batch design: instead of one cudaMemcpy per holder (API call overhead
 * dominates at ~8 µs/call × N pages), we:
 *   1. Read the inode once to collect all valid holder indices.
 *   2. One cudaMemcpy of holders[min_hidx .. max_hidx] → host buffer.
 *   3. CPU-only loop: extract read_ptr + memcpy PM → pinned DRAM.
 *
 * Coherency: call after the GPU write kernel has completed and before the
 * GPU read kernel starts (CUDA stream sync at application level).
 * Reset the DRAM pool between iterations: beaver_dram_pool_reset().
 * IMPORTANT: requires beaver_dram_pool_init() to have been called first.
 */
void gpu_f2fs_prefetch(gpu_f2fs_t *fs, int fd)
{
    if (!fs || fd < 0 || (uint32_t)fd >= fs->max_inodes) return;
    uint32_t nid = (uint32_t)fd;

    beaver_cache_t *dc = fs->data_cache;
    if (!dc || !dc->dram_dev_ptrs || !dc->dram_pool_host) return;

    /* Step 1: one cudaMemcpy for the inode */
    f2fs_inode_t inode;
    cudaMemcpy(&inode, &fs->inode_shadow[nid].inode,
               sizeof(f2fs_inode_t), cudaMemcpyDeviceToHost);

    /* Scan i_addr[] to find the range of holder indices we need */
    uint32_t min_hidx = UINT32_MAX, max_hidx = 0;
    for (uint32_t pgoff = 0; pgoff < F2FS_ADDRS_PER_INODE; ++pgoff) {
        uint32_t hidx = inode.i_addr[pgoff];
        if (hidx == (uint32_t)F2FS_NULL_ADDR || hidx >= dc->max_holders) continue;
        if (dc->dram_dev_ptrs[hidx] != NULL) continue; /* already cached */
        if (hidx < min_hidx) min_hidx = hidx;
        if (hidx > max_hidx) max_hidx = hidx;
    }

    if (min_hidx > max_hidx) return; /* nothing to prefetch */

    /* Step 2: one cudaMemcpy for all needed holders [min_hidx .. max_hidx] */
    uint32_t range = max_hidx - min_hidx + 1;
    beaver_holder_t *holders_host =
        (beaver_holder_t *)malloc(range * sizeof(beaver_holder_t));
    if (!holders_host) {
        fprintf(stderr, "gpu_f2fs_prefetch: malloc failed\n");
        return;
    }
    cudaMemcpy(holders_host, &dc->holders[min_hidx],
               range * sizeof(beaver_holder_t), cudaMemcpyDeviceToHost);

    /* Step 3: CPU-only loop — no more cudaMemcpy per page */
    for (uint32_t pgoff = 0; pgoff < F2FS_ADDRS_PER_INODE; ++pgoff) {
        uint32_t hidx = inode.i_addr[pgoff];
        if (hidx == (uint32_t)F2FS_NULL_ADDR || hidx >= dc->max_holders) continue;
        if (dc->dram_dev_ptrs[hidx] != NULL) continue; /* already cached */

        void *pm_addr = holders_host[hidx - min_hidx].read_ptr;
        if (!pm_addr) continue; /* page not yet committed */

        uint32_t slot = __sync_fetch_and_add(&dc->dram_pool_cursor, 1u);
        if (slot >= dc->dram_pool_capacity) {
            fprintf(stderr, "gpu_f2fs_prefetch: DRAM pool exhausted "
                    "(fd=%d pgoff=%u); call beaver_dram_pool_reset()\n", fd, pgoff);
            break;
        }
        char *dram_page = dc->dram_pool_host + (size_t)slot * BEAVER_PAGE_SIZE;

        /* PM → pinned DRAM (pm_addr is UVA, host-accessible) */
        memcpy(dram_page, pm_addr, BEAVER_PAGE_SIZE);

        /* Publish GPU-accessible pointer */
        void *dev_ptr = NULL;
        cudaError_t cu = cudaHostGetDevicePointer(&dev_ptr, dram_page, 0);
        if (cu != cudaSuccess || !dev_ptr) {
            fprintf(stderr, "gpu_f2fs_prefetch: cudaHostGetDevicePointer failed: %s\n",
                    cudaGetErrorString(cu));
            __sync_fetch_and_add(&dc->dram_pool_cursor, (uint32_t)-1);
            continue;
        }
        dc->dram_dev_ptrs[hidx] = dev_ptr;
    }

    free(holders_host);
}
