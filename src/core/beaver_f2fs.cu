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
 *   beaver_data_read          (__device__) — lock-free read of a data page from PM
 *   beaver_f2fs_init_holders  (host)       — pre-allocate one inode holder per nid
 *
 * Write path (both inode and data):
 *   gpm_memcpy_nodrain → pm slot          (volatile stores, no drain yet)
 *   beaver_log_write   → PM log entry     (volatile stores, no drain yet)
 *   beaver_holder_flip                    (cur update + __threadfence_system() drain
 *                                          + read_ptr publish)
 *   Total: exactly 1 drain per write.
 */

#include "gpu_f2fs.h"
#include "beaver_cow.h"
#include <cuda_runtime.h>

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
 * beaver_data_read: lock-free read of one 4 KiB data page.
 *
 * holder_idx: the value stored in inode.i_addr[pgoff] by a prior write.
 * Returns 0 on success, -1 if the page has not been written.
 *
 * Uses volatile loads to bypass GPU L2 cache (DDIO disabled).
 */
__device__ int beaver_data_read(
        gpu_f2fs_t *fs, uint32_t holder_idx, void *dst)
{
    if (holder_idx >= fs->data_cache->max_holders)
        return -1;

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
