#ifndef BEAVER_COW_H
#define BEAVER_COW_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "gpm_interface.cuh"

/* Page size — matches shadowfs and F2FS */
#define BEAVER_PAGE_SIZE  4096u
#define BEAVER_PAGE_SHIFT 12u

/* ================================================================== */
/* PM Write-Ahead Log                                                  */
/* ================================================================== */

#define BEAVER_LOG_MAGIC       0xBEA71060u
#define BEAVER_LOG_CAP_DEFAULT 4096u   /* entries: 4096 × 32B = 128 KiB */

typedef enum {
    BLOG_INVALID    = 0,
    BLOG_DATA_FLIP  = 1,   /* data page written: (nid, pgoff, new_slot) */
    BLOG_INODE_FLIP = 2,   /* inode written:     (nid, UINT32_MAX, new_slot) */
} beaver_log_type_t;

/*
 * beaver_log_entry_t — 32 bytes, PM-resident.
 *
 * magic is written LAST (commit marker).  The drain that follows
 * (inside beaver_holder_flip) makes both the data write and this log
 * entry durable in a single __threadfence_system().
 */
typedef struct {
    uint32_t magic;     /* BEAVER_LOG_MAGIC when committed           */
    uint32_t type;      /* beaver_log_type_t                         */
    uint32_t nid;       /* inode number                              */
    uint32_t pgoff;     /* page offset; UINT32_MAX for inode entries */
    uint32_t slot;      /* new active slot index after flip (0 or 1) */
    uint32_t _pad;
    uint64_t seq;       /* monotonic sequence (recovery ordering)    */
} beaver_log_entry_t;

#ifdef __cplusplus
static_assert(sizeof(beaver_log_entry_t) == 32,
              "beaver_log_entry_t must be exactly 32 bytes");
#endif

/*
 * beaver_log_t — PM-backed circular log manager.
 *
 * Lives embedded in gpu_f2fs_t (cudaMallocManaged), so all fields are
 * accessible from both host and GPU kernels.
 *
 * head and log_seq are DRAM counters (atomicAdd from GPU).
 * entries is a UVA pointer to the PM circular buffer.
 *
 * Crash recovery (host, future): scan all capacity entries for valid
 * magic; for each (nid, pgoff) take the entry with highest seq as the
 * authoritative holder state.
 */
typedef struct {
    uint32_t            head;       /* circular buffer head (DRAM, atomicAdd) */
    uint32_t            capacity;   /* number of entries                      */
    uint64_t            log_seq;    /* monotonic seq counter (DRAM, atomicAdd)*/
    beaver_log_entry_t *entries;    /* UVA pointer to PM entry array          */
    gpm_region_t        pm_region;  /* PM allocation for entries only         */
} beaver_log_t;

/*
 * beaver_data_page_id: encode (nid, pgoff) as a 64-bit holder page_id.
 * nid occupies bits [63:20], pgoff occupies bits [19:0].
 * Supports nid < 2^44 and pgoff < 2^20 (> F2FS_ADDRS_PER_INODE = 1017).
 */
static __device__ __host__ __forceinline__ uint64_t
beaver_data_page_id(uint32_t nid, uint32_t pgoff)
{
    return ((uint64_t)nid << 20) | (pgoff & 0xFFFFFu);
}

/*
 * beaver_log_write: append a log entry to the PM circular buffer.
 *
 * Issues NO drain — the drain comes from the beaver_holder_flip() that
 * immediately follows this call, so data write + log entry + read_ptr
 * update are all made durable by one __threadfence_system().
 *
 * Writes magic LAST so that a partial PM write (before the flush)
 * will not present a spurious committed entry to the recovery scanner.
 */
static __device__ __forceinline__ void
beaver_log_write(beaver_log_t *log, uint32_t type,
                 uint32_t nid, uint32_t pgoff, uint32_t slot)
{
    uint64_t seq = atomicAdd((unsigned long long *)&log->log_seq, 1ULL);
    uint32_t idx = atomicAdd(&log->head, 1u) % log->capacity;
    beaver_log_entry_t *e = &log->entries[idx];

    volatile uint32_t *p = (volatile uint32_t *)e;
    p[1] = type;
    p[2] = nid;
    p[3] = pgoff;
    p[4] = slot;
    p[5] = 0u;
    *((volatile uint64_t *)e + 3) = seq;  /* bytes 24-31 = seq field */
    /* magic written last: marks entry as committed in PM */
    p[0] = BEAVER_LOG_MAGIC;
}

/* Host API — implemented in beaver_cow.cu */
__host__ int  beaver_log_init   (beaver_log_t *log, uint32_t capacity);
__host__ void beaver_log_cleanup(beaver_log_t *log);

/* ------------------------------------------------------------------ */
/* Holder state                                                        */
/* ------------------------------------------------------------------ */
typedef enum {
    HOLDER_INIT    = 0,   /* written to PM, not yet synced to SSD */
    HOLDER_SYNCING = 1,   /* submitted to background SSD sync     */
    HOLDER_FREEING = 2,   /* being freed                          */
} beaver_holder_state_t;

/* ------------------------------------------------------------------ */
/* beaver_holder_t                                                     */
/*                                                                     */
/* GPU port of struct shadow_page_holder (shadowfs/shadow_entry.h).   */
/*                                                                     */
/* 2-slot COW:                                                         */
/*   pm_addrs[0], pm_addrs[1]  — the two alternating COW pages        */
/*   pm_addrs[2]               — partial-page log for sub-page writes */
/*                                                                     */
/* Write path:                                                         */
/*   next = (cur < 0) ? 0 : (cur + 1) % 2                            */
/*   write user data into pm_addrs[next]                              */
/*   beaver_holder_flip(holder) ← atomic: cur=next, read_ptr=pm[next]*/
/*                                                                     */
/* Read path — lock-free:                                             */
/*   addr = beaver_holder_get_read(holder)  ← volatile load          */
/*                                                                     */
/* cur == -1  →  holder is empty (no write committed yet)             */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t         gpu_lock;     /* GPU spinlock (atomicCAS 0→1)      */
    volatile int     cur;          /* active slot: -1=empty, 0, or 1    */
    unsigned int     state;        /* beaver_holder_state_t             */
    uint32_t         _pad;         /* keep read_ptr 8-byte aligned      */
    void * volatile  read_ptr;     /* atomic read pointer (GPU RCU)     */
    void            *pm_addrs[3]; /* [0][1]=COW slots, [2]=pp-log      */
    uint64_t         page_id;      /* file-page index this holder owns  */
} beaver_holder_t;

/* page_id sentinel: slot is unallocated */
#define HOLDER_PAGE_ID_NONE UINT64_MAX

/* ------------------------------------------------------------------ */
/* beaver_cache_t                                                      */
/*                                                                     */
/* holders and hash_table are in cudaMalloc device memory — purely    */
/* GPU-resident, the CPU never touches them after init.               */
/*                                                                     */
/* The cache struct itself is small (~80 bytes) and lives in          */
/* cudaMallocManaged so GPU kernels can read the pointer values and   */
/* alloc_cursor without any cudaMemcpy.                               */
/*                                                                     */
/* PM slab layout — one gpm_alloc'd region, split into 3 pages/holder:*/
/*   holders[i].pm_addrs[0] = pm_base + i*3*PAGE_SIZE                 */
/*   holders[i].pm_addrs[1] = pm_base + (i*3+1)*PAGE_SIZE             */
/*   holders[i].pm_addrs[2] = pm_base + (i*3+2)*PAGE_SIZE             */
/* pm_addrs are wired by the init kernel; read_ptr starts NULL.       */
/* ------------------------------------------------------------------ */
typedef struct {
    beaver_holder_t     *holders;    /* cudaMalloc — pure GPU device memory */
    uint32_t             max_holders;
    uint32_t            *hash_table; /* cudaMalloc — pure GPU device memory */
    uint32_t             hash_size;
    uint32_t             alloc_cursor; /* atomic bump; GPU uses atomicAdd   */
    uint32_t             is_initialized;
    /* PM slab */
    gpm_region_t         pm_region;
    void                *pm_base;    /* UVA pointer to PM slab             */
} beaver_cache_t;

/* Error codes */
typedef enum {
    BEAVER_SUCCESS               =  0,
    BEAVER_ERROR_NOT_INITIALIZED = -1,
    BEAVER_ERROR_OUT_OF_MEMORY   = -2,
    BEAVER_ERROR_PAGE_NOT_FOUND  = -3,
    BEAVER_ERROR_PM_ERROR        = -4,
} beaver_error_t;

/* ------------------------------------------------------------------ */
/* Host API — init / cleanup only                                      */
/* ------------------------------------------------------------------ */

/*
 * beaver_cache_init: allocate device memory and PM slab; launch an init
 * kernel to wire pm_addrs and zero the hash table entirely on the GPU.
 * After this call, cache may be passed to CUDA kernels; the CPU does not
 * touch holders or hash_table again.
 */
beaver_error_t beaver_cache_init(beaver_cache_t *cache, uint32_t max_holders);

/*
 * beaver_cache_cleanup: release PM region and device memory.
 */
beaver_error_t beaver_cache_cleanup(beaver_cache_t *cache);

/* ------------------------------------------------------------------ */
/* Device-side COW primitives — static __device__ inline              */
/* ------------------------------------------------------------------ */

/* GPU spinlock (replaces kernel spinlock_t) */
static __device__ __forceinline__ void
beaver_spin_lock(uint32_t *lock)
{
    while (atomicCAS(lock, 0u, 1u) != 0u)
        __nanosleep(32);
}

static __device__ __forceinline__ void
beaver_spin_unlock(uint32_t *lock)
{
    /* Release fence: all preceding stores visible before lock drops */
    __threadfence();
    atomicExch(lock, 0u);
}

/*
 * beaver_holder_get_read: lock-free read of the current readable page.
 * Mirrors rcu_dereference(holder->read_ptr).
 * Returns NULL if no write has been committed yet (cur == -1).
 */
static __device__ __forceinline__ void *
beaver_holder_get_read(beaver_holder_t *holder)
{
    return *((void * volatile *)&holder->read_ptr);
}

/*
 * beaver_holder_write_addr: address of the INACTIVE slot — where the next
 * write should go before calling beaver_holder_flip.
 * Mirrors shadowfs holder_next_addr().
 */
static __device__ __forceinline__ void *
beaver_holder_write_addr(beaver_holder_t *holder)
{
    int next = (holder->cur < 0) ? 0 : (holder->cur + 1) % 2;
    return holder->pm_addrs[next];
}

/*
 * beaver_holder_flip: publish the newly written PM slot.
 *
 * Updates cur and read_ptr with a __threadfence_system() between them.
 * The fence orders all preceding stores (PM data written before this call)
 * before the read_ptr update, so any reader that loads the new read_ptr
 * is guaranteed to observe the complete page.
 *
 * After this call the data is already durable in PM — the fence doubles as
 * a PM persistence barrier.  beaver_holder_commit is therefore NOT needed and
 * must NOT be called after beaver_holder_flip (it would add a redundant drain).
 *
 * Call site pattern:
 *   gpm_memcpy_nodrain(beaver_holder_write_addr(h), src, PAGE_SIZE);
 *   beaver_holder_flip(h);      ← single drain + publish
 *   // done — no beaver_holder_commit
 *
 * Or use beaver_holder_write_and_flip() which encapsulates the above.
 */
static __device__ __forceinline__ void
beaver_holder_flip(beaver_holder_t *holder)
{
    int next = (holder->cur < 0) ? 0 : (holder->cur + 1) % 2;
    holder->cur = next;
    __threadfence_system();   /* single drain: PM data ordered before read_ptr */
    *((void * volatile *)&holder->read_ptr) = holder->pm_addrs[next];
}

/*
 * beaver_holder_stage: write src to the inactive PM slot and update cur,
 * but issue NO fence and NO read_ptr update.
 *
 * Use when batching multiple writes before a single __threadfence_system().
 * After staging all writes, call __threadfence_system() once, then call
 * beaver_holder_publish() on each staged holder to make data visible to readers.
 *
 * Caller must hold h->gpu_lock.
 */
static __device__ __forceinline__ void
beaver_holder_stage(beaver_holder_t *holder, const void *src)
{
    void *waddr = beaver_holder_write_addr(holder);
    gpm_memcpy_nodrain(waddr, src, BEAVER_PAGE_SIZE);
    int next = (holder->cur < 0) ? 0 : (holder->cur + 1) % 2;
    holder->cur = next;   /* update internal slot state; no fence yet */
}

/*
 * beaver_holder_publish: publish the staged write to readers.
 *
 * Must be called AFTER __threadfence_system() to ensure the PM write
 * is durable before readers see the new read_ptr.
 * Caller must hold h->gpu_lock, OR ensure no concurrent writers exist.
 */
static __device__ __forceinline__ void
beaver_holder_publish(beaver_holder_t *holder)
{
    if (holder->cur >= 0)
        *((void * volatile *)&holder->read_ptr) = holder->pm_addrs[holder->cur];
}

/*
 * beaver_holder_commit: DEPRECATED — do not call after beaver_holder_flip.
 *
 * beaver_holder_flip already issues __threadfence_system(), which is the only
 * drain required for PM durability and RCU ordering.  Calling commit after
 * flip issues a second gpm_drain() that is entirely redundant.
 *
 * This function is retained only for the standalone partial-page-log path
 * (pm_addrs[2]) where flip is not called.  For normal COW page replacement
 * use beaver_holder_write_and_flip() instead of the old three-call sequence.
 */
static __device__ __forceinline__ void
beaver_holder_commit(beaver_holder_t *holder)
{
    int slot = (holder->cur < 0) ? 0 : holder->cur;
    gpm_persist(holder->pm_addrs[slot], BEAVER_PAGE_SIZE);
}

/*
 * beaver_holder_write_and_flip: canonical single-drain COW page replacement.
 *
 * Writes src into the inactive PM slot and atomically publishes it with
 * exactly one __threadfence_system().  This replaces the old triple-drain
 * pattern:
 *
 *   OLD (3 drains):  gpm_memcpy + beaver_holder_flip + beaver_holder_commit
 *   NEW (1 drain):   beaver_holder_write_and_flip
 *
 * Drain breakdown:
 *   gpm_memcpy_nodrain  — volatile stores to PM, no fence yet
 *   [inside beaver_holder_flip]:
 *     h->cur = next
 *     __threadfence_system()   ← the ONE required drain
 *     *read_ptr = pm_addrs[next]
 *
 * Caller must hold h->gpu_lock.
 */
static __device__ __forceinline__ void
beaver_holder_write_and_flip(beaver_holder_t *holder, const void *src)
{
    void *waddr = beaver_holder_write_addr(holder);
    gpm_memcpy_nodrain(waddr, src, BEAVER_PAGE_SIZE);
    beaver_holder_flip(holder);   /* contains the single __threadfence_system() */
}

/* ------------------------------------------------------------------ */
/* Utility + device-side cache operations                              */
/* ------------------------------------------------------------------ */

static __host__ __device__ __forceinline__
uint32_t beaver_hash_page_id(uint64_t page_id, uint32_t hash_size)
{
    uint64_t h = page_id ^ (page_id >> 32);
    h ^= h >> 16;
    h *= 0x45d9f3bULL;
    h ^= h >> 16;
    return (uint32_t)(h % hash_size);
}

/*
 * beaver_find_holder: lock-free lookup of holder for page_id.
 * Linear probe over the device-resident hash table.
 * Returns NULL if not found.
 */
static __device__ __forceinline__ beaver_holder_t *
beaver_find_holder(const beaver_cache_t *cache, uint64_t page_id)
{
    uint32_t slot = beaver_hash_page_id(page_id, cache->hash_size);
    for (uint32_t i = 0; i < cache->hash_size; ++i) {
        uint32_t idx = cache->hash_table[slot];
        if (idx == 0xFFFFFFFFu)
            return NULL;
        if (cache->holders[idx].page_id == page_id)
            return &cache->holders[idx];
        slot = (slot + 1) % cache->hash_size;
    }
    return NULL;
}

/*
 * beaver_hash_insert: atomic linear-probe insert into the device hash table.
 * Uses atomicCAS to claim an empty slot; safe for concurrent GPU callers.
 * Returns true on success, false if the table is full.
 */
static __device__ __forceinline__ bool
beaver_hash_insert(uint32_t *hash_table, uint32_t hash_size,
                uint64_t page_id, uint32_t holder_idx)
{
    uint32_t slot = beaver_hash_page_id(page_id, hash_size);
    for (uint32_t i = 0; i < hash_size; ++i) {
        uint32_t old = atomicCAS(&hash_table[slot], 0xFFFFFFFFu, holder_idx);
        if (old == 0xFFFFFFFFu)
            return true;
        slot = (slot + 1) % hash_size;
    }
    return false;
}

/*
 * beaver_holder_alloc: GPU-side holder allocation.
 * Atomically claims the next free holder via alloc_cursor bump, initialises
 * it, and inserts its index into the hash table.
 * Safe for concurrent calls from multiple GPU threads.
 * Returns NULL if the holder pool is exhausted.
 */
static __device__ __forceinline__ beaver_holder_t *
beaver_holder_alloc(beaver_cache_t *cache, uint64_t page_id)
{
    uint32_t idx = atomicAdd(&cache->alloc_cursor, 1u);
    if (idx >= cache->max_holders) {
        atomicSub(&cache->alloc_cursor, 1u);
        return NULL;
    }

    beaver_holder_t *h = &cache->holders[idx];
    h->gpu_lock  = 0;
    h->cur       = -1;
    h->state     = HOLDER_INIT;
    h->_pad      = 0;
    h->read_ptr  = NULL;
    h->page_id   = page_id;
    /* pm_addrs already wired by beaver_cache_init_kernel */

    beaver_hash_insert(cache->hash_table, cache->hash_size, page_id, idx);
    return h;
}

#endif /* BEAVER_COW_H */
