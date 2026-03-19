#ifndef BEAVER_COW_H
#define BEAVER_COW_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "gpm_interface.cuh"

/* Page size — matches shadowfs and F2FS */
#define BEAVER_PAGE_SIZE  4096u
#define BEAVER_PAGE_SHIFT 12u

/* ------------------------------------------------------------------ */
/* Holder state — matches shadowfs enum holder_state exactly           */
/* ------------------------------------------------------------------ */
typedef enum {
    HOLDER_INIT    = 0,   /* written to PM, not yet synced to SSD */
    HOLDER_SYNCING = 1,   /* submitted to background SSD sync     */
    HOLDER_FREEING = 2,   /* being freed                          */
} gpu_holder_state_t;

/* ------------------------------------------------------------------ */
/* gpu_shadow_holder_t                                                 */
/*                                                                     */
/* GPU port of struct shadow_page_holder (shadowfs/shadow_entry.h).   */
/*                                                                     */
/* 2-slot COW (shadowfs uses pmem_pages[0] and [1]):                  */
/*   pm_addrs[0], pm_addrs[1]  — the two alternating COW pages        */
/*   pm_addrs[2]               — partial-page log for sub-page writes */
/*                                                                     */
/* Write path (mirrors shadowfs holder_flip_locked):                  */
/*   next = (cur < 0) ? 0 : (cur + 1) % 2                            */
/*   write user data into pm_addrs[next]                              */
/*   gpu_holder_flip(holder)   ← atomic: cur=next, read_ptr=pm[next] */
/*                                                                     */
/* Read path — lock-free (mirrors rcu_dereference):                   */
/*   addr = gpu_holder_get_read(holder)   ← volatile load            */
/*                                                                     */
/* cur == -1  →  holder is empty (no write committed yet)             */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t         gpu_lock;     /* GPU spinlock (atomicCAS 0→1)      */
    volatile int     cur;          /* active slot: -1=empty, 0, or 1    */
    unsigned int     state;        /* gpu_holder_state_t                */
    uint32_t         _pad;         /* keep read_ptr 8-byte aligned      */
    void * volatile  read_ptr;     /* atomic read pointer (GPU RCU)     */
    void            *pm_addrs[3]; /* [0][1]=COW slots, [2]=pp-log      */
    uint64_t         page_id;      /* file-page index this holder owns  */
} gpu_shadow_holder_t;

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
    gpu_shadow_holder_t *holders;    /* cudaMalloc — pure GPU device memory */
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
gpu_spin_lock(uint32_t *lock)
{
    while (atomicCAS(lock, 0u, 1u) != 0u)
        __nanosleep(32);
}

static __device__ __forceinline__ void
gpu_spin_unlock(uint32_t *lock)
{
    /* Release fence: all preceding stores visible before lock drops */
    __threadfence();
    atomicExch(lock, 0u);
}

/*
 * gpu_holder_get_read: lock-free read of the current readable page.
 * Mirrors rcu_dereference(holder->read_ptr).
 * Returns NULL if no write has been committed yet (cur == -1).
 */
static __device__ __forceinline__ void *
gpu_holder_get_read(gpu_shadow_holder_t *holder)
{
    return *((void * volatile *)&holder->read_ptr);
}

/*
 * gpu_holder_write_addr: address of the INACTIVE slot — where the next
 * write should go before calling gpu_holder_flip.
 * Mirrors shadowfs holder_next_addr().
 */
static __device__ __forceinline__ void *
gpu_holder_write_addr(gpu_shadow_holder_t *holder)
{
    int next = (holder->cur < 0) ? 0 : (holder->cur + 1) % 2;
    return holder->pm_addrs[next];
}

/*
 * gpu_holder_flip: publish the newly written PM slot.
 *
 * Updates cur and read_ptr with a __threadfence_system() between them.
 * The fence orders all preceding stores (PM data written before this call)
 * before the read_ptr update, so any reader that loads the new read_ptr
 * is guaranteed to observe the complete page.
 *
 * After this call the data is already durable in PM — the fence doubles as
 * a PM persistence barrier.  gpu_holder_commit is therefore NOT needed and
 * must NOT be called after gpu_holder_flip (it would add a redundant drain).
 *
 * Call site pattern:
 *   gpm_memcpy_nodrain(gpu_holder_write_addr(h), src, PAGE_SIZE);
 *   gpu_holder_flip(h);      ← single drain + publish
 *   // done — no gpu_holder_commit
 *
 * Or use gpu_holder_write_and_flip() which encapsulates the above.
 */
static __device__ __forceinline__ void
gpu_holder_flip(gpu_shadow_holder_t *holder)
{
    int next = (holder->cur < 0) ? 0 : (holder->cur + 1) % 2;
    holder->cur = next;
    __threadfence_system();   /* single drain: PM data ordered before read_ptr */
    *((void * volatile *)&holder->read_ptr) = holder->pm_addrs[next];
}

/*
 * gpu_holder_commit: DEPRECATED — do not call after gpu_holder_flip.
 *
 * gpu_holder_flip already issues __threadfence_system(), which is the only
 * drain required for PM durability and RCU ordering.  Calling commit after
 * flip issues a second gpm_drain() that is entirely redundant.
 *
 * This function is retained only for the standalone partial-page-log path
 * (pm_addrs[2]) where flip is not called.  For normal COW page replacement
 * use gpu_holder_write_and_flip() instead of the old three-call sequence.
 */
static __device__ __forceinline__ void
gpu_holder_commit(gpu_shadow_holder_t *holder)
{
    int slot = (holder->cur < 0) ? 0 : holder->cur;
    gpm_persist(holder->pm_addrs[slot], BEAVER_PAGE_SIZE);
}

/*
 * gpu_holder_write_and_flip: canonical single-drain COW page replacement.
 *
 * Writes src into the inactive PM slot and atomically publishes it with
 * exactly one __threadfence_system().  This replaces the old triple-drain
 * pattern:
 *
 *   OLD (3 drains):  gpm_memcpy + gpu_holder_flip + gpu_holder_commit
 *   NEW (1 drain):   gpu_holder_write_and_flip
 *
 * Drain breakdown:
 *   gpm_memcpy_nodrain  — volatile stores to PM, no fence yet
 *   [inside gpu_holder_flip]:
 *     h->cur = next
 *     __threadfence_system()   ← the ONE required drain
 *     *read_ptr = pm_addrs[next]
 *
 * Caller must hold h->gpu_lock.
 */
static __device__ __forceinline__ void
gpu_holder_write_and_flip(gpu_shadow_holder_t *holder, const void *src)
{
    void *waddr = gpu_holder_write_addr(holder);
    gpm_memcpy_nodrain(waddr, src, BEAVER_PAGE_SIZE);
    gpu_holder_flip(holder);   /* contains the single __threadfence_system() */
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
 * gpu_find_holder: lock-free lookup of holder for page_id.
 * Linear probe over the device-resident hash table.
 * Returns NULL if not found.
 */
static __device__ __forceinline__ gpu_shadow_holder_t *
gpu_find_holder(const beaver_cache_t *cache, uint64_t page_id)
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
 * gpu_hash_insert: atomic linear-probe insert into the device hash table.
 * Uses atomicCAS to claim an empty slot; safe for concurrent GPU callers.
 * Returns true on success, false if the table is full.
 */
static __device__ __forceinline__ bool
gpu_hash_insert(uint32_t *hash_table, uint32_t hash_size,
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
 * gpu_holder_alloc: GPU-side holder allocation.
 * Atomically claims the next free holder via alloc_cursor bump, initialises
 * it, and inserts its index into the hash table.
 * Safe for concurrent calls from multiple GPU threads.
 * Returns NULL if the holder pool is exhausted.
 */
static __device__ __forceinline__ gpu_shadow_holder_t *
gpu_holder_alloc(beaver_cache_t *cache, uint64_t page_id)
{
    uint32_t idx = atomicAdd(&cache->alloc_cursor, 1u);
    if (idx >= cache->max_holders) {
        atomicSub(&cache->alloc_cursor, 1u);
        return NULL;
    }

    gpu_shadow_holder_t *h = &cache->holders[idx];
    h->gpu_lock  = 0;
    h->cur       = -1;
    h->state     = HOLDER_INIT;
    h->_pad      = 0;
    h->read_ptr  = NULL;
    h->page_id   = page_id;
    /* pm_addrs already wired by beaver_cache_init_kernel */

    gpu_hash_insert(cache->hash_table, cache->hash_size, page_id, idx);
    return h;
}

#endif /* BEAVER_COW_H */
