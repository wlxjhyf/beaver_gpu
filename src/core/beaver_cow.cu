/*
 * beaver_cow.cu — Host-side cache init/cleanup.
 *
 * holders and hash_table live in cudaMalloc device memory; the CPU never
 * accesses them directly after beaver_cache_init returns.  A GPU init kernel
 * wires pm_addrs and zeros the hash table.  All runtime holder operations
 * (alloc, find, flip, commit) are __device__ functions in beaver_cow.h.
 */

#include "beaver_cow.h"
#include "gpm_interface.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Init kernel: runs entirely on GPU                                   */
/*                                                                     */
/* Each thread initialises one holder (if i < max_holders) AND one    */
/* hash table slot (if i < hash_size).  Launch with enough threads to */
/* cover whichever is larger.                                          */
/* ------------------------------------------------------------------ */
__global__ static void beaver_cache_init_kernel(
        beaver_holder_t *holders,
        uint32_t            *hash_table,
        uint32_t             max_holders,
        uint32_t             hash_size,
        void                *pm_base)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < max_holders) {
        beaver_holder_t *h = &holders[i];
        h->gpu_lock    = 0;
        h->cur         = -1;
        h->state       = HOLDER_INIT;
        h->_pad        = 0;
        h->read_ptr    = NULL;
        char *base     = (char *)pm_base + (size_t)i * 3 * BEAVER_PAGE_SIZE;
        h->pm_addrs[0] = base;
        h->pm_addrs[1] = base + BEAVER_PAGE_SIZE;
        h->pm_addrs[2] = base + 2 * BEAVER_PAGE_SIZE;
        h->page_id     = HOLDER_PAGE_ID_NONE;
    }

    if (i < hash_size)
        hash_table[i] = 0xFFFFFFFFu; /* HASH_EMPTY */
}

/* ------------------------------------------------------------------ */
/* beaver_cache_init                                                   */
/* ------------------------------------------------------------------ */
beaver_error_t beaver_cache_init(beaver_cache_t *cache, uint32_t max_holders)
{
    if (!cache || max_holders == 0)
        return BEAVER_ERROR_NOT_INITIALIZED;

    memset(cache, 0, sizeof(*cache));

    /* holders in pure device memory — GPU accesses at full bandwidth */
    cudaError_t cu;
    cu = cudaMalloc((void **)&cache->holders,
                    max_holders * sizeof(beaver_holder_t));
    if (cu != cudaSuccess) {
        fprintf(stderr, "beaver_cache_init: cudaMalloc(holders): %s\n",
                cudaGetErrorString(cu));
        return BEAVER_ERROR_OUT_OF_MEMORY;
    }

    /* hash table: ~2× holders for ~50% load factor */
    cache->hash_size = max_holders * 2;
    cu = cudaMalloc((void **)&cache->hash_table,
                    cache->hash_size * sizeof(uint32_t));
    if (cu != cudaSuccess) {
        fprintf(stderr, "beaver_cache_init: cudaMalloc(hash_table): %s\n",
                cudaGetErrorString(cu));
        cudaFree(cache->holders);
        return BEAVER_ERROR_OUT_OF_MEMORY;
    }

    /* Single PM slab: 3 pages per holder (slot0, slot1, pp-log) */
    gpm_error_t gerr = gpm_init();
    if (gerr != GPM_SUCCESS) {
        fprintf(stderr, "beaver_cache_init: gpm_init failed (%d)\n", gerr);
        cudaFree(cache->hash_table);
        cudaFree(cache->holders);
        return BEAVER_ERROR_PM_ERROR;
    }

    size_t slab_size = (size_t)max_holders * 3 * BEAVER_PAGE_SIZE;
    gerr = gpm_alloc(&cache->pm_region, slab_size, "cow");
    if (gerr != GPM_SUCCESS) {
        fprintf(stderr, "beaver_cache_init: gpm_alloc(%zu B) failed (%d)\n",
                slab_size, gerr);
        cudaFree(cache->hash_table);
        cudaFree(cache->holders);
        return BEAVER_ERROR_PM_ERROR;
    }

    /* Use dev_addr (GPU-accessible pointer) as pm_base so that
     * pm_addrs stored in holders are directly dereferenceable in device code. */
    cache->pm_base = cache->pm_region.dev_addr;

    /* Initialise holders and hash table entirely on the GPU */
    uint32_t cover   = cache->hash_size > max_holders
                       ? cache->hash_size : max_holders;
    uint32_t threads = 256;
    uint32_t blocks  = (cover + threads - 1) / threads;

    beaver_cache_init_kernel<<<blocks, threads>>>(
            cache->holders, cache->hash_table,
            max_holders, cache->hash_size, cache->pm_base);

    cu = cudaDeviceSynchronize();
    if (cu != cudaSuccess) {
        fprintf(stderr, "beaver_cache_init: init kernel failed: %s\n",
                cudaGetErrorString(cu));
        gpm_free(&cache->pm_region);
        cudaFree(cache->hash_table);
        cudaFree(cache->holders);
        return BEAVER_ERROR_PM_ERROR;
    }

    cache->max_holders    = max_holders;
    cache->alloc_cursor   = 0;
    cache->is_initialized = 1;

    if (getenv("VERBOSE"))
        printf("beaver_cache_init: %u holders, PM slab %zu MiB at %p "
               "(is_pmem=%d)  holders/hash_table in GPU device memory\n",
               max_holders, slab_size >> 20,
               cache->pm_base, cache->pm_region.is_pmem);

    return BEAVER_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* beaver_cache_cleanup                                                */
/* ------------------------------------------------------------------ */
beaver_error_t beaver_cache_cleanup(beaver_cache_t *cache)
{
    if (!cache || !cache->is_initialized)
        return BEAVER_ERROR_NOT_INITIALIZED;

    beaver_dram_pool_cleanup(cache);
    gpm_free(&cache->pm_region);
    cudaFree(cache->hash_table);
    cudaFree(cache->holders);
    memset(cache, 0, sizeof(*cache));

    return BEAVER_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* DRAM prefetch layer                                                 */
/* ------------------------------------------------------------------ */

beaver_error_t beaver_dram_pool_init(beaver_cache_t *cache, uint32_t pool_pages)
{
    if (!cache || !cache->is_initialized || pool_pages == 0)
        return BEAVER_ERROR_NOT_INITIALIZED;

    if (cache->dram_pool_host)   /* already initialised */
        return BEAVER_SUCCESS;

    cudaError_t cu;

    /* Per-holder device pointer table: managed so GPU can read, CPU can write */
    cu = cudaMallocManaged((void **)&cache->dram_dev_ptrs,
                           cache->max_holders * sizeof(void *));
    if (cu != cudaSuccess) {
        fprintf(stderr, "beaver_dram_pool_init: cudaMallocManaged(dram_dev_ptrs): %s\n",
                cudaGetErrorString(cu));
        return BEAVER_ERROR_OUT_OF_MEMORY;
    }
    cudaMemset(cache->dram_dev_ptrs, 0, cache->max_holders * sizeof(void *));

    /* Pinned+mapped DRAM page pool: accessible from both CPU and GPU */
    cu = cudaHostAlloc((void **)&cache->dram_pool_host,
                       (size_t)pool_pages * BEAVER_PAGE_SIZE,
                       cudaHostAllocMapped);
    if (cu != cudaSuccess) {
        fprintf(stderr, "beaver_dram_pool_init: cudaHostAlloc(%zu MiB): %s\n",
                (size_t)pool_pages * BEAVER_PAGE_SIZE >> 20, cudaGetErrorString(cu));
        cudaFree(cache->dram_dev_ptrs);
        cache->dram_dev_ptrs = NULL;
        return BEAVER_ERROR_OUT_OF_MEMORY;
    }

    cache->dram_pool_capacity = pool_pages;
    cache->dram_pool_cursor   = 0;

    if (getenv("VERBOSE"))
        printf("beaver_dram_pool_init: %u pages (%.1f MiB pinned DRAM)\n",
               pool_pages, (double)pool_pages * BEAVER_PAGE_SIZE / (1 << 20));

    return BEAVER_SUCCESS;
}

void beaver_dram_pool_reset(beaver_cache_t *cache)
{
    if (!cache || !cache->dram_pool_host) return;
    /* Clear all DRAM pointers so GPU falls back to PM until next prefetch */
    cudaMemset(cache->dram_dev_ptrs, 0, cache->max_holders * sizeof(void *));
    cache->dram_pool_cursor = 0;
}

void beaver_dram_pool_cleanup(beaver_cache_t *cache)
{
    if (!cache) return;
    if (cache->dram_pool_host) {
        cudaFreeHost(cache->dram_pool_host);
        cache->dram_pool_host = NULL;
    }
    if (cache->dram_dev_ptrs) {
        cudaFree(cache->dram_dev_ptrs);
        cache->dram_dev_ptrs = NULL;
    }
    cache->dram_pool_capacity = 0;
    cache->dram_pool_cursor   = 0;
}

/* ------------------------------------------------------------------ */
/* beaver_log_init                                                     */
/* ------------------------------------------------------------------ */

/*
 * beaver_log_init: allocate a PM circular buffer for the write-ahead log.
 *
 * The entry array is allocated via gpm_alloc (devdax, cudaHostRegister'd).
 * The head and log_seq counters stay in DRAM (managed/device memory),
 * since they do not need to survive a crash — recovery scans the PM
 * entries by seq to rebuild holder state.
 */
int beaver_log_init(beaver_log_t *log, uint32_t capacity)
{
    if (!log || capacity == 0) return -1;

    memset(log, 0, sizeof(*log));

    size_t entries_sz = (size_t)capacity * sizeof(beaver_log_entry_t);
    gpm_error_t gerr  = gpm_alloc(&log->pm_region, entries_sz, "beaver_log");
    if (gerr != GPM_SUCCESS) {
        fprintf(stderr, "beaver_log_init: gpm_alloc(%zu B) failed (%d)\n",
                entries_sz, gerr);
        return -1;
    }

    log->entries  = (beaver_log_entry_t *)log->pm_region.dev_addr;
    log->capacity = capacity;
    log->head     = 0;
    log->log_seq  = 0;

    /* Zero all entries on PM so recovery sees no stale magic values */
    pmem_memset_persist(log->entries, 0, entries_sz);

    if (getenv("VERBOSE"))
        printf("beaver_log_init: %u entries (%.1f KiB PM) at %p\n",
               capacity, (double)entries_sz / 1024.0, (void *)log->entries);
    return 0;
}

/* ------------------------------------------------------------------ */
/* beaver_log_cleanup                                                  */
/* ------------------------------------------------------------------ */
void beaver_log_cleanup(beaver_log_t *log)
{
    if (!log || !log->entries) return;
    gpm_free(&log->pm_region);
    memset(log, 0, sizeof(*log));
}
