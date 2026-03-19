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
        gpu_shadow_holder_t *holders,
        uint32_t            *hash_table,
        uint32_t             max_holders,
        uint32_t             hash_size,
        void                *pm_base)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < max_holders) {
        gpu_shadow_holder_t *h = &holders[i];
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
                    max_holders * sizeof(gpu_shadow_holder_t));
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

    cache->pm_base = cache->pm_region.addr;

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

    gpm_free(&cache->pm_region);
    cudaFree(cache->hash_table);
    cudaFree(cache->holders);
    memset(cache, 0, sizeof(*cache));

    return BEAVER_SUCCESS;
}
