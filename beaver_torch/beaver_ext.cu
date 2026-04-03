/*
 * beaver_ext.cu — PyTorch CUDA Extension for Beaver PM Offload (Glue Layer)
 *
 * This file is a PURE GLUE LAYER: it bridges PyTorch tensors to the
 * src/core/ Beaver COW infrastructure. All PM allocation, COW logic,
 * and DRAM prefetch are delegated to:
 *   - gpm_interface.cuh: PM allocation (gpm_init, gpm_alloc)
 *   - beaver_cow.h: COW holders (beaver_cache_init, beaver_holder_write_and_flip)
 *
 * Design (per e2e_experiment_plan.md):
 *   - BeaverManager.evict_dirty: GPU → PM (COW) + trigger async prefetch
 *   - BeaverManager.evict_readonly: only release GPU memory, no PM write
 *   - BeaverManager.restore: DRAM (prefetch hit) or PM (miss) → GPU
 *
 * Tensor → Holder mapping:
 *   Each tensor is split into 4KB pages. Each page is managed by one
 *   beaver_holder_t in data_cache. The mapping is:
 *     fd (file descriptor) → array of holder indices for that tensor's pages
 *   This avoids F2FS metadata overhead while reusing Beaver COW.
 *
 * Write path (evict_dirty):
 *   For each 4KB page of the tensor:
 *     1. GPU kernel: gpm_memcpy_nodrain(holder_write_addr, gpu_src, 4KB)
 *     2. GPU kernel: beaver_holder_flip (single __threadfence_system)
 *   Then trigger async prefetch: PM → pinned DRAM (CPU memcpy)
 *
 * Read path (restore):
 *   For each 4KB page:
 *     1. Check dram_dev_ptrs[holder_idx]: if non-NULL, cudaMemcpy from DRAM
 *     2. Else fallback to PM read (slower)
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/extension.h>

/* Include src/core headers */
#include "gpm_interface.cuh"
#include "beaver_cow.h"

/* DDIO control (from tests/ddio_helper.cpp) */
extern "C" {
    uint8_t ddio_get_gpu_bus(void);
    void    ddio_disable(uint8_t bus);
    void    ddio_enable (uint8_t bus);
}

#define PAGE_SIZE       4096u
#define CHUNK_PAGES     1017u   /* F2FS_ADDRS_PER_INODE limit per "file" */
#define CHUNK_BYTES     ((size_t)CHUNK_PAGES * PAGE_SIZE)   /* ~4.16 MiB */

/* ------------------------------------------------------------------ */
/* GPU kernel: COW write using beaver_holder_write_and_flip            */
/* ------------------------------------------------------------------ */

/*
 * beaver_cow_write_kernel: write tensor pages to PM via Beaver COW.
 *
 * Each thread block handles one 4KB page. Within the block, threads
 * cooperate to copy 4KB from GPU tensor to PM holder slot, then
 * thread 0 calls beaver_holder_flip to atomically publish.
 *
 * This kernel uses the Beaver COW protocol:
 *   1. gpm_memcpy_nodrain to inactive slot (volatile stores, no fence)
 *   2. beaver_holder_flip (cur update + __threadfence_system + read_ptr publish)
 *
 * Parameters:
 *   cache: beaver_cache_t with pre-allocated holders
 *   holder_indices: array of holder indices for this tensor's pages
 *   gpu_src: source GPU tensor data
 *   num_pages: number of 4KB pages to write
 *   last_page_bytes: bytes in the last page (may be < 4KB)
 */
__global__ void beaver_cow_write_kernel(
    beaver_cache_t *cache,
    uint32_t       *holder_indices,
    const char     *gpu_src,
    uint32_t        num_pages,
    uint32_t        last_page_bytes)
{
    uint32_t page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    uint32_t holder_idx = holder_indices[page_idx];
    beaver_holder_t *h = &cache->holders[holder_idx];

    /* Determine bytes to copy for this page */
    uint32_t bytes = (page_idx == num_pages - 1) ? last_page_bytes : PAGE_SIZE;

    /* Source address in GPU tensor */
    const char *src = gpu_src + (size_t)page_idx * PAGE_SIZE;

    /* Destination: inactive PM slot */
    void *waddr = beaver_holder_write_addr(h);

    /* Cooperative copy: all threads in block participate */
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;

    /* 8-byte aligned copy */
    uint32_t n_words = bytes / 8;
    volatile uint64_t *dst64 = (volatile uint64_t *)waddr;
    const uint64_t *src64 = (const uint64_t *)src;
    for (uint32_t i = tid; i < n_words; i += stride) {
        dst64[i] = src64[i];
    }

    /* Handle tail bytes (if any) */
    uint32_t tail_start = n_words * 8;
    if (tid == 0 && tail_start < bytes) {
        volatile char *dst_tail = (volatile char *)waddr + tail_start;
        const char *src_tail = src + tail_start;
        for (uint32_t i = tail_start; i < bytes; i++) {
            dst_tail[i - tail_start] = src_tail[i - tail_start];
        }
    }

    __syncthreads();

    /* Thread 0: flip the holder (single __threadfence_system) */
    if (tid == 0) {
        /* Invalidate DRAM cache before flip */
        if (cache->dram_dev_ptrs)
            cache->dram_dev_ptrs[holder_idx] = NULL;

        beaver_holder_flip(h);
    }
}

/*
 * beaver_cow_read_kernel: read tensor pages from DRAM cache or PM.
 *
 * Each thread block handles one 4KB page. Reads from DRAM if prefetched,
 * otherwise falls back to PM (volatile loads).
 */
__global__ void beaver_cow_read_kernel(
    beaver_cache_t *cache,
    uint32_t       *holder_indices,
    char           *gpu_dst,
    uint32_t        num_pages,
    uint32_t        last_page_bytes)
{
    uint32_t page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    uint32_t holder_idx = holder_indices[page_idx];
    beaver_holder_t *h = &cache->holders[holder_idx];

    uint32_t bytes = (page_idx == num_pages - 1) ? last_page_bytes : PAGE_SIZE;
    char *dst = gpu_dst + (size_t)page_idx * PAGE_SIZE;

    /* Check DRAM cache first */
    void *dram_addr = NULL;
    if (cache->dram_dev_ptrs)
        dram_addr = cache->dram_dev_ptrs[holder_idx];

    const char *src;
    if (dram_addr) {
        /* DRAM hit: read from pinned DRAM */
        src = (const char *)dram_addr;
    } else {
        /* PM fallback: read from holder's read_ptr */
        src = (const char *)beaver_holder_get_read(h);
        if (!src) return;  /* page not yet written */
    }

    /* Cooperative copy */
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;
    uint32_t n_words = bytes / 8;

    uint64_t *dst64 = (uint64_t *)dst;
    const uint64_t *src64 = (const uint64_t *)src;
    for (uint32_t i = tid; i < n_words; i += stride) {
        dst64[i] = src64[i];
    }

    /* Tail bytes */
    uint32_t tail_start = n_words * 8;
    if (tid == 0 && tail_start < bytes) {
        for (uint32_t i = tail_start; i < bytes; i++) {
            dst[i] = src[i];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extension state                                                      */
/* ------------------------------------------------------------------ */

struct BeaverExtState {
    /* Beaver COW cache for data pages */
    beaver_cache_t *data_cache;

    /* Per-fd metadata: holder indices for each tensor's pages */
    uint32_t **fd_holders;      /* fd_holders[fd] = device array of holder indices */
    uint32_t  *fd_num_pages;    /* fd_num_pages[fd] = number of pages for this fd */
    size_t    *fd_bytes;        /* fd_bytes[fd] = actual bytes written */
    int        max_fds;
    int        next_fd;

    /* DDIO state */
    uint8_t    gpu_bus;
    bool       ddio_disabled;

    bool       initialized;
};

static BeaverExtState g_state;

#define BEXT_CHECK_INIT() \
    do { if (!g_state.initialized) \
        throw std::runtime_error("beaver_ext: not initialized"); } while(0)

#define CUDA_CHECK(call) \
    do { cudaError_t _e = (call); \
         if (_e != cudaSuccess) { \
             char _buf[256]; \
             snprintf(_buf, sizeof(_buf), "CUDA error at %s:%d — %s", \
                      __FILE__, __LINE__, cudaGetErrorString(_e)); \
             throw std::runtime_error(_buf); \
         } } while(0)

/* ------------------------------------------------------------------ */
/* init                                                                 */
/* ------------------------------------------------------------------ */

void beaver_ext_init(int64_t max_inodes, int64_t max_data_pages,
                     int64_t dram_pool_pages)
{
    if (g_state.initialized)
        throw std::runtime_error("beaver_ext: already initialized");

    /* Disable DDIO: GPU PCIe writes bypass CPU L3 and reach PM media */
    g_state.gpu_bus = ddio_get_gpu_bus();
    ddio_disable(g_state.gpu_bus);
    g_state.ddio_disabled = true;
    printf("[beaver_ext] DDIO disabled (GPU bus 0x%02x)\n", g_state.gpu_bus);

    /* Initialize GPM (maps /dev/dax1.0) */
    gpm_error_t gerr = gpm_init();
    if (gerr != GPM_SUCCESS)
        throw std::runtime_error("beaver_ext: gpm_init failed");

    /* Allocate Beaver COW cache for data pages.
     * Use cudaMallocManaged so the cache struct is accessible from both
     * host (for init/cleanup) and device (for kernel access to holders). */
    cudaError_t cu = cudaMallocManaged((void **)&g_state.data_cache, sizeof(beaver_cache_t));
    if (cu != cudaSuccess || !g_state.data_cache)
        throw std::runtime_error("beaver_ext: cudaMallocManaged data_cache failed");

    beaver_error_t berr = beaver_cache_init(g_state.data_cache, (uint32_t)max_data_pages);
    if (berr != BEAVER_SUCCESS) {
        cudaFree(g_state.data_cache);
        throw std::runtime_error("beaver_ext: beaver_cache_init failed");
    }

    /* Initialize DRAM prefetch pool */
    berr = beaver_dram_pool_init(g_state.data_cache, (uint32_t)dram_pool_pages);
    if (berr != BEAVER_SUCCESS) {
        beaver_cache_cleanup(g_state.data_cache);
        cudaFree(g_state.data_cache);
        throw std::runtime_error("beaver_ext: beaver_dram_pool_init failed");
    }

    /* Per-fd metadata */
    g_state.max_fds = (int)max_inodes;
    g_state.next_fd = 0;
    g_state.fd_holders   = (uint32_t **)calloc(max_inodes, sizeof(uint32_t *));
    g_state.fd_num_pages = (uint32_t *)calloc(max_inodes, sizeof(uint32_t));
    g_state.fd_bytes     = (size_t *)calloc(max_inodes, sizeof(size_t));
    if (!g_state.fd_holders || !g_state.fd_num_pages || !g_state.fd_bytes)
        throw std::runtime_error("beaver_ext: calloc fd tables failed");

    g_state.initialized = true;

    printf("[beaver_ext] initialized (Beaver COW backend):\n"
           "  max_fds=%d  max_data_pages=%ld  dram_pool_pages=%ld\n"
           "  PM slab: %.2f GB  DRAM pool: %.2f GB\n",
           g_state.max_fds, (long)max_data_pages, (long)dram_pool_pages,
           (double)g_state.data_cache->pm_region.size / (1UL << 30),
           (double)dram_pool_pages * PAGE_SIZE / (1UL << 30));
}

/* ------------------------------------------------------------------ */
/* create_file: allocate fd and holder indices for a tensor            */
/* ------------------------------------------------------------------ */

int64_t beaver_ext_create_file(int64_t /*hash*/)
{
    BEXT_CHECK_INIT();
    if (g_state.next_fd >= g_state.max_fds)
        throw std::runtime_error("beaver_ext: max_fds exhausted");
    return (int64_t)(g_state.next_fd++);
}

/* ------------------------------------------------------------------ */
/* write — GPU tensor → PM via Beaver COW                              */
/* ------------------------------------------------------------------ */

void beaver_ext_write(int64_t fd, int64_t data_ptr, int64_t num_bytes)
{
    BEXT_CHECK_INIT();
    if (fd < 0 || fd >= g_state.max_fds)
        throw std::invalid_argument("beaver_ext: fd out of range");
    if (num_bytes <= 0)
        throw std::invalid_argument("beaver_ext: num_bytes must be > 0");
    if (data_ptr == 0)
        throw std::runtime_error("beaver_ext: data_ptr is NULL");

    /* Verify data_ptr is device memory */
    cudaPointerAttributes attrs;
    CUDA_CHECK(cudaPointerGetAttributes(&attrs, (void *)data_ptr));
    if (attrs.type != cudaMemoryTypeDevice)
        throw std::runtime_error("beaver_ext: data_ptr is not device memory");

    uint32_t num_pages = (uint32_t)((num_bytes + PAGE_SIZE - 1) / PAGE_SIZE);
    uint32_t last_page_bytes = (uint32_t)(num_bytes % PAGE_SIZE);
    if (last_page_bytes == 0) last_page_bytes = PAGE_SIZE;

    /* Allocate holder indices if first write to this fd */
    if (g_state.fd_holders[fd] == NULL) {
        /* Allocate device array for holder indices */
        uint32_t *d_holders;
        CUDA_CHECK(cudaMalloc(&d_holders, num_pages * sizeof(uint32_t)));

        /* Allocate holders from data_cache (host-side bump allocation) */
        uint32_t *h_holders = (uint32_t *)malloc(num_pages * sizeof(uint32_t));
        if (!h_holders) throw std::runtime_error("beaver_ext: malloc h_holders failed");

        uint32_t base_idx = g_state.data_cache->alloc_cursor;
        for (uint32_t i = 0; i < num_pages; i++) {
            uint32_t idx = base_idx + i;
            if (idx >= g_state.data_cache->max_holders) {
                free(h_holders);
                cudaFree(d_holders);
                throw std::runtime_error("beaver_ext: data_cache exhausted");
            }
            h_holders[i] = idx;
        }
        g_state.data_cache->alloc_cursor = base_idx + num_pages;

        /* Copy holder indices to device */
        CUDA_CHECK(cudaMemcpy(d_holders, h_holders, num_pages * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        free(h_holders);

        g_state.fd_holders[fd] = d_holders;
        g_state.fd_num_pages[fd] = num_pages;
    }

    /* Launch COW write kernel */
    int threads = 256;
    int blocks = (int)num_pages;

    beaver_cow_write_kernel<<<blocks, threads>>>(
        g_state.data_cache,
        g_state.fd_holders[fd],
        (const char *)data_ptr,
        num_pages,
        last_page_bytes);

    CUDA_CHECK(cudaDeviceSynchronize());
    g_state.fd_bytes[fd] = (size_t)num_bytes;
}

/* ------------------------------------------------------------------ */
/* prefetch — PM → DRAM pool (CPU memcpy, during gap period)           */
/* ------------------------------------------------------------------ */

void beaver_ext_prefetch(int64_t fd)
{
    BEXT_CHECK_INIT();
    if (fd < 0 || fd >= g_state.max_fds)
        throw std::invalid_argument("beaver_ext: fd out of range");

    uint32_t num_pages = g_state.fd_num_pages[fd];
    if (num_pages == 0) return;

    beaver_cache_t *dc = g_state.data_cache;
    if (!dc->dram_pool_host || !dc->dram_dev_ptrs) return;

    /* Get holder indices from device */
    uint32_t *h_holders = (uint32_t *)malloc(num_pages * sizeof(uint32_t));
    if (!h_holders) return;
    CUDA_CHECK(cudaMemcpy(h_holders, g_state.fd_holders[fd],
                          num_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Get holders from device (batch copy) */
    uint32_t min_idx = h_holders[0], max_idx = h_holders[0];
    for (uint32_t i = 1; i < num_pages; i++) {
        if (h_holders[i] < min_idx) min_idx = h_holders[i];
        if (h_holders[i] > max_idx) max_idx = h_holders[i];
    }
    uint32_t range = max_idx - min_idx + 1;

    beaver_holder_t *holders_host = (beaver_holder_t *)malloc(range * sizeof(beaver_holder_t));
    if (!holders_host) { free(h_holders); return; }

    CUDA_CHECK(cudaMemcpy(holders_host, &dc->holders[min_idx],
                          range * sizeof(beaver_holder_t), cudaMemcpyDeviceToHost));

    /* Prefetch each page: PM → pinned DRAM */
    size_t bytes_left = g_state.fd_bytes[fd];
    for (uint32_t i = 0; i < num_pages; i++) {
        uint32_t hidx = h_holders[i];
        if (dc->dram_dev_ptrs[hidx] != NULL) continue;  /* already cached */

        beaver_holder_t *h = &holders_host[hidx - min_idx];
        void *pm_addr = h->read_ptr;
        if (!pm_addr) continue;  /* not yet written */

        /* Fixed slot mapping: holder_idx → DRAM slot (no bump allocator) */
        if (hidx >= dc->dram_pool_capacity) continue;  /* safety check */

        char *dram_page = dc->dram_pool_host + (size_t)hidx * PAGE_SIZE;
        uint32_t page_bytes = (bytes_left >= PAGE_SIZE) ? PAGE_SIZE : (uint32_t)bytes_left;

        /* PM → pinned DRAM (pm_addr is UVA, host-accessible) */
        memcpy(dram_page, pm_addr, page_bytes);

        /* Get GPU-accessible pointer for this DRAM page */
        void *dev_ptr = NULL;
        cudaError_t cu = cudaHostGetDevicePointer(&dev_ptr, dram_page, 0);
        if (cu == cudaSuccess && dev_ptr)
            dc->dram_dev_ptrs[hidx] = dev_ptr;

        bytes_left -= page_bytes;
    }

    free(holders_host);
    free(h_holders);
}

/* ------------------------------------------------------------------ */
/* read — DRAM (or PM fallback) → GPU tensor                           */
/* ------------------------------------------------------------------ */

void beaver_ext_read(int64_t fd, int64_t data_ptr, int64_t num_bytes)
{
    BEXT_CHECK_INIT();
    if (fd < 0 || fd >= g_state.max_fds)
        throw std::invalid_argument("beaver_ext: fd out of range");
    if (num_bytes <= 0)
        throw std::invalid_argument("beaver_ext: num_bytes must be > 0");

    uint32_t num_pages = g_state.fd_num_pages[fd];
    if (num_pages == 0)
        throw std::runtime_error("beaver_ext: fd not written yet");

    uint32_t last_page_bytes = (uint32_t)(num_bytes % PAGE_SIZE);
    if (last_page_bytes == 0) last_page_bytes = PAGE_SIZE;

    /* Launch read kernel */
    int threads = 256;
    int blocks = (int)num_pages;

    beaver_cow_read_kernel<<<blocks, threads>>>(
        g_state.data_cache,
        g_state.fd_holders[fd],
        (char *)data_ptr,
        num_pages,
        last_page_bytes);

    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ------------------------------------------------------------------ */
/* reset_dram_pool (no-op with fixed slot mapping)                      */
/* ------------------------------------------------------------------ */

void beaver_ext_reset_dram_pool()
{
    /* Fixed slot mapping: each holder always maps to the same DRAM slot.
     * No bump allocator to reset. Kept for API compatibility. */
}

/* ------------------------------------------------------------------ */
/* cleanup                                                              */
/* ------------------------------------------------------------------ */

void beaver_ext_cleanup()
{
    if (!g_state.initialized) return;

    /* Free per-fd holder arrays */
    for (int i = 0; i < g_state.max_fds; i++) {
        if (g_state.fd_holders[i])
            cudaFree(g_state.fd_holders[i]);
    }
    free(g_state.fd_holders);
    free(g_state.fd_num_pages);
    free(g_state.fd_bytes);

    /* Cleanup Beaver cache */
    if (g_state.data_cache) {
        beaver_cache_cleanup(g_state.data_cache);
        cudaFree(g_state.data_cache);  /* cudaMallocManaged → cudaFree */
    }

    /* Restore DDIO */
    if (g_state.ddio_disabled) {
        ddio_enable(g_state.gpu_bus);
        printf("[beaver_ext] DDIO restored (GPU bus 0x%02x)\n", g_state.gpu_bus);
    }

    /* Cleanup GPM */
    gpm_cleanup();

    g_state = BeaverExtState{};
    printf("[beaver_ext] cleaned up\n");
}

/* ------------------------------------------------------------------ */
/* pybind11 module                                                      */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Beaver GPU PM Offload (COW backend via src/core/)";

    m.def("init",   &beaver_ext_init,
          py::arg("max_inodes"), py::arg("max_data_pages"),
          py::arg("dram_pool_pages"));
    m.def("create_file", &beaver_ext_create_file, py::arg("hash"));
    m.def("write",   &beaver_ext_write,
          py::arg("fd"), py::arg("data_ptr"), py::arg("num_bytes"));
    m.def("prefetch", &beaver_ext_prefetch, py::arg("fd"));
    m.def("read",    &beaver_ext_read,
          py::arg("fd"), py::arg("data_ptr"), py::arg("num_bytes"));
    m.def("reset_dram_pool", &beaver_ext_reset_dram_pool);
    m.def("cleanup", &beaver_ext_cleanup);
}
