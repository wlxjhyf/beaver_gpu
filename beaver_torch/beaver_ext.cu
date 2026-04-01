/*
 * beaver_ext.cu — PyTorch CUDA Extension for Beaver PM Offload
 *
 * GPU-direct PM write path (primary implementation).
 *
 * Diagnostic confirmed (gpm_large_register_test --simulate-torch):
 *   cudaHostRegister works for up to 16 GB of devdax even with
 *   3 GB VRAM + 2.75 GB pinned memory pre-allocated — equivalent to
 *   loading GPT-2 XL + dram_pool. The previous "illegal memory access"
 *   was a bug in the write kernel, NOT a cudaHostRegister size limit.
 *
 * Write path (GPU-direct, crash-consistent):
 *   gpu_pm_write_kernel: volatile uint64 stores from GPU tensor → PM
 *   + __threadfence_system() per block ensures PM durability
 *   (DDIO disabled so writes bypass CPU cache and reach PM media)
 *
 * Prefetch path (gap period, CPU):
 *   PM host VA → CPU memcpy → pinned DRAM pool
 *
 * Read path (DRAM hit after prefetch):
 *   pinned DRAM pool → cudaMemcpy H2D → GPU tensor
 *
 * PM layout (flat, no COW overhead):
 *   fd N → PM offset [N * pm_stride, (N+1) * pm_stride)
 *   pm_stride = CHUNK_PAGES * PAGE_SIZE = 1017 * 4096 ≈ 4.16 MiB
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libpmem.h>
#include <torch/extension.h>

extern "C" {
    uint8_t ddio_get_gpu_bus(void);
    void    ddio_disable(uint8_t bus);
    void    ddio_enable (uint8_t bus);
}

#define PAGE_SIZE       4096u
#define CHUNK_PAGES     1017u
#define CHUNK_BYTES     ((size_t)CHUNK_PAGES * PAGE_SIZE)   /* ~4.16 MiB */
#define GPM_DEVDAX_PATH "/dev/dax1.0"

/* ------------------------------------------------------------------ */
/* GPU direct PM write kernel                                           */
/* ------------------------------------------------------------------ */

/*
 * gpu_pm_write_kernel — copy GPU tensor → PM via volatile stores.
 *
 * volatile uint64_t stores bypass the GPU L2 cache and travel directly
 * through PCIe to PM. With DDIO disabled, they reach PM media without
 * being trapped in CPU cache.
 *
 * __threadfence_system() per block: system-scope store fence that ensures
 * all preceding volatile stores from this block are visible to CPU/PM
 * before the kernel reports completion. Combined with cudaDeviceSynchronize
 * on the host, this makes each write crash-consistent.
 *
 * Thread layout: each thread handles words at positions tid, tid+total,
 * tid+2*total, ... (stride = total threads). This gives coalesced access
 * within each warp (32 consecutive threads → 256B PCIe transaction).
 */
__global__ void gpu_pm_write_kernel(volatile uint64_t *dst,
                                     const  uint64_t  *src,
                                     size_t            n_words)
{
    size_t total = (size_t)gridDim.x * blockDim.x;
    size_t tid   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n_words; i += total)
        dst[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0)
        __threadfence_system();
}

/* ------------------------------------------------------------------ */
/* Extension state                                                      */
/* ------------------------------------------------------------------ */

struct BeaverExtState {
    /* PM storage */
    char*  pm_host;           /* host VA from pmem_map_file (CPU prefetch)    */
    char*  pm_dev;            /* GPU VA from cudaHostGetDevicePointer (write)  */
    size_t pm_total;          /* total devdax size                             */
    size_t pm_registered;     /* bytes registered with cudaHostRegister        */
    size_t pm_stride;         /* PM bytes per fd = CHUNK_BYTES                 */
    int    max_fds;
    int    next_fd;

    size_t* fd_bytes;         /* fd_bytes[fd] = bytes written                  */

    /* DRAM pool: pinned host memory for prefetch → read path */
    char*  dram_pool;
    size_t dram_pool_size;
    size_t dram_pool_used;
    char** fd_dram;           /* fd_dram[fd] → DRAM slot after prefetch        */

    uint8_t gpu_bus;
    bool    ddio_disabled;
    bool    pm_registered_flag;
    bool    initialized;
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

void beaver_ext_init(int64_t max_inodes, int64_t /*max_data_pages*/,
                     int64_t dram_pool_pages)
{
    if (g_state.initialized)
        throw std::runtime_error("beaver_ext: already initialized");

    /* Disable DDIO: GPU PCIe writes bypass CPU L3 and reach PM media */
    g_state.gpu_bus = ddio_get_gpu_bus();
    ddio_disable(g_state.gpu_bus);
    g_state.ddio_disabled = true;
    printf("[beaver_ext] DDIO disabled (GPU bus 0x%02x)\n", g_state.gpu_bus);

    /* Map devdax via libpmem. host VA used for CPU-side prefetch reads. */
    size_t pm_total = 0;
    int    is_pmem  = 0;
    void*  pm_host  = pmem_map_file(GPM_DEVDAX_PATH, 0, 0, 0666,
                                    &pm_total, &is_pmem);
    if (!pm_host)
        throw std::runtime_error(
            std::string("beaver_ext: pmem_map_file failed: ") + pmem_errormsg());
    if (!is_pmem)
        fprintf(stderr, "[beaver_ext] WARNING: %s reports is_pmem=0\n",
                GPM_DEVDAX_PATH);

    g_state.pm_host  = (char*)pm_host;
    g_state.pm_total = pm_total;
    g_state.pm_stride = CHUNK_BYTES;
    g_state.max_fds   = (int)max_inodes;
    g_state.next_fd   = 0;

    size_t pm_needed = (size_t)max_inodes * CHUNK_BYTES;
    if (pm_needed > pm_total)
        throw std::runtime_error("beaver_ext: devdax too small for requested max_inodes");

    /*
     * Register the PM slab with CUDA for GPU kernel access.
     *
     * cudaHostRegister(addr, size, cudaHostRegisterMapped):
     *   - Programs GPU MMU PTEs to map PM physical addresses into GPU VA space
     *   - dev_ptr obtained via cudaHostGetDevicePointer is the GPU VA
     *   - GPU volatile stores to dev_ptr travel via PCIe to PM media
     *
     * Size validation: gpm_large_register_test --simulate-torch confirmed
     * cudaHostRegister works for up to 16 GB of devdax even after
     * pre-allocating 3 GB VRAM + 2.75 GB pinned memory.
     */
    CUDA_CHECK(cudaHostRegister(pm_host, pm_needed, cudaHostRegisterMapped));
    g_state.pm_registered     = pm_needed;
    g_state.pm_registered_flag = true;

    void *pm_dev = NULL;
    CUDA_CHECK(cudaHostGetDevicePointer(&pm_dev, pm_host, 0));
    g_state.pm_dev = (char*)pm_dev;

    printf("[beaver_ext] PM slab: %.2f GB  host=%p  dev=%p  is_pmem=%d\n",
           (double)pm_needed / (1UL << 30), pm_host, pm_dev, is_pmem);

    /* Per-fd metadata */
    g_state.fd_bytes = (size_t*)calloc(max_inodes, sizeof(size_t));
    g_state.fd_dram  = (char**)calloc(max_inodes, sizeof(char*));
    if (!g_state.fd_bytes || !g_state.fd_dram)
        throw std::runtime_error("beaver_ext: calloc fd tables failed");

    /* DRAM pool for prefetch → read (pinned host, accessible via cudaMemcpy H2D) */
    size_t pool_bytes = (size_t)dram_pool_pages * PAGE_SIZE;
    CUDA_CHECK(cudaHostAlloc((void**)&g_state.dram_pool, pool_bytes,
                             cudaHostAllocPortable));
    g_state.dram_pool_size = pool_bytes;
    g_state.dram_pool_used = 0;

    g_state.initialized = true;
    printf("[beaver_ext] initialized (GPU-direct write path):\n"
           "  max_fds=%ld  pm_stride=%.1f MiB  dram_pool=%.2f GB\n",
           (long)max_inodes,
           (double)CHUNK_BYTES / (1024.0 * 1024.0),
           (double)pool_bytes  / (1024.0 * 1024.0 * 1024.0));
}

/* ------------------------------------------------------------------ */
/* create_file                                                          */
/* ------------------------------------------------------------------ */

int64_t beaver_ext_create_file(int64_t /*hash*/)
{
    BEXT_CHECK_INIT();
    if (g_state.next_fd >= g_state.max_fds)
        throw std::runtime_error("beaver_ext: max_fds exhausted");
    return (int64_t)(g_state.next_fd++);
}

/* ------------------------------------------------------------------ */
/* write — GPU tensor → PM via volatile stores                         */
/* ------------------------------------------------------------------ */

/*
 * beaver_ext_write:
 *   Launches gpu_pm_write_kernel to copy `num_bytes` from the GPU tensor
 *   at `data_ptr` (device address from tensor.data_ptr()) to the PM slot
 *   for `fd`. cudaDeviceSynchronize ensures the write is PM-durable before
 *   returning (safe to prefetch from PM immediately after).
 *
 *   data_ptr: raw device pointer passed as int64 from Python.
 *             tensor.data_ptr() returns the CUDA device address; pybind11
 *             passes it as int64_t; (void*) cast recovers the original ptr.
 */
void beaver_ext_write(int64_t fd, int64_t data_ptr, int64_t num_bytes)
{
    BEXT_CHECK_INIT();
    if (fd < 0 || fd >= g_state.max_fds)
        throw std::invalid_argument("beaver_ext: fd out of range");
    if (num_bytes <= 0 || (size_t)num_bytes > CHUNK_BYTES)
        throw std::invalid_argument("beaver_ext: num_bytes out of range");

    if (data_ptr == 0)
        throw std::runtime_error("beaver_ext: data_ptr is NULL (tensor not on GPU?)");

    cudaPointerAttributes attrs;
    cudaError_t pa = cudaPointerGetAttributes(&attrs, (void*)data_ptr);
    if (pa != cudaSuccess) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "beaver_ext: cudaPointerGetAttributes(0x%lx) failed: %s",
                 (unsigned long)data_ptr, cudaGetErrorString(pa));
        throw std::runtime_error(buf);
    }
    if (attrs.type != cudaMemoryTypeDevice) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "beaver_ext: data_ptr 0x%lx is not device memory (type=%d, "
                 "expect cudaMemoryTypeDevice=%d). Tensor may be on CPU.",
                 (unsigned long)data_ptr, (int)attrs.type,
                 (int)cudaMemoryTypeDevice);
        throw std::runtime_error(buf);
    }

    /* PM destination: GPU-accessible device pointer for this fd's slot */
    volatile uint64_t *pm_dst =
        (volatile uint64_t *)(g_state.pm_dev + (size_t)fd * g_state.pm_stride);

    /* GPU source: device pointer from Python tensor.data_ptr() */
    const uint64_t *gpu_src = (const uint64_t *)(void *)data_ptr;

    /* Round up to 8-byte words (GPU store granularity) */
    size_t n_words = ((size_t)num_bytes + 7) / 8;

    /*
     * Thread count tuning: rawgpm_bench shows 1-writer (32 threads) gives
     * peak PM bandwidth (~3165 MB/s) for sequential access. More concurrent
     * writers fragment WPQ and reduce bandwidth. For 4 MiB chunks, 256×32
     * (8192 threads) balances latency vs WPQ congestion.
     * Cap at 512 blocks to avoid excessive fragmentation.
     */
    int threads = 256;
    int blocks  = (int)((n_words + (size_t)threads - 1) / (size_t)threads);
    if (blocks > 512) blocks = 512;
    if (blocks < 1)   blocks = 1;

    gpu_pm_write_kernel<<<blocks, threads>>>(pm_dst, gpu_src, n_words);
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

    size_t n = g_state.fd_bytes[fd];
    if (n == 0) return;   /* not yet written */

    if (g_state.dram_pool_used + n > g_state.dram_pool_size)
        throw std::runtime_error("beaver_ext: DRAM pool exhausted");

    char* dram_slot = g_state.dram_pool + g_state.dram_pool_used;
    char* pm_addr   = g_state.pm_host   + (size_t)fd * g_state.pm_stride;

    memcpy(dram_slot, pm_addr, n);

    g_state.fd_dram[fd]     = dram_slot;
    g_state.dram_pool_used += n;
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

    const char* src = g_state.fd_dram[fd];
    if (!src)
        src = g_state.pm_host + (size_t)fd * g_state.pm_stride;  /* PM fallback */

    CUDA_CHECK(cudaMemcpy((void*)data_ptr, src,
                          (size_t)num_bytes, cudaMemcpyHostToDevice));
}

/* ------------------------------------------------------------------ */
/* reset_dram_pool                                                      */
/* ------------------------------------------------------------------ */

void beaver_ext_reset_dram_pool()
{
    BEXT_CHECK_INIT();
    g_state.dram_pool_used = 0;
    memset(g_state.fd_dram, 0, (size_t)g_state.max_fds * sizeof(char*));
}

/* ------------------------------------------------------------------ */
/* cleanup                                                              */
/* ------------------------------------------------------------------ */

void beaver_ext_cleanup()
{
    if (!g_state.initialized) return;

    if (g_state.dram_pool)   cudaFreeHost(g_state.dram_pool);
    if (g_state.fd_bytes)    free(g_state.fd_bytes);
    if (g_state.fd_dram)     free(g_state.fd_dram);

    if (g_state.pm_registered_flag && g_state.pm_host)
        cudaHostUnregister(g_state.pm_host);

    if (g_state.ddio_disabled) {
        ddio_enable(g_state.gpu_bus);
        printf("[beaver_ext] DDIO restored (GPU bus 0x%02x)\n", g_state.gpu_bus);
    }

    if (g_state.pm_host)
        pmem_unmap(g_state.pm_host, g_state.pm_total);

    g_state = BeaverExtState{};
    printf("[beaver_ext] cleaned up\n");
}

/* ------------------------------------------------------------------ */
/* pybind11 module                                                      */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Beaver GPU PM Offload (GPU-direct write via cudaHostRegister'd devdax)";

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
