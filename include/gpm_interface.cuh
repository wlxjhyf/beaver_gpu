#pragma once
/*
 * GPM Interface - aligned with GPM-ASPLOS22 (libgpm.cuh / gpm-helper.cuh)
 *
 * Split into two parts:
 *   1. Device-side primitives (static __device__ inline) - callable directly
 *      from GPU kernels, no CPU involvement whatsoever.
 *   2. Host-side allocation functions (implemented in gpm_interface.cu) -
 *      pmem_map_file + cudaHostRegister, called once at setup time.
 *
 * Key design decisions matching GPM-ASPLOS22:
 *   - gpm_persist = gpm_drain() only (flush is NOT needed for GPU->PM writes
 *     that go directly through PCIe; this matches GPM-ASPLOS22 where gpm_flush
 *     is commented out inside gpm_persist).
 *   - gpm_drain = __threadfence_system() (system-scope fence, not just GPU-scope).
 *   - gpm_memcpy_nodrain uses volatile stores to give PM write semantics.
 *   - gpm_is_pmem on GPU always returns true (no reliable runtime check available).
 *   - PM regions are carved from a single devdax mapping (GPM_DEVDAX_PATH).
 *   - cudaHostRegister flag 0 (default) is used; with CUDA UVA the same pointer
 *     is accessible from both host and device, stored in gpm_region_t.addr.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <libpmem.h>

/* ------------------------------------------------------------------ */
/* Types                                                                */
/* ------------------------------------------------------------------ */

#define GPM_DWORD  unsigned long long   /* 8 bytes */
#define GPM_WORD   unsigned int         /* 4 bytes */
#define GPM_BYTE   unsigned char        /* 1 byte  */

/* devdax device path - used by gpm_init to map the entire device once */
#define GPM_DEVDAX_PATH "/dev/dax1.0"

/*
 * gpm_region_t: descriptor for a persistent memory region.
 *
 * After gpm_alloc():
 *   addr      - sub-pointer inside the global devdax mmap (host virtual address,
 *               for CPU-side memcpy/prefetch and gpm_free/unregister).
 *   dev_addr  - GPU-accessible device pointer, obtained via
 *               cudaHostGetDevicePointer after cudaHostRegister.
 *               On UVA systems dev_addr == addr, but always use dev_addr for
 *               any pointer that will be dereferenced by GPU kernels.
 *   pm_offset - byte offset from the start of the devdax device (for T2-style
 *               persistence verification via a fresh independent mmap).
 */
typedef struct {
    void*  addr;      /* PM host virtual address (CPU memcpy, gpm_free)    */
    void*  dev_addr;  /* GPU-accessible pointer (use in device code)       */
    size_t size;      /* Byte size of this region (2 MB-aligned)           */
    size_t pm_offset; /* Offset from devdax base (for remap/verify)        */
    int    is_pmem;   /* 1 = real PM device, 0 = DAX-emulated              */
    int    is_valid;  /* 1 = region is live                                */
} gpm_region_t;

/* Error codes */
typedef enum {
    GPM_SUCCESS                    =  0,
    GPM_ERROR_INIT_FAILED          = -1,
    GPM_ERROR_ALLOC_FAILED         = -2,
    GPM_ERROR_FREE_FAILED          = -3,
    GPM_ERROR_INVALID_POINTER      = -4,
    GPM_ERROR_DEVICE_NOT_SUPPORTED = -5,
    GPM_ERROR_NO_PM_DEVICE         = -6,
} gpm_error_t;

/* ------------------------------------------------------------------ */
/* Device-side PM primitives                                           */
/* (static __device__, inlined into every calling kernel)             */
/* ------------------------------------------------------------------ */

/*
 * gpm_is_pmem: range check.
 * GPU side: always true (GPM-ASPLOS22: "No way to reliably check from
 * inside GPU kernel for GPM-far").
 * Host side: delegate to libpmem.
 */
static __device__ __host__ __forceinline__ bool
gpm_is_pmem(const void* addr, size_t len)
{
#ifdef __CUDA_ARCH__
    (void)addr; (void)len;
    return true;
#else
    return pmem_is_pmem(addr, len) != 0;
#endif
}

/*
 * gpm_drain: system-scope store fence.
 * Ensures all preceding GPU stores have propagated through the PCIe path
 * and are visible to the PM media (and to the CPU).
 * Equivalent to __threadfence_system().
 */
static __device__ __forceinline__ void gpm_drain(void)
{
    __threadfence_system();
}

/*
 * gpm_flush: re-issue a range of existing PM contents via volatile stores.
 * This is a "self-write" pattern to force cache eviction on GPU L2.
 *
 * NOTE: In GPM-ASPLOS22, gpm_persist does NOT call gpm_flush (it is
 * commented out there). GPU writes to cudaHostRegister'd memory travel
 * directly through the memory subsystem to PM; a drain (fence) is
 * sufficient without an explicit flush.  This function is provided for
 * completeness and edge cases where explicit re-writes are needed.
 */
static __device__ __forceinline__ void gpm_flush(const void* addr, size_t len)
{
    size_t i = 0;

    /* 8-byte aligned fast path */
    volatile GPM_DWORD* b = (volatile GPM_DWORD*)((GPM_BYTE*)addr);
    for (; i + sizeof(GPM_DWORD) <= len
           && ((size_t)addr % sizeof(GPM_DWORD)) == 0;
         i += sizeof(GPM_DWORD), b++) {
        (*b) = *((GPM_DWORD*)((GPM_BYTE*)addr + i));
    }

    /* 4-byte tail */
    volatile GPM_WORD* c = (volatile GPM_WORD*)((GPM_BYTE*)addr + i);
    for (; i + sizeof(GPM_WORD) <= len
           && ((size_t)addr % sizeof(GPM_WORD)) == 0;
         i += sizeof(GPM_WORD), c++) {
        (*c) = *((GPM_WORD*)((GPM_BYTE*)addr + i));
    }

    /* byte tail */
    volatile GPM_BYTE* d = (volatile GPM_BYTE*)((GPM_BYTE*)addr + i);
    for (; i < len; i++, d++) {
        (*d) = *((GPM_BYTE*)addr + i);
    }
}

/*
 * gpm_persist: make a PM range durable from inside a GPU kernel.
 *
 * Matches GPM-ASPLOS22 gpm_persist:
 *   //gpm_flush(addr, len);   <- commented out
 *   gpm_drain();
 *
 * Every GPU thread that writes to PM should call gpm_persist (or at least
 * gpm_drain) before releasing ownership of the data.
 */
static __device__ __forceinline__ void gpm_persist(const void* addr, size_t len)
{
    /* gpm_flush(addr, len); -- not needed for GPU->PM via PCIe path */
    (void)addr; (void)len;
    gpm_drain();
}

/*
 * gpm_memcpy_nodrain_warp: warp-cooperative volatile stores to PM.
 *
 * ALL 32 threads in the warp must call this together with the same
 * (dest, src, len) and their own lane index (threadIdx.x & 31).
 * Thread `lane` writes 8B words at offsets lane, lane+32, lane+64, ... ,
 * producing a coalesced 256B PCIe write transaction per iteration across
 * the warp — vs. 512 independent 8B transactions in the scalar path.
 *
 * Requires: dest and src are 8B-aligned; len is a multiple of 8.
 * Does NOT issue a drain.
 */
static __device__ __forceinline__ void
gpm_memcpy_nodrain_warp(void *dest, const void *src, size_t len, uint32_t lane)
{
    const uint32_t n_words = (uint32_t)(len / sizeof(GPM_DWORD));
    volatile GPM_DWORD       *d = (volatile GPM_DWORD *)dest;
    const          GPM_DWORD *s = (const          GPM_DWORD *)src;
    for (uint32_t i = lane; i < n_words; i += 32u)
        d[i] = s[i];
}

/*
 * gpm_memcpy_nodrain: copy [src, src+len) to PM dest using volatile stores.
 * Volatile stores give PM write semantics (no reordering past the store).
 * Does NOT issue a drain; call gpm_drain() or gpm_persist() afterwards
 * if durability is required before other threads see the result.
 */
static __device__ __forceinline__ cudaError_t
gpm_memcpy_nodrain(void* dest, const void* src, size_t len)
{
    size_t i = 0;

    volatile GPM_DWORD* b = (volatile GPM_DWORD*)dest;
    for (; i + sizeof(GPM_DWORD) <= len
           && ((size_t)dest % sizeof(GPM_DWORD)) == 0
           && ((size_t)src  % sizeof(GPM_DWORD)) == 0;
         i += sizeof(GPM_DWORD), b++) {
        (*b) = *((GPM_DWORD*)src + (i / sizeof(GPM_DWORD)));
    }

    volatile GPM_WORD* c = (volatile GPM_WORD*)((GPM_BYTE*)dest + i);
    for (; i + sizeof(GPM_WORD) <= len
           && ((size_t)dest % sizeof(GPM_WORD)) == 0
           && ((size_t)src  % sizeof(GPM_WORD)) == 0;
         i += sizeof(GPM_WORD), c++) {
        (*c) = *((GPM_WORD*)src + (i / sizeof(GPM_WORD)));
    }

    volatile GPM_BYTE* d = (volatile GPM_BYTE*)((GPM_BYTE*)dest + i);
    for (; i < len; i++, d++) {
        (*d) = *((GPM_BYTE*)src + i);
    }
    return cudaSuccess;
}

/*
 * gpm_memcpy: copy src to PM dest and drain (fully durable on return).
 */
static __device__ __forceinline__ cudaError_t
gpm_memcpy(void* dest, const void* src, size_t len)
{
    gpm_memcpy_nodrain(dest, src, len);
    gpm_drain();
    return cudaSuccess;
}

/*
 * gpm_memset_nodrain: fill PM dest with val via volatile stores, no drain.
 */
static __device__ __forceinline__ cudaError_t
gpm_memset_nodrain(void* dest, unsigned char val, size_t len)
{
    size_t i = 0;

    if (val == 0) {
        volatile GPM_DWORD* b = (volatile GPM_DWORD*)dest;
        for (; i + sizeof(GPM_DWORD) <= len
               && ((size_t)dest % sizeof(GPM_DWORD)) == 0;
             i += sizeof(GPM_DWORD), b++) {
            (*b) = 0ULL;
        }
        volatile GPM_WORD* c = (volatile GPM_WORD*)((GPM_BYTE*)dest + i);
        for (; i + sizeof(GPM_WORD) <= len
               && ((size_t)dest % sizeof(GPM_WORD)) == 0;
             i += sizeof(GPM_WORD), c++) {
            (*c) = 0U;
        }
    }

    volatile GPM_BYTE* d = (volatile GPM_BYTE*)((GPM_BYTE*)dest + i);
    for (; i < len; i++, d++) {
        (*d) = val;
    }
    return cudaSuccess;
}

/*
 * gpm_memset: fill PM dest with val and drain.
 */
static __device__ __forceinline__ cudaError_t
gpm_memset(void* dest, unsigned char val, size_t len)
{
    gpm_memset_nodrain(dest, val, len);
    gpm_drain();
    return cudaSuccess;
}

/* ------------------------------------------------------------------ */
/* Host-side allocation functions (implemented in gpm_interface.cu)   */
/*                                                                     */
/* Declared as __host__ so nvcc knows they are host-only.             */
/* The declarations must be visible in both host and device passes     */
/* (nvcc parses the whole file in both passes), so no __CUDA_ARCH__   */
/* guard is used here.                                                 */
/* ------------------------------------------------------------------ */

/*
 * gpm_init: verify CUDA device presence.
 * Does NOT check for PM device here - that happens lazily in gpm_alloc.
 */
__host__ gpm_error_t gpm_init(void);

/* gpm_cleanup: no-op for now (regions must be freed individually). */
__host__ gpm_error_t gpm_cleanup(void);

/*
 * DDIO control - header-only, all static inline (same pattern as GPM-ASPLOS22).
 * See ddio.h for full documentation.
 *
 * Usage:
 *   uint8_t bus = gpm_find_gpu_bus();
 *   gpm_persist_begin(bus);          // disable DDIO before GPU PM kernel
 *   my_kernel<<<...>>>(pm_ptr);
 *   cudaDeviceSynchronize();
 *   gpm_persist_end(bus);           // restore DDIO
 */
#ifndef __CUDA_ARCH__          /* ddio.h is host-only (uses libpci) */
#include "ddio.h"

/* Aliases matching GPM-ASPLOS22's gpm_persist_begin / gpm_persist_end */
static inline void gpm_persist_begin(uint8_t gpu_bus) { gpm_ddio_off(gpu_bus); }
static inline void gpm_persist_end  (uint8_t gpu_bus) { gpm_ddio_on (gpu_bus); }
#endif /* !__CUDA_ARCH__ */

/*
 * gpm_alloc: carve a sub-region from the global devdax mapping.
 *
 * Slices `size` bytes (rounded up to 2 MB) from the bump-pointer allocator
 * over the devdax base, then calls cudaHostRegister(sub, size, 0).
 * Under CUDA UVA the same addr is valid on host and device.
 *
 * tag: optional label for log output; may be NULL.
 */
__host__ gpm_error_t gpm_alloc(gpm_region_t* region, size_t size, const char* tag);

/*
 * gpm_free: release a PM region.
 * Calls cudaHostUnregister + pmem_persist on the sub-region.
 * The global devdax mapping is NOT unmapped (stays alive until gpm_cleanup).
 */
__host__ gpm_error_t gpm_free(gpm_region_t* region);
