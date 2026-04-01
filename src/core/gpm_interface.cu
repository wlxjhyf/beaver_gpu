/*
 * gpm_interface.cu - Host-side GPM allocation/deallocation.
 *
 * Design (Linux 6.1 compatible):
 *   - gpm_init maps the entire /dev/dax1.0 devdax device ONCE with
 *     pmem_map_file.  devdax mmap produces VM_PFNMAP pages (not
 *     _PAGE_DEVMAP), so pin_user_pages(FOLL_LONGTERM) in cudaHostRegister
 *     succeeds on Linux 6.1+.
 *   - gpm_alloc slices sub-regions from the global base via a bump-pointer
 *     allocator (2 MB alignment) and registers each slice with
 *     cudaHostRegister(ptr, size, 0).
 *   - gpm_free unregisters the CUDA mapping.  No CPU-side pmem_persist is
 *     issued: DDIO is disabled during GPU PM writes, so GPU writes bypass
 *     CPU cache entirely; __threadfence_system() (gpm_persist on GPU) is
 *     the only drain needed.  The global mmap stays alive until gpm_cleanup.
 *   - gpm_cleanup unmaps the global devdax mapping.
 *
 * Device-side primitives (gpm_drain, gpm_persist, gpm_memcpy, etc.) are
 * all static __device__ inline functions defined in gpm_interface.cuh.
 * Nothing GPU-kernel-related belongs here.
 */

#include "gpm_interface.cuh"
#include "ddio.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Print only when VERBOSE=1 is set in environment */
static int gpm_verbose(void) { return getenv("VERBOSE") != NULL; }

/* 2 MB alignment for each gpm_alloc slice (matches devdax page granularity) */
#define GPM_ALLOC_ALIGN (2UL * 1024 * 1024)

static int    gpm_initialized    = 0;
static void*  gpm_devdax_base    = NULL;  /* base of the full devdax mmap  */
static size_t gpm_devdax_total   = 0;     /* total size of devdax device   */
static int    gpm_devdax_is_pmem = 0;     /* 1 = real Optane PM            */
static size_t gpm_devdax_offset  = 0;     /* bump-pointer for next alloc   */

gpm_error_t gpm_init(void)
{
    if (gpm_initialized)
        return GPM_SUCCESS;

    /* Verify CUDA device */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "gpm_init: no CUDA devices found\n");
        return GPM_ERROR_DEVICE_NOT_SUPPORTED;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (gpm_verbose())
        printf("gpm_init: device = %s (compute %d.%d)\n",
               prop.name, prop.major, prop.minor);

    /* Map the entire devdax device once.
     * size=0 tells pmem_map_file to determine the size from the device. */
    void* base = pmem_map_file(GPM_DEVDAX_PATH, 0 /* whole device */,
                               0 /* no PMEM_FILE_CREATE */, 0666,
                               &gpm_devdax_total, &gpm_devdax_is_pmem);
    if (!base) {
        fprintf(stderr, "gpm_init: pmem_map_file(%s) failed: %s\n",
                GPM_DEVDAX_PATH, pmem_errormsg());
        return GPM_ERROR_INIT_FAILED;
    }

    gpm_devdax_base   = base;
    gpm_devdax_offset = 0;

    if (gpm_verbose())
        printf("gpm_init: devdax base=%p  total=%.1f GB  is_pmem=%d\n",
               base, (double)gpm_devdax_total / (1024.0 * 1024.0 * 1024.0),
               gpm_devdax_is_pmem);

    if (!gpm_devdax_is_pmem)
        fprintf(stderr, "gpm_init: WARNING - device reports is_pmem=0; "
                "durability guarantees may not hold\n");

    gpm_initialized = 1;
    return GPM_SUCCESS;
}

gpm_error_t gpm_cleanup(void)
{
    if (!gpm_initialized)
        return GPM_SUCCESS;

    if (gpm_devdax_base) {
        pmem_unmap(gpm_devdax_base, gpm_devdax_total);
        gpm_devdax_base  = NULL;
        gpm_devdax_total = 0;
    }

    gpm_devdax_offset  = 0;
    gpm_devdax_is_pmem = 0;
    gpm_initialized    = 0;
    return GPM_SUCCESS;
}

gpm_error_t gpm_alloc(gpm_region_t* region, size_t size, const char* tag)
{
    if (!region || size == 0)
        return GPM_ERROR_INVALID_POINTER;
    if (!gpm_initialized)
        return GPM_ERROR_INIT_FAILED;

    /* Round up to 2 MB alignment */
    size_t alloc_size = (size + GPM_ALLOC_ALIGN - 1) & ~(GPM_ALLOC_ALIGN - 1);

    if (gpm_devdax_offset + alloc_size > gpm_devdax_total) {
        fprintf(stderr, "gpm_alloc: devdax space exhausted "
                "(need %zu, available %zu)\n",
                alloc_size, gpm_devdax_total - gpm_devdax_offset);
        return GPM_ERROR_ALLOC_FAILED;
    }

    void* sub = (char*)gpm_devdax_base + gpm_devdax_offset;

    /* Register sub-region with CUDA.
     * cudaHostRegisterMapped explicitly requests GPU address-space mapping;
     * required for device kernels to dereference PM addresses at any scale.
     * devdax VM_PFNMAP pages pass pin_user_pages(FOLL_LONGTERM) on Linux 6.1+. */
    cudaError_t cuda_err = cudaHostRegister(sub, alloc_size,
                                            cudaHostRegisterMapped);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "gpm_alloc: cudaHostRegister(%p, %zu, Mapped) failed: %s\n",
                sub, alloc_size, cudaGetErrorString(cuda_err));
        return GPM_ERROR_ALLOC_FAILED;
    }

    /* Obtain the GPU-accessible device pointer.
     * On UVA systems this equals sub, but calling cudaHostGetDevicePointer
     * makes the mapping explicit and is the only correct way to get a device
     * pointer for memory registered with cudaHostRegister. */
    void *dev_ptr = NULL;
    cuda_err = cudaHostGetDevicePointer(&dev_ptr, sub, 0);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "gpm_alloc: cudaHostGetDevicePointer(%p) failed: %s\n",
                sub, cudaGetErrorString(cuda_err));
        cudaHostUnregister(sub);
        return GPM_ERROR_ALLOC_FAILED;
    }
    if (dev_ptr != sub)
        fprintf(stderr, "gpm_alloc: NOTE host VA %p != dev VA %p "
                "(using dev VA for GPU access)\n", sub, dev_ptr);

    region->addr      = sub;
    region->dev_addr  = dev_ptr;
    region->size      = alloc_size;
    region->pm_offset = gpm_devdax_offset;
    region->is_pmem   = gpm_devdax_is_pmem;
    region->is_valid  = 1;

    if (gpm_verbose())
        printf("gpm_alloc[%s]: %zu bytes at %p  offset=%zu  is_pmem=%d\n",
               tag ? tag : "rgn", alloc_size, sub, gpm_devdax_offset,
               gpm_devdax_is_pmem);

    gpm_devdax_offset += alloc_size;
    return GPM_SUCCESS;
}

gpm_error_t gpm_free(gpm_region_t* region)
{
    if (!region || !region->is_valid)
        return GPM_ERROR_INVALID_POINTER;

    /* Unregister CUDA mapping for this sub-region */
    cudaError_t cuda_err = cudaHostUnregister(region->addr);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "gpm_free: cudaHostUnregister warning: %s\n",
                cudaGetErrorString(cuda_err));
        /* Non-fatal: continue cleanup */
    }

    /* No CPU-side persist: DDIO is disabled during GPU writes, so GPU
     * PCIe writes bypass CPU L3 cache.  gpu_holder_commit already issued
     * __threadfence_system(); clwb/sfence here would be no-ops. */

    /* The global devdax mapping stays alive; offset is not reclaimed
     * (bump allocator - no compaction for this prototype). */
    memset(region, 0, sizeof(gpm_region_t));
    return GPM_SUCCESS;
}
