/*
 * gpu_vfs.h — GPU Virtual File System: interface types only.
 *
 * This file has NO knowledge of any concrete file system implementation.
 * It defines only:
 *   - gpu_fs_type_t : FS type tag enum
 *   - gpu_vfs_t     : VFS handle = {FS instance ptr, type tag}
 *
 * Dependency rule (must be enforced):
 *   gpu_vfs.h  ──►  <cuda_runtime.h>   (nothing FS-specific)
 *
 * VFS dispatch (vfs_create / vfs_open / vfs_write / vfs_read / vfs_close)
 * is defined in gpu_f2fs.h as __forceinline__ direct calls.  This avoids
 * CUDA separable-compilation limitations with cross-TU device function
 * pointer calls: storing a device function pointer in __device__ global
 * memory and calling it from a kernel in a different translation unit is
 * unreliable and causes cudaErrorIllegalInstruction on Ampere.
 *
 * Architecture:
 *
 *   GPU Kernel Code
 *        │  #include "gpu_f2fs.h"
 *        │  calls vfs_open / vfs_write / vfs_read / vfs_close  (forceinline)
 *        ▼
 *   gpu_f2fs.h  — vfs_* dispatch (direct calls, compile-time resolved)
 *        │
 *        ├──► F2FS-PM backend  (use_cow=1, Beaver COW)
 *        └──► F2FS-PM baseline (use_cow=0, Checkpoint)
 *            [future: add GPU_FS_F2FS_SSD case to vfs_* switch]
 */

#ifndef GPU_VFS_H
#define GPU_VFS_H

#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* FS type tag                                                         */
/* ------------------------------------------------------------------ */

typedef enum {
    GPU_FS_F2FS_PM  = 0,   /* our system: F2FS + PM + Beaver COW        */
    /* GPU_FS_F2FS_SSD = 1, future: F2FS + SSD + Checkpoint (GoFS-style) */
} gpu_fs_type_t;

/* ------------------------------------------------------------------ */
/* gpu_vfs_t — VFS handle (allocate with cudaMallocManaged)          */
/* ------------------------------------------------------------------ */

typedef struct {
    void          *fs;    /* concrete FS instance (device-accessible)   */
    gpu_fs_type_t  type;  /* type tag                                   */
} gpu_vfs_t;

#endif /* GPU_VFS_H */
