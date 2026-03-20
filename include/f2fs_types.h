/*
 * f2fs_types.h — Shared F2FS inode format and constants.
 *
 * This header is dependency-free (only cuda_runtime.h + stdint.h) so that
 * both the Beaver-enhanced F2FS (gpu_f2fs.h) and the pure F2FS baseline
 * (gpu_f2fs_ckpt.h) can share the same on-disk inode layout without
 * pulling in any Beaver COW headers.
 */

#ifndef F2FS_TYPES_H
#define F2FS_TYPES_H

#include <cuda_runtime.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

#define F2FS_PAGE_SIZE  4096u

/*
 * Direct block address entries in one inode page.
 * Header = 28 bytes; remaining = 4096 - 28 = 4068 = 1017 × 4.
 */
#define F2FS_ADDRS_PER_INODE  1017u

/* Sentinel values */
#define F2FS_NULL_ADDR    0xFFFFFFFFu   /* i_addr slot not allocated   */
#define F2FS_NAME_EMPTY   0xFFFFFFFFu   /* name_table slot is empty    */

/* ------------------------------------------------------------------ */
/* f2fs_inode_t — on-disk inode page (exactly 4096 bytes)            */
/*                                                                     */
/* i_addr[] semantics depend on which FS uses this struct:            */
/*   Beaver F2FS (gpu_f2fs.h):  stores data holder_idx in data_cache */
/*   Pure F2FS baseline:        stores physical PM block address      */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t  name_hash;                    /* file name hash          */
    uint32_t  i_uid;
    uint32_t  i_gid;
    uint32_t  i_blocks;                     /* allocated data pages    */
    uint64_t  i_size;                       /* file size in bytes      */
    uint16_t  i_mode;
    uint8_t   i_advise;
    uint8_t   _pad0;
    /* offset 28 — direct block addresses / holder indices */
    uint32_t  i_addr[F2FS_ADDRS_PER_INODE];
} f2fs_inode_t;   /* sizeof must equal F2FS_PAGE_SIZE */

#ifdef __cplusplus
static_assert(sizeof(f2fs_inode_t) == F2FS_PAGE_SIZE,
              "f2fs_inode_t must be exactly 4096 bytes");
#endif

/* ------------------------------------------------------------------ */
/* f2fs_name_slot: hash mix for name-table lookup                     */
/* ------------------------------------------------------------------ */
static __device__ __host__ __forceinline__ uint32_t
f2fs_name_slot(uint32_t h, uint32_t sz)
{
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h % sz;
}

#endif /* F2FS_TYPES_H */
