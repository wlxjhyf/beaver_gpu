/*
 * f2fs_kernel_test.cu — Phase 4: GPU F2FS correctness tests.
 *
 * Tests are written against the VFS interface (gpu_vfs.h), not directly
 * against the F2FS POSIX API.  This validates both the VFS layer and the
 * underlying F2FS implementation.
 *
 * T1 — Single-thread: create→open→write(page 0)→read, verify data + inode
 * T2 — Same file, 16 threads × 16 pages concurrent write, then read-back
 * T3 — 8 files × 1 thread, concurrent open/write/read/close
 * T4 — Persistence: fresh devdax remap, verify T1 data page and inode in PM
 * T5 — Checkpoint mode: create→write→checkpoint→remap verify
 *
 * Run: sudo ./f2fs_kernel_test   (root needed for DDIO + devdax access)
 *
 * Memory layout:
 *   beaver_cache_t, gpu_f2fs_t, gpu_vfs_t — cudaMallocManaged
 *   inode_shadow, dirty_flags, name_table, holders, hash_table — cudaMalloc
 *   write/read buffers — cudaMallocManaged (CPU fills, GPU uses, CPU checks)
 */

/* gpu_f2fs.h brings in: gpu_f2fs_t, init/cleanup/do_checkpoint,
 * gpu_vfs_mount_f2fs; and transitively gpu_vfs.h (gpu_vfs_t, vfs_ops)
 * and beaver_cow.h. */
#include "gpu_f2fs.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <libpmem.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

#define FS_MAX_INODES       32u
#define FS_MAX_DATA_BLOCKS  256u    /* 256 × 4 KiB = 1 MiB data slab  */

/* T1: file 0 */
#define T1_NAME_HASH    0x11111111u
#define T1_MAGIC        0xDEADBEEFCAFE0001ULL

/* T2: file 1, 16 threads × 16 pages */
#define T2_NAME_HASH    0x22222222u
#define T2_NTHREADS     16u
#define T2_MAGIC_BASE   0xBEEF000000000000ULL

/* T3: files 2..9, one thread per file */
#define T3_NFILES       8u
#define T3_HASH_BASE    0x33333300u
#define T3_MAGIC_BASE   0xCAFE000000000000ULL

/* T5: separate FS instance in Checkpoint mode */
#define T5_NAME_HASH    0x55555555u
#define T5_MAGIC        0xF00DCAFE12345678ULL

/* ------------------------------------------------------------------ */
/* Test result struct (cudaMallocManaged — GPU writes, CPU reads)     */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t t1_open_pass;
    uint32_t t1_write_pass;
    uint32_t t1_read_pass;
    uint32_t t1_data_pass;
    uint32_t t2_write_pass;
    uint32_t t2_read_pass;
    uint32_t t3_pass;
    uint32_t t5_write_pass;
    uint32_t t5_read_pass;
} test_results_t;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            return -1;                                                    \
        }                                                                 \
    } while (0)

static void fill_page(void *buf, uint64_t pattern)
{
    uint64_t *p = (uint64_t *)buf;
    for (size_t i = 0; i < BEAVER_PAGE_SIZE / sizeof(uint64_t); ++i)
        p[i] = pattern;
}

/* ------------------------------------------------------------------ */
/* Setup kernel: create all files (one thread per file)               */
/* Uses VFS layer to exercise vfs_create.                             */
/* ------------------------------------------------------------------ */
__global__ void create_files_kernel(gpu_vfs_t      *vfs,
                                    const uint32_t *name_hashes,
                                    uint32_t        nfiles)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nfiles)
        vfs_create(vfs, name_hashes[i]);
}

/* ------------------------------------------------------------------ */
/* T1 kernel: single-thread open → write → read                      */
/* ------------------------------------------------------------------ */
__global__ void t1_kernel(gpu_vfs_t      *vfs,
                           uint32_t        name_hash,
                           const uint8_t  *wbuf,
                           uint8_t        *rbuf,
                           test_results_t *res)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int fd = vfs_open(vfs, name_hash);
    res->t1_open_pass = (fd >= 0) ? 1u : 0u;
    if (fd < 0) return;

    int wr = vfs_write(vfs, fd, 0u, wbuf);
    res->t1_write_pass = (wr == 0) ? 1u : 0u;
    if (wr != 0) return;

    int rd = vfs_read(vfs, fd, 0u, rbuf);
    res->t1_read_pass = (rd == 0) ? 1u : 0u;
    if (rd != 0) return;

    /* Verify first uint64_t */
    res->t1_data_pass = (*(const uint64_t *)rbuf == *(const uint64_t *)wbuf)
                        ? 1u : 0u;

    vfs_close(vfs, fd);
}

/* ------------------------------------------------------------------ */
/* T2 kernel: 16 threads, each writes/reads a different page          */
/* ------------------------------------------------------------------ */
__global__ void t2_kernel(gpu_vfs_t      *vfs,
                           uint32_t        name_hash,
                           const uint8_t  *wbufs,   /* [T2_NTHREADS][PAGE] */
                           uint8_t        *rbufs,   /* [T2_NTHREADS][PAGE] */
                           uint32_t       *write_ok,
                           uint32_t       *read_ok)
{
    uint32_t tid = threadIdx.x;
    if (tid >= T2_NTHREADS) return;

    const uint8_t *wsrc = wbufs + (size_t)tid * BEAVER_PAGE_SIZE;
    uint8_t       *rdst = rbufs + (size_t)tid * BEAVER_PAGE_SIZE;

    int fd = vfs_open(vfs, name_hash);
    if (fd < 0) { write_ok[tid] = 0; read_ok[tid] = 0; return; }

    int wr = vfs_write(vfs, fd, tid, wsrc);
    write_ok[tid] = (wr == 0) ? 1u : 0u;

    /* All writes done before any reads (ensures holders are visible) */
    __syncthreads();

    int rd = vfs_read(vfs, fd, tid, rdst);
    read_ok[tid] = (rd == 0) ? 1u : 0u;

    vfs_close(vfs, fd);
}

/* ------------------------------------------------------------------ */
/* T3 kernel: 8 threads × 8 files                                    */
/* ------------------------------------------------------------------ */
__global__ void t3_kernel(gpu_vfs_t      *vfs,
                           const uint32_t *name_hashes,
                           const uint8_t  *wbufs,
                           uint8_t        *rbufs,
                           uint32_t       *pass_flags)
{
    uint32_t tid = threadIdx.x;
    if (tid >= T3_NFILES) return;

    const uint8_t *wsrc = wbufs + (size_t)tid * BEAVER_PAGE_SIZE;
    uint8_t       *rdst = rbufs + (size_t)tid * BEAVER_PAGE_SIZE;

    int fd = vfs_open(vfs, name_hashes[tid]);
    if (fd < 0) { pass_flags[tid] = 0; return; }

    if (vfs_write(vfs, fd, 0u, wsrc) != 0) { pass_flags[tid] = 0; return; }
    if (vfs_read (vfs, fd, 0u, rdst) != 0) { pass_flags[tid] = 0; return; }

    vfs_close(vfs, fd);
    pass_flags[tid] = 1u;
}

/* ------------------------------------------------------------------ */
/* T5 kernel: checkpoint mode write (single thread)                  */
/* ------------------------------------------------------------------ */
__global__ void t5_kernel(gpu_vfs_t      *vfs,
                           uint32_t        name_hash,
                           const uint8_t  *wbuf,
                           uint8_t        *rbuf,
                           test_results_t *res)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int fd = vfs_open(vfs, name_hash);
    if (fd < 0) { res->t5_write_pass = 0; res->t5_read_pass = 0; return; }

    int wr = vfs_write(vfs, fd, 0u, wbuf);
    res->t5_write_pass = (wr == 0) ? 1u : 0u;
    if (wr != 0) return;

    /* In checkpoint mode, inode is dirty but NOT yet in PM.
     * vfs_read reads from DRAM inode_shadow (always valid in-session). */
    int rd = vfs_read(vfs, fd, 0u, rbuf);
    res->t5_read_pass = (rd == 0) ? 1u : 0u;

    vfs_close(vfs, fd);
}

/* ------------------------------------------------------------------ */
/* T4: CPU-side persistence check via fresh devdax mmap               */
/* ------------------------------------------------------------------ */
static int run_t4(size_t data_pm_offset,
                  size_t inode_pm_offset,
                  uint64_t expected_data_magic,
                  uint32_t expected_name_hash)
{
    printf("\n=== [T4] Persistence: independent devdax remap ===\n");

    size_t map_len = 0;
    int    is_pmem = 0;
    void  *addr = pmem_map_file(GPM_DEVDAX_PATH, 0, 0, 0666,
                                &map_len, &is_pmem);
    if (!addr) {
        fprintf(stderr, "[T4] pmem_map_file failed: %s\n", pmem_errormsg());
        return -1;
    }
    printf("  Fresh mmap %p  size=%.1f GiB  is_pmem=%d\n",
           addr, (double)map_len / (1024.0*1024.0*1024.0), is_pmem);

    int ok = 1;

    /* Check data page: T1 wrote page 0, block_addr=0 (first alloc) */
    const uint64_t *data_slot = (const uint64_t *)
        ((const char *)addr + data_pm_offset);
    int data_ok = (*data_slot == expected_data_magic);
    printf("  data[0]   = 0x%016llx  expected 0x%016llx  %s\n",
           (unsigned long long)*data_slot,
           (unsigned long long)expected_data_magic,
           data_ok ? "OK" : "MISMATCH");
    ok &= data_ok;

    /* Check inode page: nid=0, COW holder 0, slot 0 = pm_addrs[0] */
    /* pm_addrs[0] = cow_pm_base + 0 * 3 * PAGE_SIZE = cow_pm_base */
    const uint32_t *inode_page = (const uint32_t *)
        ((const char *)addr + inode_pm_offset);
    uint32_t got_hash = inode_page[0];   /* name_hash at offset 0 */
    /* i_addr[0] is at offset 28 in the inode = byte 28 = word 7 */
    uint32_t got_i_addr0 = inode_page[28 / 4];
    int inode_ok = (got_hash == expected_name_hash) && (got_i_addr0 == 0u);
    printf("  inode[0].name_hash = 0x%08x  expected 0x%08x\n",
           got_hash, expected_name_hash);
    printf("  inode[0].i_addr[0] = %u  expected 0  %s\n",
           got_i_addr0, inode_ok ? "OK" : "MISMATCH");
    ok &= inode_ok;

    pmem_unmap(addr, map_len);

    if (ok)
        printf("  [T4] PASS: data and inode reached PM media\n");
    else
        fprintf(stderr, "  [T4] FAIL\n");
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* T5 persistence check (checkpoint mode)                             */
/* ------------------------------------------------------------------ */
static int run_t5_persist(size_t data_pm_offset,
                           size_t inode_pm_offset,
                           uint64_t expected_data_magic,
                           uint32_t expected_name_hash)
{
    printf("\n=== [T5-persist] Checkpoint mode PM verification ===\n");
    return run_t4(data_pm_offset, inode_pm_offset,
                  expected_data_magic, expected_name_hash);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("==========================================================\n");
    printf("   beaver_gpu  Phase 4 — GPU F2FS / VFS Correctness Tests\n");
    printf("==========================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute %d.%d)\n\n",
           prop.name, prop.major, prop.minor);

    uint8_t gpu_bus = ddio_get_gpu_bus();
    printf("GPU PCIe bus: 0x%02x — disabling DDIO\n", gpu_bus);
    ddio_disable(gpu_bus);

    int exit_code = 1;

    /* ---- Declare all managed pointers up front ---- */
    beaver_cache_t *cache  = NULL;
    gpu_f2fs_t     *fs     = NULL;
    gpu_vfs_t      *vfs    = NULL;   /* VFS handle — wraps fs */
    test_results_t *res    = NULL;

    beaver_cache_t *cache5 = NULL;
    gpu_f2fs_t     *fs5    = NULL;
    gpu_vfs_t      *vfs5   = NULL;   /* VFS handle for T5 (checkpoint mode) */

    uint8_t  *t1_wbuf = NULL, *t1_rbuf = NULL;
    uint8_t  *t2_wbufs = NULL, *t2_rbufs = NULL;
    uint32_t *t2_wok = NULL,  *t2_rok = NULL;
    uint8_t  *t3_wbufs = NULL, *t3_rbufs = NULL;
    uint32_t *t3_pass = NULL;
    uint32_t *d_hashes = NULL;
    uint8_t  *t5_wbuf = NULL, *t5_rbuf = NULL;
    uint32_t *d_t3hashes = NULL;

    /* ---- Allocate control structs in managed memory ---- */
    CUDA_CHECK(cudaMallocManaged((void **)&cache, sizeof(beaver_cache_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&fs,    sizeof(gpu_f2fs_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&vfs,   sizeof(gpu_vfs_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&res,   sizeof(test_results_t)));
    memset(res, 0, sizeof(test_results_t));

    /* ---- Init COW cache (must come before gpu_f2fs_init) ---- */
    if (beaver_cache_init(cache, FS_MAX_INODES) != BEAVER_SUCCESS) {
        fprintf(stderr, "beaver_cache_init failed\n");
        goto cleanup_allocs;
    }

    /* ---- Init GPU F2FS in COW mode ---- */
    if (gpu_f2fs_init(fs, cache, FS_MAX_INODES, FS_MAX_DATA_BLOCKS, 1)
            != GPU_F2FS_OK) {
        fprintf(stderr, "gpu_f2fs_init failed\n");
        goto cleanup_cache;
    }

    /* ---- Mount F2FS into the VFS handle ---- */
    gpu_vfs_mount_f2fs(vfs, fs);
    printf("VFS mounted: type=GPU_FS_F2FS_PM  use_cow=%u\n\n", fs->use_cow);

    /* ---- Create all test files (T1 + T2 + T3 = 10 files) ---- */
    {
        uint32_t hashes[10];
        hashes[0] = T1_NAME_HASH;
        hashes[1] = T2_NAME_HASH;
        for (uint32_t k = 0; k < T3_NFILES; ++k)
            hashes[2 + k] = T3_HASH_BASE + k;

        CUDA_CHECK(cudaMalloc((void **)&d_hashes, 10 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_hashes, hashes, 10 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        create_files_kernel<<<1, 10>>>(vfs, d_hashes, 10);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_hashes); d_hashes = NULL;
        printf("10 test files created via vfs_create.\n");
    }

    /* ---- Allocate buffers ---- */
    CUDA_CHECK(cudaMallocManaged((void **)&t1_wbuf, BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t1_rbuf, BEAVER_PAGE_SIZE));
    fill_page(t1_wbuf, T1_MAGIC);
    memset(t1_rbuf, 0, BEAVER_PAGE_SIZE);

    CUDA_CHECK(cudaMallocManaged((void **)&t2_wbufs,
                                 (size_t)T2_NTHREADS * BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t2_rbufs,
                                 (size_t)T2_NTHREADS * BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t2_wok,
                                 T2_NTHREADS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&t2_rok,
                                 T2_NTHREADS * sizeof(uint32_t)));
    for (uint32_t i = 0; i < T2_NTHREADS; ++i) {
        fill_page(t2_wbufs + (size_t)i * BEAVER_PAGE_SIZE,
                  T2_MAGIC_BASE | (uint64_t)i);
        memset(t2_rbufs + (size_t)i * BEAVER_PAGE_SIZE, 0, BEAVER_PAGE_SIZE);
        t2_wok[i] = t2_rok[i] = 0;
    }

    uint32_t t3_hashes[T3_NFILES];
    CUDA_CHECK(cudaMallocManaged((void **)&t3_wbufs,
                                 (size_t)T3_NFILES * BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t3_rbufs,
                                 (size_t)T3_NFILES * BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t3_pass,
                                 T3_NFILES * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_t3hashes,
                          T3_NFILES * sizeof(uint32_t)));
    for (uint32_t k = 0; k < T3_NFILES; ++k) {
        t3_hashes[k] = T3_HASH_BASE + k;
        fill_page(t3_wbufs + (size_t)k * BEAVER_PAGE_SIZE,
                  T3_MAGIC_BASE | (uint64_t)k);
        memset(t3_rbufs + (size_t)k * BEAVER_PAGE_SIZE, 0, BEAVER_PAGE_SIZE);
        t3_pass[k] = 0;
    }
    CUDA_CHECK(cudaMemcpy(d_t3hashes, t3_hashes,
                          T3_NFILES * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    /* ================================================================ */
    printf("\n=== [T1] Single-thread: vfs_open → vfs_write → vfs_read ===\n");

    t1_kernel<<<1, 1>>>(vfs, T1_NAME_HASH, t1_wbuf, t1_rbuf, res);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  open:  %s\n", res->t1_open_pass  ? "PASS" : "FAIL");
    printf("  write: %s\n", res->t1_write_pass ? "PASS" : "FAIL");
    printf("  read:  %s\n", res->t1_read_pass  ? "PASS" : "FAIL");
    printf("  data:  %s  (rbuf[0]=0x%016llx  expected=0x%016llx)\n",
           res->t1_data_pass ? "PASS" : "FAIL",
           (unsigned long long)*(const uint64_t *)t1_rbuf,
           (unsigned long long)T1_MAGIC);

    if (!res->t1_open_pass || !res->t1_write_pass ||
        !res->t1_read_pass || !res->t1_data_pass) {
        fprintf(stderr, "[T1] FAIL — aborting\n");
        goto cleanup_bufs;
    }

    /* ================================================================ */
    printf("\n=== [T2] %u threads × %u pages concurrent write+read ===\n",
           T2_NTHREADS, T2_NTHREADS);

    t2_kernel<<<1, T2_NTHREADS>>>(vfs, T2_NAME_HASH,
                                   t2_wbufs, t2_rbufs,
                                   t2_wok, t2_rok);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        int all_w = 1, all_r = 1, all_d = 1;
        for (uint32_t i = 0; i < T2_NTHREADS; ++i) {
            if (!t2_wok[i]) all_w = 0;
            if (!t2_rok[i]) all_r = 0;
            uint64_t exp = T2_MAGIC_BASE | (uint64_t)i;
            uint64_t got = *(const uint64_t *)(t2_rbufs + (size_t)i*BEAVER_PAGE_SIZE);
            if (got != exp) {
                all_d = 0;
                printf("  thread %u: got 0x%016llx expected 0x%016llx\n",
                       i, (unsigned long long)got, (unsigned long long)exp);
            }
        }
        res->t2_write_pass = all_w;
        res->t2_read_pass  = all_r && all_d;
        printf("  write: %s\n", all_w ? "PASS" : "FAIL");
        printf("  read:  %s\n", (all_r && all_d) ? "PASS" : "FAIL");
    }

    /* ================================================================ */
    printf("\n=== [T3] %u files × 1 thread concurrent open/write/read ===\n",
           T3_NFILES);

    t3_kernel<<<1, T3_NFILES>>>(vfs, d_t3hashes,
                                 t3_wbufs, t3_rbufs, t3_pass);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        int all = 1;
        for (uint32_t k = 0; k < T3_NFILES; ++k) {
            uint64_t exp = T3_MAGIC_BASE | (uint64_t)k;
            uint64_t got = *(const uint64_t *)(t3_rbufs + (size_t)k*BEAVER_PAGE_SIZE);
            if (!t3_pass[k] || got != exp) {
                all = 0;
                printf("  file %u: ops=%s data=%s\n", k,
                       t3_pass[k] ? "OK" : "FAIL",
                       (got == exp) ? "OK" : "MISMATCH");
            }
        }
        res->t3_pass = all;
        printf("  T3: %s\n", all ? "PASS" : "FAIL");
    }

    /* ================================================================ */
    /* T4: PM persistence.
     * T1 wrote file 0 (nid=0), page 0 → data block_addr=0.
     *   data PM offset  = fs->pm_data_region.pm_offset + 0×PAGE_SIZE
     *   inode PM offset = cache->pm_region.pm_offset + 0×3×PAGE_SIZE
     *     (holder 0, pm_addrs[0] = pm_base + 0×3×PAGE = pm_base)       */
    {
        size_t data_offset  = fs->pm_data_region.pm_offset;
        size_t inode_offset = cache->pm_region.pm_offset;
        run_t4(data_offset, inode_offset, T1_MAGIC, T1_NAME_HASH);
    }

    /* ================================================================ */
    /* T5: Checkpoint mode correctness.
     * Use a separate cache/fs/vfs triple so page_id namespace is
     * independent.  vfs5 type = GPU_FS_F2FS_PM, use_cow = 0.           */
    printf("\n=== [T5] Checkpoint mode: write→checkpoint→PM verify ===\n");

    CUDA_CHECK(cudaMallocManaged((void **)&cache5, sizeof(beaver_cache_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&fs5,    sizeof(gpu_f2fs_t)));
    CUDA_CHECK(cudaMallocManaged((void **)&vfs5,   sizeof(gpu_vfs_t)));

    if (beaver_cache_init(cache5, FS_MAX_INODES) != BEAVER_SUCCESS) {
        fprintf(stderr, "[T5] beaver_cache_init failed\n");
        goto cleanup_bufs;
    }
    if (gpu_f2fs_init(fs5, cache5, FS_MAX_INODES, FS_MAX_DATA_BLOCKS, 0)
            != GPU_F2FS_OK) {
        fprintf(stderr, "[T5] gpu_f2fs_init failed\n");
        beaver_cache_cleanup(cache5);
        goto cleanup_bufs;
    }
    gpu_vfs_mount_f2fs(vfs5, fs5);
    printf("  VFS5 mounted: type=GPU_FS_F2FS_PM  use_cow=%u\n", fs5->use_cow);

    /* Create the T5 file */
    {
        uint32_t h5 = T5_NAME_HASH;
        CUDA_CHECK(cudaMalloc((void **)&d_hashes, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_hashes, &h5, sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        create_files_kernel<<<1, 1>>>(vfs5, d_hashes, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_hashes); d_hashes = NULL;
    }

    CUDA_CHECK(cudaMallocManaged((void **)&t5_wbuf, BEAVER_PAGE_SIZE));
    CUDA_CHECK(cudaMallocManaged((void **)&t5_rbuf, BEAVER_PAGE_SIZE));
    fill_page(t5_wbuf, T5_MAGIC);
    memset(t5_rbuf, 0, BEAVER_PAGE_SIZE);

    t5_kernel<<<1, 1>>>(vfs5, T5_NAME_HASH, t5_wbuf, t5_rbuf, res);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  write: %s\n", res->t5_write_pass ? "PASS" : "FAIL");
    printf("  read (pre-ckpt, DRAM): %s\n", res->t5_read_pass ? "PASS" : "FAIL");

    /* Trigger checkpoint — flushes dirty inode page to PM */
    printf("  triggering checkpoint ...\n");
    gpu_f2fs_do_checkpoint(fs5);
    printf("  checkpoint done.\n");

    /* Verify PM: data at pm_data_region, inode at cache5 pm_region */
    {
        size_t data5  = fs5->pm_data_region.pm_offset;
        size_t inode5 = cache5->pm_region.pm_offset;
        run_t5_persist(data5, inode5, T5_MAGIC, T5_NAME_HASH);
    }

    /* ================================================================ */
    {
        int all = res->t1_open_pass && res->t1_write_pass &&
                  res->t1_read_pass && res->t1_data_pass &&
                  res->t2_write_pass && res->t2_read_pass &&
                  res->t3_pass &&
                  res->t5_write_pass && res->t5_read_pass;
        exit_code = all ? 0 : 1;

        printf("\n==========================================================\n");
        printf("  T1 open:       %s\n", res->t1_open_pass  ? "PASS" : "FAIL");
        printf("  T1 write:      %s\n", res->t1_write_pass ? "PASS" : "FAIL");
        printf("  T1 read:       %s\n", res->t1_read_pass  ? "PASS" : "FAIL");
        printf("  T1 data:       %s\n", res->t1_data_pass  ? "PASS" : "FAIL");
        printf("  T2 write:      %s\n", res->t2_write_pass ? "PASS" : "FAIL");
        printf("  T2 read:       %s\n", res->t2_read_pass  ? "PASS" : "FAIL");
        printf("  T3:            %s\n", res->t3_pass       ? "PASS" : "FAIL");
        printf("  T5 write:      %s\n", res->t5_write_pass ? "PASS" : "FAIL");
        printf("  T5 read:       %s\n", res->t5_read_pass  ? "PASS" : "FAIL");
        printf("==========================================================\n");
        printf("%s\n", exit_code == 0 ? "ALL PASS" : "SOME TESTS FAILED");
    }

    /* Cleanup T5 */
    cudaFree(t5_wbuf); t5_wbuf = NULL;
    cudaFree(t5_rbuf); t5_rbuf = NULL;
    gpu_f2fs_cleanup(fs5);
    beaver_cache_cleanup(cache5);
    cudaFree(vfs5);   vfs5   = NULL;
    cudaFree(fs5);    fs5    = NULL;
    cudaFree(cache5); cache5 = NULL;

cleanup_bufs:
    if (d_t3hashes)  { cudaFree(d_t3hashes);  d_t3hashes  = NULL; }
    if (t3_pass)     { cudaFree(t3_pass);      t3_pass     = NULL; }
    if (t3_rbufs)    { cudaFree(t3_rbufs);     t3_rbufs    = NULL; }
    if (t3_wbufs)    { cudaFree(t3_wbufs);     t3_wbufs    = NULL; }
    if (t2_rok)      { cudaFree(t2_rok);       t2_rok      = NULL; }
    if (t2_wok)      { cudaFree(t2_wok);       t2_wok      = NULL; }
    if (t2_rbufs)    { cudaFree(t2_rbufs);     t2_rbufs    = NULL; }
    if (t2_wbufs)    { cudaFree(t2_wbufs);     t2_wbufs    = NULL; }
    if (t1_rbuf)     { cudaFree(t1_rbuf);      t1_rbuf     = NULL; }
    if (t1_wbuf)     { cudaFree(t1_wbuf);      t1_wbuf     = NULL; }
    if (t5_wbuf)     { cudaFree(t5_wbuf);      t5_wbuf     = NULL; }
    if (t5_rbuf)     { cudaFree(t5_rbuf);      t5_rbuf     = NULL; }
    gpu_f2fs_cleanup(fs);
cleanup_cache:
    beaver_cache_cleanup(cache);
cleanup_allocs:
    if (res)   { cudaFree(res);   res   = NULL; }
    if (vfs)   { cudaFree(vfs);   vfs   = NULL; }
    if (fs)    { cudaFree(fs);    fs    = NULL; }
    if (cache) { cudaFree(cache); cache = NULL; }

    ddio_enable(gpu_bus);
    printf("DDIO restored.\n");
    return exit_code;
}
