/*
 * rawgpm_bench.cu — Raw GPU→PM bandwidth benchmark (阶段A).
 *
 * Reproduces GPM-ASPLOS22 Figure 3 methodology on local hardware,
 * extended with dual-PM sections for two devdax devices:
 *   dax1.0  (NUMA 1, Socket 1): P1-DIMMA2 + P1-DIMMD2  interleaved
 *   dax0.0  (NUMA 0, Socket 2): P2-DIMMA2 + P2-DIMMD2  interleaved
 *
 *   (a) CPU→PM single dax1.0 : N CPU threads, each writes 1GB/N via
 *                               pmem_memcpy_persist.
 *   (b) GPU→PM single dax1.0 : N GPU warps, each writes 1GB/N via
 *                               volatile stores + gpm_drain.
 *   (c) CPU→PM dual dax1.0+0 : N CPU threads split N/2 per device,
 *                               total 2 GB, aggregate throughput.
 *   (d) GPU→PM dual dax1.0+0 : N GPU warps split N/2 per device,
 *                               two PM channels written concurrently.
 *
 * Run: sudo ./benchmarks/rawgpm_bench
 */

#include "gpm_interface.cuh"
#include <cuda_runtime.h>
#include <libpmem.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ── Configuration ──────────────────────────────────────────────── */

#define RAWGPM_PAGE_SIZE    4096u
#define RAWGPM_TOTAL_BYTES  (1ULL * 1024 * 1024 * 1024)   /* 1 GB per device */

#define GPM_DEVDAX_PATH2  "/dev/dax0.0"   /* second PM device (NUMA 0) */

#define CPU_MAX_THREADS   48u
#define CPU_NT_COUNT      7u
static const uint32_t CPU_NT[CPU_NT_COUNT] = {1, 2, 4, 8, 16, 32, 48};

#define GPU_NW_COUNT  8u
static const uint32_t GPU_NW[GPU_NW_COUNT] = {1, 2, 4, 8, 16, 32, 48, 128};

#define RAWGPM_REPS  3u

/* ── GPU Kernels: single PM ─────────────────────────────────────── */

static __device__ __forceinline__ void pm_store8_volatile(void *dst, unsigned long long v)
{
    *(volatile unsigned long long *)dst = v;
}

static __device__ __forceinline__ void pm_store16_cs(void *dst, uint4 v)
{
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        : : "l"((unsigned long long)dst),
            "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

/*
 * Write-through store primitives for Ada Lovelace (sm_89).
 *
 * On Ada, volatile stores (st.global.wb) go through the L2 write-back cache.
 * Multiple warps competing for L2 eviction bandwidth cap aggregate GPU→PM
 * throughput at ~3.2 GB/s regardless of warp count.
 *
 * st.global.wt (write-through) writes to L1 and propagates immediately to
 * memory without L2 eviction rate limiting the throughput.  This should
 * allow GPU bandwidth to scale with warp count on Ada, recovering parity
 * with CPU MOVNTI (which bypasses all cache levels entirely).
 */
static __device__ __forceinline__ void pm_store8_wt(void *dst, unsigned long long v)
{
    asm volatile(
        "st.global.wt.u64 [%0], %1;"
        : : "l"((unsigned long long)dst), "l"(v)
        : "memory");
}

static __device__ __forceinline__ void pm_store16_wt(void *dst, uint4 v)
{
    asm volatile(
        "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
        : : "l"((unsigned long long)dst),
            "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__global__ void k_pm_8b(void *pm_base, uint32_t n_writers, const void *src,
                         uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    char *dst = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;
    const unsigned long long *s64 = (const unsigned long long *)src;
    const uint32_t dwords_per_page = RAWGPM_PAGE_SIZE / 8u;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        unsigned long long *d = (unsigned long long *)(dst + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t i = lane; i < dwords_per_page; i += blockDim.x)
            pm_store8_volatile(d + i, s64[i % (RAWGPM_PAGE_SIZE / 8)]);
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_cs(void *pm_base, uint32_t n_writers, const uint4 *src4,
                         uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t i = lane; i < N4; i += blockDim.x)
            pm_store16_cs(d + i, src4[i % N4]);
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_cs64(void *pm_base, uint32_t n_writers, const uint4 *src4,
                           uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t base = 0; base < N4; base += 4 * blockDim.x) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; k++)
                pm_store16_cs(d + base + lane + k * blockDim.x,
                              src4[(base + lane + k * blockDim.x) % N4]);
        }
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_cs128(void *pm_base, uint32_t n_writers, const uint4 *src4,
                            uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++)
            pm_store16_cs(d + lane + k * blockDim.x,
                          src4[(lane + k * blockDim.x) % N4]);
    }
    if (lane == 0) gpm_drain();
}

/* ── GPU Kernels: write-through variants (Ada L2 bypass) ────────── */
/*
 * k_pm_wt8: 8-byte write-through stores, one per warp lane per iteration.
 *   Should bypass Ada's L2 write-back path, allowing bandwidth to scale
 *   with warp count unlike the volatile/cs variants.
 */
__global__ void k_pm_wt8(void *pm_base, uint32_t n_writers, const void *src,
                          uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    char *dst = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;
    const unsigned long long *s64 = (const unsigned long long *)src;
    const uint32_t dwords_per_page = RAWGPM_PAGE_SIZE / 8u;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        unsigned long long *d = (unsigned long long *)(dst + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t i = lane; i < dwords_per_page; i += blockDim.x)
            pm_store8_wt(d + i, s64[i % (RAWGPM_PAGE_SIZE / 8)]);
    }
    if (lane == 0) gpm_drain();
}

/*
 * k_pm_wt128: 16-byte write-through stores, 8 per lane = 128B per iteration.
 *   Higher store throughput per warp compared to wt8.
 */
__global__ void k_pm_wt128(void *pm_base, uint32_t n_writers, const uint4 *src4,
                            uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    if (writer >= n_writers) return;

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)writer * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++)
            pm_store16_wt(d + lane + k * blockDim.x,
                          src4[(lane + k * blockDim.x) % N4]);
    }
    if (lane == 0) gpm_drain();
}

/* ── GPU Kernels: dual PM ───────────────────────────────────────── */
/*
 * Each block selects its target PM device based on blockIdx.x:
 *   blockIdx.x < n_per_dev  → pm0 (dax1.0, NUMA 1)
 *   blockIdx.x >= n_per_dev → pm1 (dax0.0, NUMA 0)
 * Total blocks launched = 2 × n_per_dev.
 * Both PM channels are written concurrently in the same kernel launch.
 */

__global__ void k_pm_dual_8b(void *pm0, void *pm1, uint32_t n_per_dev,
                              const void *src, uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    void    *pm_base;
    uint32_t local_w;
    if (writer < n_per_dev) { pm_base = pm0; local_w = writer; }
    else                     { pm_base = pm1; local_w = writer - n_per_dev; }

    char *dst = (char *)pm_base + (uint64_t)local_w * pages_per_writer * RAWGPM_PAGE_SIZE;
    const unsigned long long *s64 = (const unsigned long long *)src;
    const uint32_t dwords_per_page = RAWGPM_PAGE_SIZE / 8u;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        unsigned long long *d = (unsigned long long *)(dst + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t i = lane; i < dwords_per_page; i += blockDim.x)
            pm_store8_volatile(d + i, s64[i % (RAWGPM_PAGE_SIZE / 8)]);
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_dual_cs(void *pm0, void *pm1, uint32_t n_per_dev,
                              const uint4 *src4, uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    void    *pm_base;
    uint32_t local_w;
    if (writer < n_per_dev) { pm_base = pm0; local_w = writer; }
    else                     { pm_base = pm1; local_w = writer - n_per_dev; }

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)local_w * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t i = lane; i < N4; i += blockDim.x)
            pm_store16_cs(d + i, src4[i % N4]);
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_dual_cs64(void *pm0, void *pm1, uint32_t n_per_dev,
                                const uint4 *src4, uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    void    *pm_base;
    uint32_t local_w;
    if (writer < n_per_dev) { pm_base = pm0; local_w = writer; }
    else                     { pm_base = pm1; local_w = writer - n_per_dev; }

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)local_w * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        for (uint32_t base = 0; base < N4; base += 4 * blockDim.x) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; k++)
                pm_store16_cs(d + base + lane + k * blockDim.x,
                              src4[(base + lane + k * blockDim.x) % N4]);
        }
    }
    if (lane == 0) gpm_drain();
}

__global__ void k_pm_dual_cs128(void *pm0, void *pm1, uint32_t n_per_dev,
                                 const uint4 *src4, uint32_t pages_per_writer)
{
    uint32_t writer = blockIdx.x;
    uint32_t lane   = threadIdx.x;
    void    *pm_base;
    uint32_t local_w;
    if (writer < n_per_dev) { pm_base = pm0; local_w = writer; }
    else                     { pm_base = pm1; local_w = writer - n_per_dev; }

    const uint32_t N4 = RAWGPM_PAGE_SIZE / 16u;
    char *dst_base = (char *)pm_base + (uint64_t)local_w * pages_per_writer * RAWGPM_PAGE_SIZE;

    for (uint32_t p = 0; p < pages_per_writer; p++) {
        uint4 *d = (uint4 *)(dst_base + (uint64_t)p * RAWGPM_PAGE_SIZE);
        #pragma unroll
        for (uint32_t k = 0; k < 8; k++)
            pm_store16_cs(d + lane + k * blockDim.x,
                          src4[(lane + k * blockDim.x) % N4]);
    }
    if (lane == 0) gpm_drain();
}

/* ── NUMA cpuset helpers ─────────────────────────────────────────── */

#include <sched.h>

/*
 * Parse /sys/devices/system/node/nodeX/cpulist (e.g. "0-11,24-35")
 * into a cpu_set_t.  Missing file → empty set (no binding).
 */
static void parse_cpulist(const char *path, cpu_set_t *set)
{
    CPU_ZERO(set);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char buf[512];
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return; }
    fclose(f);

    char *p = buf;
    while (*p && *p != '\n') {
        int lo = (int)strtol(p, &p, 10);
        if (*p == '-') {
            ++p;
            int hi = (int)strtol(p, &p, 10);
            for (int c = lo; c <= hi; c++) CPU_SET(c, set);
        } else {
            CPU_SET(lo, set);
        }
        if (*p == ',') ++p;
    }
}

/* Per-NUMA-node cpusets, filled once in main */
static cpu_set_t g_node_cpuset[2];
static int       g_cpuset_valid = 0;

static void init_numa_cpusets(void)
{
    parse_cpulist("/sys/devices/system/node/node0/cpulist", &g_node_cpuset[0]);
    parse_cpulist("/sys/devices/system/node/node1/cpulist", &g_node_cpuset[1]);
    g_cpuset_valid = (CPU_COUNT(&g_node_cpuset[0]) > 0 &&
                      CPU_COUNT(&g_node_cpuset[1]) > 0);
    if (g_cpuset_valid)
        printf("NUMA cpusets: node0=%d CPUs  node1=%d CPUs\n",
               CPU_COUNT(&g_node_cpuset[0]), CPU_COUNT(&g_node_cpuset[1]));
    else
        fprintf(stderr, "NUMA cpusets: parse failed, CPU binding disabled\n");
}

/* ── CPU multi-thread PM writer ──────────────────────────────────── */

typedef struct {
    void       *pm_dst;
    const void *src;
    size_t      len;
    int         numa_node;   /* 0 or 1 → bind to that node; -1 → no bind */
} cpu_worker_arg_t;

static void *cpu_pm_worker(void *arg)
{
    cpu_worker_arg_t *a = (cpu_worker_arg_t *)arg;
    /* Bind to the NUMA node that owns the target PM device */
    if (g_cpuset_valid && a->numa_node >= 0 && a->numa_node <= 1)
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                               &g_node_cpuset[a->numa_node]);
    pmem_memcpy_persist(a->pm_dst, a->src, a->len);
    return NULL;
}

/* ── Timing helpers ──────────────────────────────────────────────── */

typedef struct { struct timespec t0; } hrtimer_t;
static void   ht_start(hrtimer_t *t) { clock_gettime(CLOCK_MONOTONIC, &t->t0); }
static double ht_elapsed_ms(const hrtimer_t *t) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec  - t->t0.tv_sec)  * 1000.0
         + (double)(t1.tv_nsec - t->t0.tv_nsec) / 1.0e6;
}

/* Single device: 1 GB / elapsed */
static double throughput_mbps(double ms) {
    if (ms <= 0.0) return -1.0;
    return (double)RAWGPM_TOTAL_BYTES / (ms / 1000.0) / (1024.0 * 1024.0);
}

/* Dual device: 2 GB / elapsed (aggregate across both PM channels) */
static double dual_throughput_mbps(double ms) {
    if (ms <= 0.0) return -1.0;
    return 2.0 * (double)RAWGPM_TOTAL_BYTES / (ms / 1000.0) / (1024.0 * 1024.0);
}

#define WARP_SIZE 32u
static void grid_for_writers(uint32_t n_writers, uint32_t *g, uint32_t *b) {
    *b = WARP_SIZE;
    *g = n_writers;
}

#define TIMED_KERNEL(...)                                          \
    ([&]() -> double {                                             \
        double _best = 1e18;                                       \
        for (uint32_t _r = 0; _r < RAWGPM_REPS; _r++) {          \
            hrtimer_t _t; ht_start(&_t);                          \
            __VA_ARGS__;                                           \
            cudaDeviceSynchronize();                               \
            double _ms = ht_elapsed_ms(&_t);                      \
            if (_ms < _best) _best = _ms;                         \
        }                                                          \
        return _best;                                              \
    }())

/* ── DDIO helpers ────────────────────────────────────────────────── */

extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ── main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("================================================================\n");
    printf("   Raw GPM Bandwidth Benchmark  (Fig.3 + Dual-PM extension)\n");
    printf("================================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU        : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("PM dev 0   : %s  (NUMA 1, Socket 1)\n", GPM_DEVDAX_PATH);
    printf("PM dev 1   : %s  (NUMA 0, Socket 2)\n", GPM_DEVDAX_PATH2);
    printf("Total data : %llu MB per device  (%llu MB dual aggregate)\n",
           (unsigned long long)(RAWGPM_TOTAL_BYTES >> 20),
           (unsigned long long)(RAWGPM_TOTAL_BYTES >> 19));
    printf("Reps       : %u  (min elapsed)\n\n", RAWGPM_REPS);

    /* NUMA cpusets for thread affinity binding */
    init_numa_cpusets();

    uint8_t bus = ddio_get_gpu_bus();
    printf("PCIe bus 0x%02x — disabling DDIO\n\n", bus);
    ddio_disable(bus);

    if (gpm_init() != GPM_SUCCESS) {
        fprintf(stderr, "gpm_init failed\n");
        ddio_enable(bus);
        return 1;
    }

    /* ── Locals ─────────────────────────────────────────────────── */
    uint8_t *d_src       = NULL;
    void    *h_src       = NULL;
    void    *pm1_mmap    = NULL;   /* full dax0.0 mmap               */
    size_t   pm1_mmap_len= 0;
    void    *pm1_addr    = NULL;   /* 1 GB slice registered with CUDA */
    int      pm1_is_pmem = 0;
    int      pm1_registered = 0;

    if (cudaMalloc((void **)&d_src, RAWGPM_PAGE_SIZE) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_src failed\n"); goto done;
    }
    cudaMemset(d_src, 0xAB, RAWGPM_PAGE_SIZE);

    h_src = malloc(RAWGPM_PAGE_SIZE);
    if (!h_src) { fprintf(stderr, "malloc h_src failed\n"); goto done; }
    memset(h_src, 0xAB, RAWGPM_PAGE_SIZE);

    /* ── Allocate 1 GB on dax1.0 (via gpm_alloc) ────────────────── */
    {
        gpm_region_t region;
        memset(&region, 0, sizeof region);
        if (gpm_alloc(&region, RAWGPM_TOTAL_BYTES, "rawgpm-dev0") != GPM_SUCCESS) {
            fprintf(stderr, "gpm_alloc dax1.0 (1GB) failed\n"); goto done;
        }
        printf("PM0 region : %llu MB at %p  (dax1.0)\n\n",
               (unsigned long long)(RAWGPM_TOTAL_BYTES >> 20), region.addr);

        /* ── (a) GPU VRAM→CPU DRAM→PM  single dax1.0 ───────────────
         * Two-step CPU-path alternative (data originates on GPU):
         *   Step 1: cudaMemcpy GPU VRAM → CPU DRAM  (single transfer,
         *           models bringing results back from GPU to host)
         *   Step 2: N CPU threads pmem_memcpy_persist CPU DRAM → PM
         *           (N threads is the varying parameter, same as before)
         * Total time covers both steps end-to-end.
         *
         * DDIO ON: default state for CPU-path operation; PCIe writes
         * land in CPU L3. pmem_memcpy_persist handles flush+drain.
         * ─────────────────────────────────────────────────────────── */
        ddio_enable(bus);
        printf("ddio: DDIO re-enabled for section (a) [CPU-path, DDIO on]\n");
        printf("=== (a) GPU VRAM→CPU DRAM→PM  single dax1.0  [CPU-path alternative] ===\n");
        printf("  Step1=cudaMemcpy(GPU→CPU DRAM)  Step2=N threads pmem_memcpy_persist→PM\n");
        printf("  %-12s  %10s  %12s  %8s\n",
               "CPU Threads", "Per-th MB", "MB/s", "Speedup");
        printf("  %-12s  %10s  %12s  %8s\n",
               "------------", "----------", "------------", "--------");

        double ref1_mbps = -1.0;

        /* 1 GB source buffer in GPU VRAM + 1 GB intermediate CPU DRAM buffer */
        void *d_src_large = NULL;
        void *cpu_buf     = NULL;
        if (cudaMalloc(&d_src_large, RAWGPM_TOTAL_BYTES) != cudaSuccess) {
            fprintf(stderr, "  cudaMalloc d_src_large (1GB) failed — skipping (a)\n");
            goto skip_a;
        }
        cudaMemset(d_src_large, 0xAB, RAWGPM_TOTAL_BYTES);
        cpu_buf = malloc(RAWGPM_TOTAL_BYTES);
        if (!cpu_buf) {
            fprintf(stderr, "  malloc cpu_buf (1GB) failed — skipping (a)\n");
            cudaFree(d_src_large); d_src_large = NULL;
            goto skip_a;
        }
        memset(cpu_buf, 0, RAWGPM_TOTAL_BYTES);

        for (uint32_t ti = 0; ti < CPU_NT_COUNT; ti++) {
            uint32_t nt  = CPU_NT[ti];
            size_t   per = RAWGPM_TOTAL_BYTES / nt;

            cpu_worker_arg_t *args = (cpu_worker_arg_t *)malloc(nt * sizeof(*args));
            pthread_t        *tids = (pthread_t *)malloc(nt * sizeof(pthread_t));
            if (!args || !tids) { free(args); free(tids); break; }

            for (uint32_t i = 0; i < nt; i++) {
                args[i].pm_dst    = (char *)region.addr + (size_t)i * per;
                args[i].src       = (char *)cpu_buf     + (size_t)i * per;
                args[i].len       = per;
                args[i].numa_node = 1;   /* dax1.0 is NUMA 1 */
            }

            /* warm-up: step1 + step2 */
            cudaMemcpy(cpu_buf, d_src_large, RAWGPM_TOTAL_BYTES, cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < nt; i++)
                pthread_create(&tids[i], NULL, cpu_pm_worker, &args[i]);
            for (uint32_t i = 0; i < nt; i++)
                pthread_join(tids[i], NULL);

            double best_ms = 1e18;
            for (uint32_t r = 0; r < RAWGPM_REPS; r++) {
                hrtimer_t t; ht_start(&t);
                /* step 1: GPU VRAM → CPU DRAM */
                cudaMemcpy(cpu_buf, d_src_large, RAWGPM_TOTAL_BYTES,
                           cudaMemcpyDeviceToHost);
                /* step 2: CPU DRAM → PM  (pmem_memcpy_persist includes flush+drain) */
                for (uint32_t i = 0; i < nt; i++)
                    pthread_create(&tids[i], NULL, cpu_pm_worker, &args[i]);
                for (uint32_t i = 0; i < nt; i++)
                    pthread_join(tids[i], NULL);
                double ms = ht_elapsed_ms(&t);
                if (ms < best_ms) best_ms = ms;
            }

            double mbps = throughput_mbps(best_ms);
            if (ti == 0) ref1_mbps = mbps;
            double speedup = (ref1_mbps > 0) ? mbps / ref1_mbps : -1.0;
            printf("  %-12u  %10.1f  %12.1f  %7.2fx\n",
                   nt, (double)per / (1024.0 * 1024.0), mbps, speedup);
            fflush(stdout);
            free(args); free(tids);
        }

        free(cpu_buf);
        cudaFree(d_src_large);
        d_src_large = NULL;
        skip_a:;
        ddio_disable(bus);
        printf("ddio: DDIO re-disabled for section (b) [GPU kernel stores]\n\n");

        /* ── (b) GPU→PM single dax1.0 ───────────────────────────── */
        printf("\n=== (b) GPU→PM  single dax1.0  [kernel volatile/cs/wt stores] ===\n");
        printf("  (ref: 1-stream cudaMemcpy GPU→PM: %.1f MB/s)\n", ref1_mbps);
        printf("  %-12s  %10s  %12s  %12s  %12s  %12s  %12s  %12s\n",
               "Writers", "Per-wr KB", "vol/8B", "cs/16B", "cs/64B", "cs/128B",
               "wt/8B", "wt/128B");
        printf("  %-12s  %10s  %12s  %12s  %12s  %12s  %12s  %12s\n",
               "------------", "----------",
               "------------", "------------", "------------", "------------",
               "------------", "------------");

        for (uint32_t ti = 0; ti < GPU_NW_COUNT; ti++) {
            uint32_t nw = GPU_NW[ti];
            uint64_t total_pages  = RAWGPM_TOTAL_BYTES / RAWGPM_PAGE_SIZE;
            uint32_t pages_per_wr = (uint32_t)(total_pages / nw);
            if (pages_per_wr == 0) pages_per_wr = 1;
            uint32_t active_nw    = (uint32_t)(total_pages / pages_per_wr);
            if (active_nw > nw) active_nw = nw;

            uint32_t g, b;
            grid_for_writers(active_nw, &g, &b);

            /* warm-up */
            k_pm_cs128<<<g, b>>>(region.addr, active_nw,
                                  (const uint4 *)d_src, pages_per_wr);
            cudaDeviceSynchronize();

            double ms_vol   = TIMED_KERNEL(
                k_pm_8b<<<g, b>>>(region.addr, active_nw, d_src, pages_per_wr));
            double ms_cs    = TIMED_KERNEL(
                k_pm_cs<<<g, b>>>(region.addr, active_nw,
                                  (const uint4 *)d_src, pages_per_wr));
            double ms_cs64  = TIMED_KERNEL(
                k_pm_cs64<<<g, b>>>(region.addr, active_nw,
                                    (const uint4 *)d_src, pages_per_wr));
            double ms_cs128 = TIMED_KERNEL(
                k_pm_cs128<<<g, b>>>(region.addr, active_nw,
                                     (const uint4 *)d_src, pages_per_wr));
            double ms_wt8   = TIMED_KERNEL(
                k_pm_wt8<<<g, b>>>(region.addr, active_nw, d_src, pages_per_wr));
            double ms_wt128 = TIMED_KERNEL(
                k_pm_wt128<<<g, b>>>(region.addr, active_nw,
                                     (const uint4 *)d_src, pages_per_wr));

            printf("  %-12u  %10u  %12.1f  %12.1f  %12.1f  %12.1f  %12.1f  %12.1f\n",
                   active_nw, pages_per_wr * RAWGPM_PAGE_SIZE / 1024u,
                   throughput_mbps(ms_vol),  throughput_mbps(ms_cs),
                   throughput_mbps(ms_cs64), throughput_mbps(ms_cs128),
                   throughput_mbps(ms_wt8),  throughput_mbps(ms_wt128));
            fflush(stdout);
        }

        /* ── (b2) GPU→PM single dax1.0 [8 threads per writer] ──────
         * Same as (b) but blockDim = 8 instead of 32.
         * Only vol/8B and cs/16B are run — the unrolled cs128/wt128
         * kernels require blockDim.x == 32 to cover a full page.
         * ─────────────────────────────────────────────────────────── */
        printf("\n=== (b2) GPU→PM  single dax1.0  [8 threads per writer] ===\n");
        printf("  %-12s  %10s  %12s  %12s\n",
               "Writers", "Per-wr KB", "vol/8B", "cs/16B");
        printf("  %-12s  %10s  %12s  %12s\n",
               "------------", "----------", "------------", "------------");

        for (uint32_t ti = 0; ti < GPU_NW_COUNT; ti++) {
            uint32_t nw = GPU_NW[ti];
            uint64_t total_pages  = RAWGPM_TOTAL_BYTES / RAWGPM_PAGE_SIZE;
            uint32_t pages_per_wr = (uint32_t)(total_pages / nw);
            if (pages_per_wr == 0) pages_per_wr = 1;
            uint32_t active_nw    = (uint32_t)(total_pages / pages_per_wr);
            if (active_nw > nw) active_nw = nw;

            uint32_t g = active_nw, b = 8u;   /* 8 threads per writer */

            /* warm-up */
            k_pm_8b<<<g, b>>>(region.addr, active_nw, d_src, pages_per_wr);
            cudaDeviceSynchronize();

            double ms_vol = TIMED_KERNEL(
                k_pm_8b<<<g, b>>>(region.addr, active_nw, d_src, pages_per_wr));
            double ms_cs  = TIMED_KERNEL(
                k_pm_cs<<<g, b>>>(region.addr, active_nw,
                                  (const uint4 *)d_src, pages_per_wr));

            printf("  %-12u  %10u  %12.1f  %12.1f\n",
                   active_nw, pages_per_wr * RAWGPM_PAGE_SIZE / 1024u,
                   throughput_mbps(ms_vol), throughput_mbps(ms_cs));
            fflush(stdout);
        }

        /* ── Map dax0.0 for dual-PM sections ────────────────────── */
        pm1_mmap = pmem_map_file(GPM_DEVDAX_PATH2, 0, 0, 0666,
                                  &pm1_mmap_len, &pm1_is_pmem);
        if (!pm1_mmap) {
            fprintf(stderr, "\n[dual] pmem_map_file(%s) failed: %s\n"
                            "       Skipping sections (c) and (d).\n",
                    GPM_DEVDAX_PATH2, pmem_errormsg());
            goto single_done;
        }
        printf("\nPM1 mapped : %.1f GB at %p  is_pmem=%d  (%s)\n",
               (double)pm1_mmap_len / (1024.0*1024.0*1024.0),
               pm1_mmap, pm1_is_pmem, GPM_DEVDAX_PATH2);

        /* Register first 1 GB of dax0.0 with CUDA */
        pm1_addr = pm1_mmap;
        cudaError_t cerr;
        cerr = cudaHostRegister(pm1_addr, RAWGPM_TOTAL_BYTES, 0);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "[dual] cudaHostRegister dax0.0 failed: %s\n"
                            "       Skipping sections (c) and (d).\n",
                    cudaGetErrorString(cerr));
            goto single_done;
        }
        pm1_registered = 1;
        printf("PM1 region : %llu MB registered with CUDA at %p\n\n",
               (unsigned long long)(RAWGPM_TOTAL_BYTES >> 20), pm1_addr);

        /* ── (c) CPU→PM dual dax1.0 + dax0.0 ───────────────────── */
        /*
         * Threads 0..nt/2-1 write to dax1.0 (NUMA 1).
         * Threads nt/2..nt-1 write to dax0.0 (NUMA 0).
         * Total data = 2 × 1 GB.  Throughput = aggregate MB/s.
         * Note: NUMA-aware results require numactl binding per socket.
         */
        printf("=== (c) CPU→PM  dual dax1.0+dax0.0  [N/2 per device] ===\n");
        printf("  %-12s  %10s  %12s  %12s\n",
               "CPU Threads", "N/dev", "Agg MB/s", "vs single");
        printf("  %-12s  %10s  %12s  %12s\n",
               "------------", "----------", "------------", "------------");

        for (uint32_t ti = 0; ti < CPU_NT_COUNT; ti++) {
            uint32_t nt = CPU_NT[ti];
            if (nt < 2) {
                printf("  %-12u  %10s  %12s  %12s\n", nt, "N/A", "N/A", "(need ≥2)");
                continue;
            }
            uint32_t n_per_dev = nt / 2;
            size_t   per       = RAWGPM_TOTAL_BYTES / n_per_dev;

            void *h_buf = malloc(2 * RAWGPM_TOTAL_BYTES);
            if (!h_buf) { fprintf(stderr, "  malloc failed\n"); break; }
            memset(h_buf, 0xAB, 2 * RAWGPM_TOTAL_BYTES);

            cpu_worker_arg_t *args = (cpu_worker_arg_t *)malloc(nt * sizeof(*args));
            pthread_t *tids = (pthread_t *)malloc(nt * sizeof(pthread_t));
            if (!args || !tids) { free(h_buf); free(args); free(tids); break; }

            for (uint32_t i = 0; i < n_per_dev; i++) {
                /* first half → dax1.0 (NUMA 1) */
                args[i].pm_dst    = (char *)region.addr + (size_t)i * per;
                args[i].src       = (char *)h_buf        + (size_t)i * per;
                args[i].len       = per;
                args[i].numa_node = 1;
            }
            for (uint32_t i = 0; i < n_per_dev; i++) {
                /* second half → dax0.0 (NUMA 0) */
                uint32_t j = n_per_dev + i;
                args[j].pm_dst    = (char *)pm1_addr + (size_t)i * per;
                args[j].src       = (char *)h_buf + RAWGPM_TOTAL_BYTES + (size_t)i * per;
                args[j].len       = per;
                args[j].numa_node = 0;
            }

            /* warm-up */
            for (uint32_t i = 0; i < nt; i++)
                pthread_create(&tids[i], NULL, cpu_pm_worker, &args[i]);
            for (uint32_t i = 0; i < nt; i++)
                pthread_join(tids[i], NULL);

            double best_ms = 1e18;
            for (uint32_t r = 0; r < RAWGPM_REPS; r++) {
                hrtimer_t t; ht_start(&t);
                for (uint32_t i = 0; i < nt; i++)
                    pthread_create(&tids[i], NULL, cpu_pm_worker, &args[i]);
                for (uint32_t i = 0; i < nt; i++)
                    pthread_join(tids[i], NULL);
                double ms = ht_elapsed_ms(&t);
                if (ms < best_ms) best_ms = ms;
            }

            double agg = dual_throughput_mbps(best_ms);
            printf("  %-12u  %10u  %12.1f  %11.2fx\n",
                   nt, n_per_dev, agg, (ref1_mbps > 0) ? agg / ref1_mbps : -1.0);
            fflush(stdout);
            free(h_buf); free(args); free(tids);
        }

        /* ── (d) GPU→PM dual dax1.0 + dax0.0 ───────────────────── */
        /*
         * Total writers = 2 × n_per_dev launched as one kernel.
         * n_per_dev warps → dax1.0, n_per_dev warps → dax0.0.
         * Both PM channels written concurrently within one cudaDeviceSynchronize.
         * Aggregate = 2 GB / elapsed.
         */
        printf("\n=== (d) GPU→PM  dual dax1.0+dax0.0  [N/2 warps per device] ===\n");
        printf("  %-12s  %10s  %12s  %12s  %12s  %12s\n",
               "Tot writers", "N/dev", "vol/8B", "cs/16B", "cs/64B", "cs/128B");
        printf("  %-12s  %10s  %12s  %12s  %12s  %12s\n",
               "------------", "----------",
               "------------", "------------", "------------", "------------");

        for (uint32_t ti = 0; ti < GPU_NW_COUNT; ti++) {
            uint32_t nw = GPU_NW[ti];
            if (nw < 2) {
                printf("  %-12u  %10s  %12s  %12s  %12s  %12s\n",
                       nw, "N/A", "N/A", "N/A", "N/A", "(need ≥2)");
                continue;
            }
            uint32_t n_per_dev = nw / 2;

            uint64_t total_pages  = RAWGPM_TOTAL_BYTES / RAWGPM_PAGE_SIZE;
            uint32_t pages_per_wr = (uint32_t)(total_pages / n_per_dev);
            if (pages_per_wr == 0) pages_per_wr = 1;

            /* warm-up both devices */
            k_pm_dual_cs128<<<2*n_per_dev, WARP_SIZE>>>(
                region.addr, pm1_addr, n_per_dev,
                (const uint4 *)d_src, pages_per_wr);
            cudaDeviceSynchronize();

            double ms_vol = TIMED_KERNEL(
                k_pm_dual_8b<<<2*n_per_dev, WARP_SIZE>>>(
                    region.addr, pm1_addr, n_per_dev, d_src, pages_per_wr));
            double ms_cs = TIMED_KERNEL(
                k_pm_dual_cs<<<2*n_per_dev, WARP_SIZE>>>(
                    region.addr, pm1_addr, n_per_dev,
                    (const uint4 *)d_src, pages_per_wr));
            double ms_cs64 = TIMED_KERNEL(
                k_pm_dual_cs64<<<2*n_per_dev, WARP_SIZE>>>(
                    region.addr, pm1_addr, n_per_dev,
                    (const uint4 *)d_src, pages_per_wr));
            double ms_cs128 = TIMED_KERNEL(
                k_pm_dual_cs128<<<2*n_per_dev, WARP_SIZE>>>(
                    region.addr, pm1_addr, n_per_dev,
                    (const uint4 *)d_src, pages_per_wr));

            printf("  %-12u  %10u  %12.1f  %12.1f  %12.1f  %12.1f\n",
                   2*n_per_dev, n_per_dev,
                   dual_throughput_mbps(ms_vol),
                   dual_throughput_mbps(ms_cs),
                   dual_throughput_mbps(ms_cs64),
                   dual_throughput_mbps(ms_cs128));
            fflush(stdout);
        }

single_done:
        gpm_free(&region);
    }

    printf("\n================================================================\n");
    printf("   Raw GPM Benchmark complete.\n");
    printf("================================================================\n");

done:
    if (pm1_registered) cudaHostUnregister(pm1_addr);
    if (pm1_mmap)       pmem_unmap(pm1_mmap, pm1_mmap_len);
    if (d_src) cudaFree(d_src);
    if (h_src) free(h_src);
    ddio_enable(bus);
    printf("DDIO restored.\n");
    return 0;
}
