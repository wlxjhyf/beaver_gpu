/*
 * microbench.cu — Microbenchmark main driver.
 *
 * Runs three workloads × three systems (Basic / cuFile / beaver_gpu-COW)
 * across 7 thread counts, printing one table per workload.
 *
 * Storage:
 *   Basic + cuFile : /mnt/pmem0  (ext4-dax on /dev/pmem0, fsdax mode)
 *   beaver         : /dev/dax1.0 (devdax, byte-addressable via GPM)
 *
 * Run: sudo ./benchmarks/microbench
 * (root needed for DDIO control + devdax access)
 */

#include "bench_config.h"
#include "bench_utils.h"
#include "gpm_interface.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

/* Declarations from baseline translation units */
int    basic_init(void);
void   basic_cleanup(void);
double basic_run(mb_workload_t wl, uint32_t nthreads);

int    pmem_bench_init(void);
void   pmem_bench_cleanup(void);
double pmem_run(mb_workload_t wl, uint32_t nthreads);

int    cufile_init(void);
void   cufile_cleanup(void);
double cufile_run(mb_workload_t wl, uint32_t nthreads);

int    beaver_bench_init(void);
void   beaver_bench_cleanup(void);
double beaver_run(mb_workload_t wl, uint32_t nthreads);

/* DDIO helpers from tests/ddio_helper.cpp */
extern "C" {
uint8_t ddio_get_gpu_bus(void);
void    ddio_disable(uint8_t bus);
void    ddio_enable (uint8_t bus);
}

/* ── Preflight: check /mnt/pmem0 is accessible ───────────────────── */

static int check_pmem_mount(void)
{
    struct stat st;
    if (stat(MB_PMEM_MOUNT, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr,
                "[microbench] ERROR: %s is not a directory.\n"
                "  Mount ext4-dax before running:\n"
                "    sudo mkfs.ext4 /dev/pmem0\n"
                "    sudo mkdir -p %s\n"
                "    sudo mount -o dax /dev/pmem0 %s\n",
                MB_PMEM_MOUNT, MB_PMEM_MOUNT, MB_PMEM_MOUNT);
        return -1;
    }
    return 0;
}

/* ── Run one workload across all thread counts ───────────────────── */

static void run_workload(mb_workload_t wl,
                          int basic_ok, int pmem_ok, int cufile_ok)
{
    printf("\n### %s ###\n", MB_WORKLOAD_NAMES[wl]);
    printf("  %u writes/thread × %u B  —  timing: task-start to task-complete\n",
           MB_WRITES_PER_THREAD, MB_PAGE_SIZE);
    mb_print_table_header();

    for (uint32_t ti = 0; ti < MB_NT_COUNT; ti++) {
        uint32_t nt = MB_THREAD_COUNTS[ti];

        double r_basic  = basic_ok  ? basic_run (wl, nt) : -1.0;
        double r_pmem   = pmem_ok   ? pmem_run  (wl, nt) : -1.0;
        double r_cufile = cufile_ok ? cufile_run(wl, nt) : -1.0;
        double r_beaver =             beaver_run(wl, nt);

        mb_print_row(nt, r_basic, r_pmem, r_cufile, r_beaver);
        fflush(stdout);
    }
}

/* ── main ────────────────────────────────────────────────────────── */

int main(void)
{
    printf("============================================================\n");
    printf("   beaver_gpu  Microbenchmark  (GoFS §4.1/4.2 aligned)\n");
    printf("============================================================\n");

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU  : %s  (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("PM   : %s (devdax/beaver)  %s (fsdax/Basic+cuFile)\n",
           GPM_DEVDAX_PATH, MB_PMEM_MOUNT);

    /* DDIO disable (required for GPU→PM writes) */
    uint8_t gpu_bus = ddio_get_gpu_bus();
    printf("PCIe bus 0x%02x — disabling DDIO\n\n", gpu_bus);
    ddio_disable(gpu_bus);

    /* GPM init (beaver path) */
    if (gpm_init() != GPM_SUCCESS) {
        fprintf(stderr, "[microbench] gpm_init failed\n");
        ddio_enable(gpu_bus);
        return 1;
    }

    /* Check /mnt/pmem0 */
    int pmem_ok = (check_pmem_mount() == 0);

    /* Init each baseline */
    int basic_ok  = pmem_ok && (basic_init()       == 0);
    int pmem_b_ok = pmem_ok && (pmem_bench_init()  == 0);
    int cufile_ok = pmem_ok && (cufile_init()       == 0);
    int beaver_ok = (beaver_bench_init()            == 0);

    if (!basic_ok)
        fprintf(stdout, "[microbench] pwrite+fsync baseline unavailable (see above).\n");
    if (!pmem_b_ok)
        fprintf(stdout, "[microbench] pmem_persist baseline unavailable (see above).\n");
    if (!cufile_ok)
        fprintf(stdout, "[microbench] cuFile baseline unavailable (see above).\n");
    if (!beaver_ok) {
        fprintf(stderr, "[microbench] beaver init failed — aborting.\n");
        goto done;
    }

    printf("\nWorkload parameters:\n");
    printf("  Threads          : ");
    for (uint32_t i = 0; i < MB_NT_COUNT; i++)
        printf("%u%s", MB_THREAD_COUNTS[i], i+1 < MB_NT_COUNT ? ", " : "\n");
    printf("  Data/thread      : %u writes × %u B = %u KiB\n",
           MB_WRITES_PER_THREAD, MB_PAGE_SIZE, MB_DATA_PER_THREAD / 1024);
    printf("  Metric           : aggregate write throughput (MB/s)\n");
    printf("  pwrite+fsync     : pwrite() to ext4-dax + fdatasync() → PM\n");
    printf("  pmem_persist     : pmem_memcpy_persist() direct to PM (no FS)\n");
    printf("  cuFile           : cuFileWrite() GDS path → ext4-dax → PM\n");
    printf("  Beaver           : GPU warp-cooperative COW → PM (kernel→sync)\n");

    /* Run the three workloads */
    run_workload(MB_SEQ,   basic_ok, pmem_b_ok, cufile_ok);
    run_workload(MB_RAND,  basic_ok, pmem_b_ok, cufile_ok);
    run_workload(MB_MULTI, basic_ok, pmem_b_ok, cufile_ok);

    printf("\n============================================================\n");
    printf("   Microbenchmark complete.\n");
    printf("============================================================\n");

done:
    if (basic_ok)  basic_cleanup();
    if (pmem_b_ok) pmem_bench_cleanup();
    if (cufile_ok) cufile_cleanup();
    beaver_bench_cleanup();

    ddio_enable(gpu_bus);
    printf("DDIO restored.\n");
    return 0;
}
