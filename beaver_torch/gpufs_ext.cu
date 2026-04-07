/*
 * gpufs_ext.cu — PyTorch CUDA Extension for GPUfs I/O
 *
 * Wraps GPUfs gopen/gwrite/gread/gclose (__device__ functions) into a
 * Python-callable C++ interface.
 *
 * Write path per call (gpufs_write):
 *   1. Copy path string to device memory
 *   2. Launch gpufs_write_kernel (async, on GPUfs kernelStream)
 *      — kernel: each block opens file, writes its FS_BLOCKSIZE chunks, closes
 *   3. run_gpufs_handler() — CPU poll loop serving GPU I/O requests (blocks
 *      until kernelStream completes)
 *   4. cudaDeviceSynchronize() — ensure async_close DMAs complete
 *   5. async_close_loop() — drain remaining async-close page writes (pwrite)
 *
 * Read path per call (gpufs_read): symmetric to write.
 *
 * GPUGlobals (IPC queues + 2 GB page pool) is created once in gpufs_init()
 * and reused across all write/read calls. Call gpufs_cleanup() at the end.
 *
 * IMPORTANT: call gpufs_init() BEFORE any PyTorch GPU operations so that
 *   cudaSetDeviceFlags(cudaDeviceMapHost) succeeds before the CUDA context
 *   is created by PyTorch.
 */

/* ── 1. Standard headers first ──────────────────────────────────────── */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <string>

/* ── 2. GPUfs headers ────────────────────────────────────────────────── */
/* Order matters: fs_debug before everything, host_loop brings in
   fs_initializer (GPUGlobals + initializer decl) and run_gpufs_handler,
   fs_calls adds the __device__ gopen/gwrite/gread/gclose declarations. */
#include "fs_debug.cu.h"
#include "host_loop.h"
#include "fs_calls.cu.h"

/* ── 3. PyTorch extension header (after CUDA / GPUfs) ────────────────── */
#include <torch/extension.h>

/* ─────────────────────────────────────────────────────────────────────── */
/*  Global GPUfs state (init once, reuse across write/read calls)          */
/* ─────────────────────────────────────────────────────────────────────── */
static volatile GPUGlobals* g_globals = nullptr;
static int g_device = 0;

/* Number of blocks / threads for the write/read kernels.
   copy_block in GPUfs is most efficient with >= 256 threads per block. */
#define GPUFS_NBLOCKS   16
#define GPUFS_NTHREADS  256

/* ─────────────────────────────────────────────────────────────────────── */
/*  GPU kernels                                                             */
/* ─────────────────────────────────────────────────────────────────────── */

/*
 * gpufs_write_kernel
 *
 * Each block opens the same destination file (GPUfs handles ref-counting),
 * writes a strided range of FS_BLOCKSIZE (4 KB) chunks, then closes.
 * All threads within a block cooperate on each gwrite call via copy_block().
 */
__global__ void gpufs_write_kernel(const char* d_path,
                                   uchar* gpu_src,
                                   size_t total_size)
{
    __shared__ int fd;
    BEGIN_SINGLE_THREAD
        fd = gopen(d_path, O_GWRONCE);
    END_SINGLE_THREAD

    if (fd < 0) return;

    for (size_t off = (size_t)blockIdx.x * FS_BLOCKSIZE;
         off < total_size;
         off += (size_t)gridDim.x * FS_BLOCKSIZE)
    {
        size_t chunk = (total_size - off < (size_t)FS_BLOCKSIZE)
                       ? (total_size - off)
                       : (size_t)FS_BLOCKSIZE;
        gwrite(fd, off, chunk, gpu_src + off);
    }

    BEGIN_SINGLE_THREAD
        gclose(fd);
    END_SINGLE_THREAD
}

/*
 * gpufs_read_kernel — symmetric to write.
 */
__global__ void gpufs_read_kernel(const char* d_path,
                                  uchar* gpu_dst,
                                  size_t total_size)
{
    __shared__ int fd;
    BEGIN_SINGLE_THREAD
        fd = gopen(d_path, O_GRDONLY);
    END_SINGLE_THREAD

    if (fd < 0) return;

    for (size_t off = (size_t)blockIdx.x * FS_BLOCKSIZE;
         off < total_size;
         off += (size_t)gridDim.x * FS_BLOCKSIZE)
    {
        size_t chunk = (total_size - off < (size_t)FS_BLOCKSIZE)
                       ? (total_size - off)
                       : (size_t)FS_BLOCKSIZE;
        gread(fd, off, chunk, gpu_dst + off);
    }

    BEGIN_SINGLE_THREAD
        gclose(fd);
    END_SINGLE_THREAD
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  Host C++ API                                                            */
/* ─────────────────────────────────────────────────────────────────────── */

void gpufs_init()
{
    /* cudaSetDeviceFlags must precede CUDA context creation.
       Ignore cudaErrorSetOnActiveProcess: PyTorch may have already called it
       with cudaDeviceMapHost (required for cudaHostAllocMapped). */
    cudaError_t flag_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flag_err != cudaSuccess &&
        flag_err != cudaErrorSetOnActiveProcess) {
        throw std::runtime_error(
            std::string("gpufs_init: cudaSetDeviceFlags failed: ") +
            cudaGetErrorString(flag_err));
    }

    CUDA_SAFE_CALL(cudaGetDevice(&g_device));
    initializer(&g_globals);
}

/*
 * gpufs_write — write GPU tensor data to PM via GPUfs.
 *
 * h_path  : host-side file path (must be < FILENAME_SIZE = 64 bytes)
 * gpu_ptr : device pointer (from tensor.data_ptr())
 * size    : byte count
 */
void gpufs_write(const char* h_path, void* gpu_ptr, size_t size)
{
    if (!g_globals)
        throw std::runtime_error("gpufs_write: call gpufs_init() first");

    size_t path_len = strnlen(h_path, FILENAME_SIZE);
    if (path_len >= FILENAME_SIZE)
        throw std::runtime_error(
            std::string("gpufs_write: path too long (max ") +
            std::to_string(FILENAME_SIZE - 1) + " chars): " + h_path);

    /* Copy path to device */
    char* d_path;
    CUDA_SAFE_CALL(cudaMalloc(&d_path, path_len + 1));
    CUDA_SAFE_CALL(cudaMemcpy(d_path, h_path, path_len + 1,
                              cudaMemcpyHostToDevice));

    /* Launch kernel asynchronously on GPUfs's dedicated stream */
    gpufs_write_kernel<<<GPUFS_NBLOCKS, GPUFS_NTHREADS, 0,
                         g_globals->streamMgr->kernelStream>>>(
        d_path, (uchar*)gpu_ptr, size);

    /* CPU handler: serves GPU I/O requests until kernelStream completes */
    run_gpufs_handler(g_globals, g_device);

    /* Ensure all async-close DMAs (on async_close_stream) are flushed */
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    /* Drain any async-close page writes whose DMA just completed */
    async_close_loop(g_globals);

    CUDA_SAFE_CALL(cudaFree(d_path));
}

/*
 * gpufs_read — read PM data into GPU tensor via GPUfs.
 */
void gpufs_read(const char* h_path, void* gpu_ptr, size_t size)
{
    if (!g_globals)
        throw std::runtime_error("gpufs_read: call gpufs_init() first");

    size_t path_len = strnlen(h_path, FILENAME_SIZE);
    if (path_len >= FILENAME_SIZE)
        throw std::runtime_error(
            std::string("gpufs_read: path too long (max ") +
            std::to_string(FILENAME_SIZE - 1) + " chars): " + h_path);

    char* d_path;
    CUDA_SAFE_CALL(cudaMalloc(&d_path, path_len + 1));
    CUDA_SAFE_CALL(cudaMemcpy(d_path, h_path, path_len + 1,
                              cudaMemcpyHostToDevice));

    gpufs_read_kernel<<<GPUFS_NBLOCKS, GPUFS_NTHREADS, 0,
                        g_globals->streamMgr->kernelStream>>>(
        d_path, (uchar*)gpu_ptr, size);

    run_gpufs_handler(g_globals, g_device);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaFree(d_path));
}

/*
 * gpufs_cleanup — release GPUGlobals (frees 2 GB page pool, IPC queues, etc.)
 */
void gpufs_cleanup()
{
    if (g_globals) {
        delete g_globals;
        g_globals = nullptr;
    }
}

/* ─────────────────────────────────────────────────────────────────────── */
/*  PyTorch pybind11 binding                                                */
/* ─────────────────────────────────────────────────────────────────────── */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &gpufs_init,
          "Initialize GPUfs global state (call before any PyTorch GPU ops)");

    m.def("write",
          [](py::bytes path, int64_t ptr, int64_t size) {
              std::string p = path;
              gpufs_write(p.c_str(), reinterpret_cast<void*>(ptr),
                          static_cast<size_t>(size));
          },
          py::arg("path"), py::arg("ptr"), py::arg("size"),
          "Write GPU tensor to PM file via GPUfs RPC");

    m.def("read",
          [](py::bytes path, int64_t ptr, int64_t size) {
              std::string p = path;
              gpufs_read(p.c_str(), reinterpret_cast<void*>(ptr),
                         static_cast<size_t>(size));
          },
          py::arg("path"), py::arg("ptr"), py::arg("size"),
          "Read PM file into GPU tensor via GPUfs RPC");

    m.def("cleanup", &gpufs_cleanup,
          "Release GPUfs global state");
}
