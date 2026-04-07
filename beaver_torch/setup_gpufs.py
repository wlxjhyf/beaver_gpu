"""
setup_gpufs.py — Build gpufs_ext PyTorch CUDA extension.

Usage (from beaver_torch/ directory):
    python setup_gpufs.py build_ext --inplace

Compiles GPUfs source files DIRECTLY (not linking libgpufs.a) to avoid
fat-binary hash conflicts between pre-compiled objects and our extension.
All .cu files are compiled together in the same build so NVCC generates
consistent __fatbinwrap symbols.

run_gpufs_handler() is header-only (from host_loop.h), compiled into
gpufs_ext.cu's translation unit.
"""

import os
import torch.utils.cpp_extension as _cpp_ext

# CUDA 13.0 toolkit vs PyTorch built with CUDA 12.8: major-version mismatch.
_cpp_ext._check_cuda_version = lambda *a, **kw: None

from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

# ── Paths ──────────────────────────────────────────────────────────────
PROJ_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GPUFS_ROOT     = os.path.join(PROJ_ROOT, "third-party", "gpufs")
GPUFS_INC      = os.path.join(GPUFS_ROOT, "include")
GPUFS_SRC      = os.path.join(GPUFS_ROOT, "libgpufs")      # .cu source files
GPUFS_CON_SRC  = os.path.join(GPUFS_ROOT, "gpufs_con_lib.user")  # con lib

NVTX_INC = os.path.join(
    os.path.expanduser("~"),
    "env/torch/lib/python3.13/site-packages/nvidia/nvtx/include"
)

assert os.path.isdir(GPUFS_INC), f"GPUfs include dir not found: {GPUFS_INC}"
assert os.path.isdir(GPUFS_SRC), f"GPUfs source dir not found: {GPUFS_SRC}"

# ── RPATH ──────────────────────────────────────────────────────────────
import torch
TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
CUDA_LIB  = "/usr/local/cuda/lib64"

# ── Sources ─────────────────────────────────────────────────────────────
# Our extension + all GPUfs .cu source files compiled together in one build.
# Timer and con-lib are CPU-only (.cpp), compiled as C++ by NVCC.
def _gpufs(name):
    return os.path.join(GPUFS_SRC, name)

def _con(name):
    return os.path.join(GPUFS_CON_SRC, name)

SOURCES = [
    os.path.join(os.path.dirname(__file__), "gpufs_ext.cu"),
    # GPUfs CUDA sources
    _gpufs("cpu_ipc.cu"),
    _gpufs("fs_calls.cu"),
    _gpufs("fs_initializer.cu"),
    _gpufs("fs_structures.cu"),
    _gpufs("hashMap.cu"),
    _gpufs("mallocfree.cu"),
    _gpufs("fs_debug.cu"),
    _gpufs("async_ipc.cu"),
    _gpufs("generic_ringbuf.cu"),
    # GPUfs CPU-only sources
    _gpufs("timer.cpp"),
    _con("gpufs_con_lib.cpp"),
]

# ── NVCC flags ─────────────────────────────────────────────────────────
# -dc              : relocatable device code (required for cross-TU device calls)
# -maxrregcount 32 : match GPUfs original compilation, prevents register
#                    spills in deep GPUfs device call chains
NVCC_FLAGS = [
    "-O3",
    "--generate-code=arch=compute_89,code=sm_89",
    "-Xcompiler", "-fPIC",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-dc",
    "-maxrregcount", "32",
    "-DDEBUG_NOINLINE=",   # GPUfs requires this macro; empty = no noinline attr
    "-DNVTX_SUPPRESS_V2_DEPRECATION_WARNING",
]

# ── Extension ──────────────────────────────────────────────────────────
ext = CUDAExtension(
    name="gpufs_ext",
    sources=SOURCES,
    include_dirs=[
        GPUFS_INC,
        GPUFS_SRC,       # libgpufs sources include each other via relative paths
        GPUFS_CON_SRC,
        "/usr/local/cuda/include",
        NVTX_INC,
    ],
    library_dirs=[
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ],
    libraries=["cudart", "pthread"],
    extra_link_args=[
        f"-Wl,-rpath,{TORCH_LIB}",
        f"-Wl,-rpath,{CUDA_LIB}",
    ],
    extra_compile_args={
        "cxx":  ["-O3", "-fPIC",
                 "-DDEBUG_NOINLINE=",
                 "-DNVTX_SUPPRESS_V2_DEPRECATION_WARNING"],
        "nvcc": NVCC_FLAGS,
    },
    dlink=True,  # device link step to resolve cross-TU __device__ calls
)

setup(
    name="gpufs_ext",
    version="0.1.0",
    description="GPUfs PyTorch CUDA extension (GPU→CPU RPC I/O to PM)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
