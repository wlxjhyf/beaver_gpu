"""
setup.py — Build beaver_ext PyTorch CUDA extension.

Usage (from beaver_torch/ directory):
    pip install -e .

This extension links against src/core/ for Beaver COW infrastructure:
  - gpm_interface.cu: PM allocation (gpm_init, gpm_alloc)
  - beaver_cow.cu: COW cache management (beaver_cache_init, beaver_dram_pool_init)

The extension itself (beaver_ext.cu) is a pure glue layer that bridges
PyTorch tensors to the Beaver COW backend.
"""

import os
import torch.utils.cpp_extension as _cpp_ext

# CUDA 13.0 toolkit vs PyTorch built with CUDA 12.8: major-version mismatch.
# Safe to suppress: extension links against PyTorch's bundled libcudart.
_cpp_ext._check_cuda_version = lambda *a, **kw: None

from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

# ── Paths ──────────────────────────────────────────────────────────────
PROJ_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCLUDE    = os.path.join(PROJ_ROOT, "include")
SRC_CORE   = os.path.join(PROJ_ROOT, "src", "core")
TESTS_DIR  = os.path.join(PROJ_ROOT, "tests")

assert os.path.isdir(INCLUDE), f"include dir not found: {INCLUDE}"
assert os.path.isdir(SRC_CORE), f"src/core dir not found: {SRC_CORE}"

# ── Source files ───────────────────────────────────────────────────────
# beaver_ext.cu: PyTorch glue layer (this extension)
# ddio_helper.cpp: DDIO control wrapper
# gpm_interface.cu: PM allocation (gpm_init, gpm_alloc, gpm_free)
# beaver_cow.cu: COW cache management (beaver_cache_init, beaver_dram_pool_init)
SOURCES = [
    os.path.join(os.path.dirname(__file__), "beaver_ext.cu"),
    os.path.join(TESTS_DIR, "ddio_helper.cpp"),
    os.path.join(SRC_CORE, "gpm_interface.cu"),
    os.path.join(SRC_CORE, "beaver_cow.cu"),
]

# ── NVCC flags ─────────────────────────────────────────────────────────
# -dc: Generate relocatable device code (required for separate compilation)
# --extended-lambda: Allow __device__ lambdas
# --expt-relaxed-constexpr: Allow constexpr in device code
NVCC_FLAGS = [
    "-O3",
    "-use_fast_math",
    "--generate-code=arch=compute_89,code=sm_89",
    "-Xcompiler", "-fPIC",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
    "-dc",  # Relocatable device code for separate compilation
]

# ── RPATH ──────────────────────────────────────────────────────────────
import torch
TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
CUDA_LIB  = "/usr/local/cuda/lib64"

# ── Extension ───────────────────────────────────────────────────────────
ext = CUDAExtension(
    name="beaver_ext",
    sources=SOURCES,
    include_dirs=[INCLUDE, "/usr/local/cuda/include"],
    libraries=["pmem", "pci"],
    library_dirs=["/usr/lib/x86_64-linux-gnu", "/usr/local/lib"],
    extra_link_args=[
        f"-Wl,-rpath,{TORCH_LIB}",
        f"-Wl,-rpath,{CUDA_LIB}",
    ],
    extra_compile_args={
        "cxx":  ["-O3", "-fPIC"],
        "nvcc": NVCC_FLAGS,
    },
    # Enable device linking for separate compilation
    dlink=True,
)

setup(
    name="beaver_ext",
    version="0.2.0",
    description="Beaver GPU PM Offload — PyTorch CUDA extension (COW backend)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
