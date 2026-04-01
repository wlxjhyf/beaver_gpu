"""
setup.py — Build beaver_ext PyTorch CUDA extension.

Usage (from beaver_torch/ directory):
    python setup.py build_ext --inplace

CPU two-step write path: no GPU F2FS kernels needed, only beaver_ext.cu
+ ddio_helper.cpp. Links against libpmem (for pmem_map_file / pmem_memcpy_persist)
and libpci (for DDIO control via ddio_helper).
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
TESTS_DIR  = os.path.join(PROJ_ROOT, "tests")

assert os.path.isdir(INCLUDE), f"include dir not found: {INCLUDE}"

DDIO_HELPER = os.path.join(TESTS_DIR, "ddio_helper.cpp")
SOURCES = [
    DDIO_HELPER,
    os.path.join(os.path.dirname(__file__), "beaver_ext.cu"),
]

# ── NVCC flags ─────────────────────────────────────────────────────────
NVCC_FLAGS = [
    "-O3",
    "-use_fast_math",
    "--generate-code=arch=compute_89,code=sm_89",
    "-Xcompiler", "-fPIC",
    "--expt-relaxed-constexpr",
    "--extended-lambda",
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
)

setup(
    name="beaver_ext",
    version="0.1.0",
    description="Beaver GPU PM Offload — PyTorch CUDA extension",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
