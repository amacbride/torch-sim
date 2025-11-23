"""Setup script for hip_torch_nl PyTorch C++ Extension.

Build with:
    cd hip_torch_nl
    pip install -e .

Or build in-place:
    python setup.py build_ext --inplace
"""

import os
from pathlib import Path

# This is a HIP extension for AMD GPUs - configure environment for ROCm build
# If CUDA_HOME points to a non-existent path (common on ROCm-only systems), clear it
if "CUDA_HOME" in os.environ:
    cuda_home = os.environ["CUDA_HOME"]
    if not os.path.exists(cuda_home) or not os.path.exists(os.path.join(cuda_home, "bin")):
        del os.environ["CUDA_HOME"]

# Find and set ROCM_HOME
rocm_paths = ["/opt/rocm", "/opt/rocm-7.0.1", "/opt/rocm-6.0.0"]
for rocm_path in rocm_paths:
    if os.path.exists(rocm_path):
        os.environ["ROCM_HOME"] = rocm_path
        # PyTorch's CUDAExtension uses CUDA_HOME - point it to ROCm for hipcc
        if "CUDA_HOME" not in os.environ:
            os.environ["CUDA_HOME"] = rocm_path
        break

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
here = Path(__file__).parent.absolute()

# Source files - use .cu extension for PyTorch auto-hipification
sources = [
    str(here / "csrc" / "hip_neighborlist.cpp"),
    str(here / "csrc" / "hip_neighborlist_kernel.cu"),
]

# Extra compile args - keep simple, let PyTorch handle hipification
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],  # ROCm treats this as hipcc flags
}

setup(
    name="hip_torch_nl",
    version="0.1.0",
    description="HIP-accelerated neighbor list for PyTorch (shares PyTorch HIP context)",
    author="TorchSim Contributors",
    ext_modules=[
        CUDAExtension(
            name="hip_torch_nl._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["hip_torch_nl"],
    package_dir={"hip_torch_nl": "."},
    python_requires=">=3.9",
)
