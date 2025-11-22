"""Setup script for hip_torch_nl PyTorch C++ Extension.

Build with:
    cd hip_torch_nl
    pip install -e .

Or build in-place:
    python setup.py build_ext --inplace
"""

import os
from pathlib import Path

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
