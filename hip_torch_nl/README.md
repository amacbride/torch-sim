# hip_torch_nl: PyTorch Extension for HIP Neighbor Lists

A PyTorch C++ Extension version of hip_nl that integrates with PyTorch's HIP context, solving the context conflict issue that prevents the standalone hip_nl from being used alongside PyTorch GPU operations.

## Key Difference from hip_nl

| Feature | hip_nl | hip_torch_nl |
|---------|--------|--------------|
| HIP Context | Standalone (via ctypes) | Shared with PyTorch |
| Memory Management | Manual hipMalloc | PyTorch allocator |
| PyTorch Compatibility | ❌ Conflicts | ✅ Compatible |
| Production Use | ❌ Not possible | ✅ Possible |

## Building

### Prerequisites

- ROCm toolkit (5.0 or later)
- PyTorch with ROCm support
- CMake (for PyTorch extension building)

### Installation

```bash
cd hip_torch_nl

# Set environment for gfx1102 hardware (adjust for your GPU)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Install in development mode
pip install -e .

# Or build in-place without installing
python setup.py build_ext --inplace
```

## Usage

```python
import torch
from hip_torch_nl import hip_torch_nl, HIP_TORCH_NL_AVAILABLE

if HIP_TORCH_NL_AVAILABLE:
    # Create tensors on GPU
    positions = torch.rand(1000, 3, device='cuda') * 10.0
    cell = torch.eye(3, device='cuda') * 10.0
    pbc = torch.tensor([True, True, True], device='cuda')
    cutoff = torch.tensor(3.0)

    # Compute neighbor list
    mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

    print(f"Found {mapping.shape[1]} pairs")

    # Can now use other PyTorch GPU operations without conflict!
    forces = torch.zeros_like(positions)
    # ... compute forces using mapping ...
```

## Integration with TorchSim

Once verified working, hip_torch_nl can be integrated into `torchsim_nl()` as the preferred backend for AMD GPUs:

```python
# In torch_sim/neighbors.py
def torchsim_nl(...):
    if torch.version.hip is not None and HIP_TORCH_NL_AVAILABLE:
        return hip_torch_nl(positions, cell, pbc, cutoff, sort_id)
    # ... other backends ...
```

## Architecture

```
hip_torch_nl/
├── __init__.py           # Python wrapper with compatibility mode
├── setup.py              # PyTorch extension build configuration
├── csrc/
│   ├── hip_neighborlist.cpp      # C++ binding layer (pybind11)
│   └── hip_neighborlist_kernel.hip  # HIP kernel (same algorithm as hip_nl)
└── dev_tests/
    └── test_hip_torch_nl.py      # Tests
```

## How It Works

1. **Shared HIP Context**: By compiling as a PyTorch C++ Extension using `torch.utils.cpp_extension.CUDAExtension`, the HIP kernel runs in PyTorch's HIP context instead of creating a separate one.

2. **PyTorch Memory Allocator**: Output tensors are allocated using PyTorch's `torch::empty()`, which uses the same memory pool as all other PyTorch operations.

3. **Stream Integration**: The kernel launches on PyTorch's current CUDA/HIP stream via `at::cuda::getCurrentCUDAStream()`, ensuring proper synchronization.

## Comparison with hip_nl

The HIP kernel code is nearly identical to hip_nl. The key changes are:

- Removed `extern "C"` wrapper and manual `hipMalloc`/`hipMemcpy`
- Use PyTorch tensor API for memory allocation
- Use PyTorch's stream for kernel launch
- Return PyTorch tensors directly

## License

BSD-3-Clause (same as TorchSim)
