# hip_nl: HIP-Accelerated Neighbor Lists for TorchSim

A HIP-native neighbor list implementation optimized for AMD GPUs, providing significant performance improvements over PyTorch's standard neighbor list implementation.

## Current Status: Development/Testing

⚠️ **Important Limitation**: hip_nl currently **cannot be used in production** alongside PyTorch GPU operations due to HIP context conflicts. See [Limitations](#limitations) for details.

### What Works:
- ✅ Correctness verified against `standard_nl`
- ✅ 7-25x performance improvement over `standard_nl`
- ✅ All PBC configurations (full, mixed, none)
- ✅ Standalone testing in isolation

### What Doesn't Work Yet:
- ❌ Running alongside PyTorch GPU operations in the same process
- ❌ Production use in TorchSim simulations

## Features

- **HIP-Native**: Direct HIP kernel implementation for optimal AMD GPU performance
- **Drop-in Replacement**: Compatible with TorchSim's neighbor list API
- **Minimum Image Convention**: Proper handling of periodic boundary conditions
- **Wave-Optimized**: Leverages AMD's 64-wide wavefronts for efficient parallel execution
- **Compatibility Mode**: Option to exactly match `standard_nl` output

## Building

### Prerequisites

- ROCm toolkit (5.0 or later)
- HIP compiler (`hipcc`)
- CMake (3.16 or later)
- PyTorch with ROCm support

### Compilation

```bash
cd hip_nl
./build.sh
```

This will create `libhip_nl.so` in the `hip_nl` directory.

## Usage

### Environment Variables

hip_nl requires specific environment variables:

```tcsh
# Required for gfx1102 hardware (adjust for your GPU)
setenv HSA_OVERRIDE_GFX_VERSION 11.0.0

# Required to enable hip_nl in torchsim_nl()
setenv USE_HIP_NL 1
```

### As a Standalone Function

```python
import torch
from hip_nl import hip_nl, HIP_NL_AVAILABLE

if HIP_NL_AVAILABLE:
    positions = torch.rand(1000, 3) * 10.0
    cell = torch.eye(3) * 10.0
    pbc = torch.tensor([True, True, True])
    cutoff = torch.tensor(3.0)

    # compatible_mode=True (default) matches standard_nl exactly
    mapping, shifts = hip_nl(positions, cell, pbc, cutoff)

    # compatible_mode=False returns all valid pairs (more correct)
    mapping, shifts = hip_nl(positions, cell, pbc, cutoff, compatible_mode=False)

    print(f"Found {mapping.shape[1]} pairs")
```

### Integration with TorchSim

When `USE_HIP_NL=1` is set, `torchsim_nl()` automatically uses hip_nl:

```python
from torch_sim.neighbors import torchsim_nl

# Automatically uses hip_nl if USE_HIP_NL=1 and on ROCm
mapping, shifts = torchsim_nl(positions, cell, pbc, cutoff)
```

## Testing

hip_nl tests must be run in isolation (not as part of the main test suite):

```tcsh
# Set environment
setenv HSA_OVERRIDE_GFX_VERSION 11.0.0
setenv USE_HIP_NL 1

# Run tests
python -m pytest hip_nl/dev_tests/test_hip_nl.py -v
```

## Performance

### Benchmark Results

| System Size | standard_nl | hip_nl | Speedup |
|-------------|-------------|--------|---------|
| 216 atoms   | 6.46 ms     | 0.25 ms | **25.6x** |
| 512 atoms   | 8.87 ms     | 0.57 ms | **15.5x** |
| 1000 atoms  | 15.84 ms    | 1.04 ms | **15.2x** |
| 2744 atoms  | 41.75 ms    | 5.66 ms | **7.4x** |

### Running Benchmarks

```tcsh
setenv HSA_OVERRIDE_GFX_VERSION 11.0.0

# Stage 1: Benchmark standard_nl
python hip_nl/dev_tests/benchmark_comparison.py --stage standard

# Stage 2: Benchmark hip_nl (in fresh shell)
setenv USE_HIP_NL 1
python hip_nl/dev_tests/benchmark_comparison.py --stage hip_nl

# Compare results
python hip_nl/dev_tests/benchmark_comparison.py --compare
```

## Limitations

### Critical: PyTorch HIP Context Conflict

hip_nl uses a standalone HIP context loaded via `ctypes` with `RTLD_GLOBAL`. This conflicts with PyTorch's HIP backend:

```
hip_nl loads HIP libraries → Creates standalone HIP context
                                    ↓
PyTorch HIP backend fails: "invalid device function" or "error 100"
```

**Impact**: You cannot use hip_nl and PyTorch GPU operations in the same Python process. This means:
- TorchSim simulations requiring GPU-accelerated models (MACE, etc.) cannot currently use hip_nl
- hip_nl can only be tested in isolation

### Other Limitations

- Currently supports only "full" neighbor lists (not half lists)
- Requires CPU tensor conversion for memory management
- No cell list optimization (O(n²) scaling)

## Known Bug in standard_nl

hip_nl discovered a bug in `standard_nl` where valid neighbor pairs are missed at non-periodic boundaries. Details:

- **Affected**: Non-periodic boundary conditions (`pbc=False` on any dimension)
- **Symptom**: Valid pairs within cutoff are not returned
- **Cause**: Cell-list partitioning incorrectly skips pairs at boundary cells
- **Workaround**: hip_nl's `compatible_mode=True` (default) replicates this behavior for compatibility

When `compatible_mode=False`, hip_nl returns all valid pairs, which is more correct but differs from `standard_nl`.

## Future Work

### Path to Production Use

To enable hip_nl in production TorchSim simulations, it must be reimplemented to share PyTorch's HIP context:

1. **PyTorch C++ Extension**: Rewrite as a PyTorch extension using `torch.utils.cpp_extension`
2. **HIP Interop**: Use PyTorch's HIP stream and memory allocator
3. **Custom Op Registration**: Register as a custom PyTorch operator

### Other Improvements

1. **Cell list algorithm**: For O(n) scaling with large systems
2. **Half neighbor lists**: Reduce redundant pair computation
3. **Batch processing**: Handle multiple systems simultaneously
4. **Direct GPU memory**: Eliminate CPU round-trips

## Algorithm

Uses a brute-force pairwise comparison approach:

- **Parallelization**: Wave-level parallelism with each wave processing one reference atom
- **Memory**: Shared memory caching of box matrices to minimize global memory traffic
- **Optimization**: Wave ballot voting for efficient valid pair identification

## License

BSD-3-Clause (same as TorchSim)

## References

Based on the vesin neighbor list library by Guillaume Fraux:
- https://github.com/Luthaf/vesin
