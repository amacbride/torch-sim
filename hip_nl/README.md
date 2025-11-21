# hip_nl: HIP-Accelerated Neighbor Lists for TorchSim

A HIP-native neighbor list implementation optimized for AMD GPUs, providing a drop-in replacement for TorchSim's neighbor list computations.

## Features

- **HIP-Native**: Direct HIP kernel implementation for optimal AMD GPU performance
- **Drop-in Replacement**: Compatible with TorchSim's neighbor list API
- **Minimum Image Convention**: Proper handling of periodic boundary conditions
- **Wave-Optimized**: Leverages AMD's 64-wide wavefronts for efficient parallel execution

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

### As a Standalone Function

```python
import torch
from hip_nl import hip_nl, HIP_NL_AVAILABLE

if HIP_NL_AVAILABLE:
    # Create test system
    positions = torch.rand(1000, 3, device='cuda')
    cell = torch.eye(3, device='cuda') * 10.0
    pbc = torch.tensor([True, True, True], device='cuda')
    cutoff = torch.tensor(3.0, device='cuda')

    # Compute neighbor list
    mapping, shifts = hip_nl(positions, cell, pbc, cutoff)

    print(f"Found {mapping.shape[1]} pairs")
```

### Integration with TorchSim

The `hip_nl` function can be integrated into TorchSim's neighbor list selection:

```python
from torch_sim.neighbors import torchsim_nl

# torchsim_nl will automatically use hip_nl if available on AMD GPUs
mapping, shifts = torchsim_nl(positions, cell, pbc, cutoff)
```

## Performance

Designed for microsecond-scale molecular dynamics simulations on datacenter AMD GPUs. Target performance is parity or better than PyTorch's standard neighbor list implementation.

### Benchmarks

Run benchmarks with:

```bash
python benchmark_nl_performance.py
```

## Algorithm

Uses a brute-force pairwise comparison approach similar to vesin's CUDA implementation:

- **Parallelization**: Wave-level parallelism with each wave processing one reference atom
- **Memory**: Shared memory caching of box matrices to minimize global memory traffic
- **Optimization**: Wave ballot voting for efficient valid pair identification

## Limitations

- Currently supports only "full" neighbor lists (not half lists)
- Requires conversion to/from CPU for memory management (will be optimized)
- No cell list optimization yet (future work for large systems)

## Future Improvements

1. **Zero-copy GPU memory**: Direct GPU memory management without CPU transfer
2. **Cell list algorithm**: For better O(n) scaling with large systems
3. **Half neighbor lists**: Reduce redundant pair computation
4. **Batch processing**: Handle multiple systems simultaneously

## License

BSD-3-Clause (same as TorchSim)

## References

Based on the vesin neighbor list library by Guillaume Fraux:
- https://github.com/Luthaf/vesin
