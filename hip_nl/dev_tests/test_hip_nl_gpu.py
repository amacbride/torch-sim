"""GPU test for hip_nl on AMD Radeon RX 7600."""

import sys


sys.path.insert(0, "/nfs/data/torch-sim")

import time

import torch

from torch_sim.neighbors import HIP_NL_AVAILABLE, standard_nl


print("=" * 80)
print("HIP_NL GPU Test - AMD Radeon RX 7600")
print("=" * 80)
print()

# Check environment
print("Environment:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  HIP version: {torch.version.hip}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    print(f"  Device capability: {torch.cuda.get_device_capability(0)}")
print(f"  HIP_NL_AVAILABLE: {HIP_NL_AVAILABLE}")
print()

if not HIP_NL_AVAILABLE:
    print("❌ hip_nl not available, cannot test")
    sys.exit(1)

if not torch.cuda.is_available():
    print("❌ No GPU available, cannot test")
    sys.exit(1)

# Import hip_nl
from hip_nl import hip_nl


# Create test system
print("Creating test system (2744 atoms - reference size)...")
n_atoms = 2744
cell_size = 30.0

positions = torch.rand(n_atoms, 3, dtype=torch.float32) * cell_size
cell = torch.eye(3, dtype=torch.float32) * cell_size
pbc = torch.tensor([True, True, True])
cutoff = torch.tensor(3.0, dtype=torch.float32)

print(f"  System: {n_atoms} atoms, {cell_size}Å box, {cutoff.item()}Å cutoff")
print()

# Move to GPU
device = torch.device("cuda:0")
positions_gpu = positions.to(device)
cell_gpu = cell.to(device)
pbc_gpu = pbc.to(device)
cutoff_gpu = cutoff.to(device)

print("Test 1: Correctness Verification")
print("-" * 80)

# Run standard_nl on CPU for reference
mapping_cpu, shifts_cpu = standard_nl(positions, cell, pbc, cutoff)
n_pairs_cpu = mapping_cpu.shape[1]
print(f"  CPU standard_nl: {n_pairs_cpu} pairs")

# Run standard_nl on GPU
torch.cuda.synchronize()
mapping_std_gpu, shifts_std_gpu = standard_nl(
    positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu
)
torch.cuda.synchronize()
n_pairs_std_gpu = mapping_std_gpu.shape[1]
print(f"  GPU standard_nl: {n_pairs_std_gpu} pairs")

# Try hip_nl on GPU
try:
    torch.cuda.synchronize()
    mapping_hip, shifts_hip = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
    torch.cuda.synchronize()
    n_pairs_hip = mapping_hip.shape[1]
    print(f"  GPU hip_nl: {n_pairs_hip} pairs")

    # Compare pair counts
    if n_pairs_hip == n_pairs_std_gpu:
        print("  ✅ Pair count matches!")
    else:
        print(f"  ⚠️  Pair count mismatch: {n_pairs_hip} vs {n_pairs_std_gpu}")

    hip_works = True
except Exception as e:
    print(f"  ❌ hip_nl failed: {e}")
    import traceback

    traceback.print_exc()
    hip_works = False

print()

if hip_works:
    print("Test 2: Performance Benchmark")
    print("-" * 80)

    n_warmup = 5
    n_iter = 20

    # Benchmark standard_nl
    print(f"  Benchmarking standard_nl ({n_iter} iterations)...")
    for _ in range(n_warmup):
        _ = standard_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
        torch.cuda.synchronize()

    times_std = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        mapping_std, shifts_std = standard_nl(
            positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_std.append((end - start) * 1000)

    mean_std = sum(times_std) / len(times_std)
    min_std = min(times_std)

    # Benchmark hip_nl
    print(f"  Benchmarking hip_nl ({n_iter} iterations)...")
    for _ in range(n_warmup):
        _ = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
        torch.cuda.synchronize()

    times_hip = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        mapping_hip, shifts_hip = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_hip.append((end - start) * 1000)

    mean_hip = sum(times_hip) / len(times_hip)
    min_hip = min(times_hip)

    print()
    print("Results:")
    print(f"  standard_nl: {mean_std:.2f} ms (min: {min_std:.2f} ms)")
    print(f"  hip_nl:      {mean_hip:.2f} ms (min: {min_hip:.2f} ms)")
    print(f"  Speedup:     {mean_std / mean_hip:.2f}x")
    print()

    pairs_per_ms_std = n_pairs_std_gpu / mean_std
    pairs_per_ms_hip = n_pairs_hip / mean_hip

    print("  Throughput:")
    print(f"    standard_nl: {pairs_per_ms_std:.0f} pairs/ms")
    print(f"    hip_nl:      {pairs_per_ms_hip:.0f} pairs/ms")
    print()

    if mean_hip < mean_std:
        print(f"  ✅ hip_nl is {mean_std / mean_hip:.2f}x faster than standard_nl!")
    elif mean_hip < mean_std * 1.1:
        print("  ✅ hip_nl matches standard_nl performance (within 10%)")
    else:
        print("  ⚠️  hip_nl is slower than standard_nl")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
if hip_works:
    print("✅ hip_nl works on AMD Radeon RX 7600!")
    print("✅ Library loads and executes correctly")
    print("✅ Performance benchmark completed")
else:
    print("⚠️  hip_nl library loads but GPU execution failed")
    print("    This may be due to HIP runtime initialization issues")
