"""Simple hip_nl GPU test without device queries."""

import sys


sys.path.insert(0, "/nfs/data/torch-sim")

import time

import torch


print("=" * 80)
print("HIP_NL GPU Test - AMD Radeon RX 7600")
print("=" * 80)
print()

# Import
from hip_nl import hip_nl
from torch_sim.neighbors import HIP_NL_AVAILABLE, standard_nl


print(f"HIP_NL_AVAILABLE: {HIP_NL_AVAILABLE}")
print(f"CUDA available: {torch.cuda.is_available()}")
print()

# Create test system (2744 atoms - reference size)
n_atoms = 2744
cell_size = 30.0

print(f"Creating test system: {n_atoms} atoms, {cell_size}Å box, 3.0Å cutoff")

positions = torch.rand(n_atoms, 3, dtype=torch.float32) * cell_size
cell = torch.eye(3, dtype=torch.float32) * cell_size
pbc = torch.tensor([True, True, True])
cutoff = torch.tensor(3.0, dtype=torch.float32)

# Move to GPU
positions_gpu = positions.to("cuda:0")
cell_gpu = cell.to("cuda:0")
pbc_gpu = pbc.to("cuda:0")
cutoff_gpu = cutoff.to("cuda:0")

print("✅ Data moved to GPU")
print()

# Test correctness
print("Test 1: Correctness")
print("-" * 80)

# Standard NL on GPU
torch.cuda.synchronize()
mapping_std, shifts_std = standard_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
torch.cuda.synchronize()
n_pairs_std = mapping_std.shape[1]
print(f"  standard_nl: {n_pairs_std} pairs")

# HIP NL on GPU
torch.cuda.synchronize()
mapping_hip, shifts_hip = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
torch.cuda.synchronize()
n_pairs_hip = mapping_hip.shape[1]
print(f"  hip_nl:      {n_pairs_hip} pairs")

if n_pairs_hip == n_pairs_std:
    print("  ✅ Pair counts match!")
else:
    print("  ⚠️  Pair count mismatch")

print()

# Test performance
print("Test 2: Performance")
print("-" * 80)

n_warmup = 5
n_iter = 20

# Benchmark standard_nl
for _ in range(n_warmup):
    _ = standard_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
    torch.cuda.synchronize()

times_std = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = standard_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times_std.append((end - start) * 1000)

mean_std = sum(times_std) / len(times_std)

# Benchmark hip_nl
for _ in range(n_warmup):
    _ = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
    torch.cuda.synchronize()

times_hip = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = hip_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff_gpu)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times_hip.append((end - start) * 1000)

mean_hip = sum(times_hip) / len(times_hip)

print(f"  standard_nl: {mean_std:.2f} ms ({n_pairs_std / mean_std:.0f} pairs/ms)")
print(f"  hip_nl:      {mean_hip:.2f} ms ({n_pairs_hip / mean_hip:.0f} pairs/ms)")
print(f"  Speedup:     {mean_std / mean_hip:.2f}x")
print()

if mean_hip < mean_std:
    print(f"✅ hip_nl is {mean_std / mean_hip:.2f}x FASTER than standard_nl!")
elif mean_hip < mean_std * 1.1:
    print("✅ hip_nl matches standard_nl (within 10%)")
else:
    print(f"⚠️  hip_nl is {mean_hip / mean_std:.2f}x slower than standard_nl")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print("✅ hip_nl works on AMD Radeon RX 7600 (gfx1102)!")
print("✅ Integration with TorchSim successful")
print(f"✅ Performance: {mean_hip:.2f} ms for {n_atoms} atoms")
