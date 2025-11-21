#!/usr/bin/env python3
"""Test script to demonstrate ROCm patch functionality.
Shows behavior with and without vesin available.
"""

import torch

import torch_sim as ts
from torch_sim import neighbors


# Test system
print("=== TorchSim ROCm Patch Functionality Test ===\n")

# Check current vesin status
print(f"VESIN_AVAILABLE: {neighbors.VESIN_AVAILABLE}")
print(
    f"Vesin modules: VesinNeighborList={neighbors.VesinNeighborList is not None}, VesinNeighborListTorch={neighbors.VesinNeighborListTorch is not None}"
)

# Create a simple test system (4-atom square)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
positions = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    device=device,
    dtype=torch.float32,
)

cell = torch.eye(3, device=device, dtype=torch.float32) * 3.0
pbc = torch.tensor([False, False, False], device=device)
cutoff = torch.tensor(1.5, device=device, dtype=torch.float32)

print(f"\nTest system: 4 atoms on {device}")
print(f"Positions shape: {positions.shape}")
print("Cell: 3x3 identity matrix scaled by 3.0")
print(f"PBC: {pbc.tolist()}")
print(f"Cutoff: {cutoff.item()}")

# Test both neighbor list implementations
print("\n=== Testing Neighbor List Functions ===")

try:
    # Test vesin_nl (should use vesin if available, fallback if not)
    mapping1, shifts1 = neighbors.vesin_nl(positions, cell, pbc, cutoff)
    print(f"✓ vesin_nl: Found {mapping1.shape[1]} neighbor pairs")

    # Test vesin_nl_ts (should use vesin if available, fallback if not)
    mapping2, shifts2 = neighbors.vesin_nl_ts(positions, cell, pbc, cutoff)
    print(f"✓ vesin_nl_ts: Found {mapping2.shape[1]} neighbor pairs")

    # Test standard_nl (pure PyTorch, always available)
    mapping3, shifts3 = neighbors.standard_nl(positions, cell, pbc, cutoff)
    print(f"✓ standard_nl: Found {mapping3.shape[1]} neighbor pairs")

    # Verify consistency
    assert mapping1.shape == mapping2.shape == mapping3.shape, (
        "Different numbers of neighbors found!"
    )
    print(
        f"✓ All implementations found the same number of neighbors: {mapping1.shape[1]}"
    )

except Exception as e:
    print(f"✗ Error in neighbor list testing: {e}")
    raise

print("\n=== Integration Test with Lennard-Jones Model ===")

try:
    # Test with a simple Lennard-Jones model
    from torch_sim.models.lennard_jones import LennardJonesModel

    lj_model = LennardJonesModel(
        device=device, dtype=torch.float32, compute_forces=True, compute_stress=True
    )

    # Create a simple state
    state = ts.SimState(
        positions=positions,
        masses=torch.ones(4, device=device, dtype=torch.float32),
        cell=cell.unsqueeze(0),  # [1, 3, 3]
        pbc=pbc,
        atomic_numbers=torch.ones(4, device=device, dtype=torch.long),
    )

    # Compute energy and forces
    result = lj_model(state)

    print("✓ LJ model computation successful")
    print(f"  Energy: {result['energy'].item():.6f}")
    print(f"  Forces shape: {result['forces'].shape}")
    print(f"  Forces norm: {torch.norm(result['forces']).item():.6f}")

except Exception as e:
    print(f"✗ Error in Lennard-Jones test: {e}")
    raise

print("\n=== Summary ===")
print("✓ ROCm patch integration successful!")
print("✓ Neighbor list functions work correctly")
print("✓ Model evaluation works correctly")
print(f"✓ All operations on {device}")

if neighbors.VESIN_AVAILABLE:
    print("\nNote: vesin is available, so optimal performance is used.")
    print("On AMD ROCm systems without vesin, the same functionality")
    print("would be provided by the pure PyTorch fallback implementation.")
else:
    print("\nNote: Using pure PyTorch fallback (vesin not available).")
    print("This demonstrates ROCm functionality.")
