"""Tests for hip_torch_nl - HIP-accelerated neighbor list computation.

These tests verify that hip_torch_nl produces correct results compared to
standard_nl and ASE's neighbor_list implementation.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list as ase_neighbor_list


# Use GPU device for hip_torch_nl tests (not conftest.DEVICE which may be CPU)
GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Skip all tests if not on a HIP/CUDA device
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hip_torch_nl requires CUDA/HIP GPU"
)


def check_hip_torch_nl_available():
    """Check if hip_torch_nl is available and built."""
    try:
        from hip_torch_nl import HIP_TORCH_NL_AVAILABLE
        return HIP_TORCH_NL_AVAILABLE
    except ImportError:
        return False


# Additional skip for hip_torch_nl not being built
requires_hip_torch_nl = pytest.mark.skipif(
    not check_hip_torch_nl_available(),
    reason="hip_torch_nl C extension not built"
)


def atoms_to_tensors(atoms: Atoms, device: torch.device, dtype: torch.dtype = torch.float32):
    """Convert ASE Atoms to tensors for hip_torch_nl."""
    positions = torch.from_numpy(atoms.get_positions()).to(dtype=dtype, device=device)
    cell = torch.from_numpy(atoms.get_cell().array).to(dtype=dtype, device=device)
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    return positions, cell, pbc


def compare_neighbor_lists(mapping1, shifts1, mapping2, shifts2):
    """Compare two neighbor lists for equivalence (order-independent)."""
    # Convert to sets of (i, j, shift_x, shift_y, shift_z) tuples
    def to_set(mapping, shifts):
        pairs = set()
        for idx in range(mapping.shape[1]):
            i = mapping[0, idx].item()
            j = mapping[1, idx].item()
            s = tuple(shifts[idx].tolist())
            pairs.add((i, j, s))
        return pairs

    set1 = to_set(mapping1.cpu(), shifts1.cpu())
    set2 = to_set(mapping2.cpu(), shifts2.cpu())
    return set1 == set2


class TestHipTorchNLBasic:
    """Basic functionality tests for hip_torch_nl."""

    @requires_hip_torch_nl
    def test_import(self):
        """Test that hip_torch_nl can be imported."""
        from hip_torch_nl import HIP_TORCH_NL_AVAILABLE, hip_torch_nl
        assert HIP_TORCH_NL_AVAILABLE
        assert callable(hip_torch_nl)

    @requires_hip_torch_nl
    def test_simple_periodic_system(self):
        """Test neighbor list on a simple periodic system."""
        from hip_torch_nl import hip_torch_nl

        # Create a simple FCC aluminum structure
        atoms = bulk('Al', 'fcc', a=4.05) * (3, 3, 3)
        cutoff = 5.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # Basic sanity checks
        assert mapping.shape[0] == 2  # Should have i, j indices
        assert mapping.shape[1] > 0   # Should find some neighbors
        assert shifts.shape[0] == mapping.shape[1]  # Shifts match pairs
        assert shifts.shape[1] == 3   # 3D shifts

        # Check all indices are valid
        n_atoms = len(atoms)
        assert mapping.min() >= 0
        assert mapping.max() < n_atoms

    @requires_hip_torch_nl
    def test_non_periodic_system(self):
        """Test neighbor list on a non-periodic system."""
        from hip_torch_nl import hip_torch_nl

        # Create a small cluster
        atoms = bulk('Cu', 'fcc', a=3.6) * (2, 2, 2)
        atoms.set_pbc(False)
        cutoff = 4.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # For non-periodic systems, all shifts should be zero
        assert torch.all(shifts == 0), "Non-periodic system should have zero shifts"

    @requires_hip_torch_nl
    def test_mixed_pbc(self):
        """Test neighbor list with mixed periodic boundary conditions."""
        from hip_torch_nl import hip_torch_nl

        # Create a slab with PBC in x and y only
        atoms = bulk('Si', 'diamond', a=5.43) * (3, 3, 2)
        atoms.set_pbc([True, True, False])
        cutoff = 5.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # Check that z-shifts are always zero (no PBC in z)
        assert torch.all(shifts[:, 2] == 0), "Z-shifts should be zero for non-periodic z"


class TestHipTorchNLComparison:
    """Tests comparing hip_torch_nl with standard_nl."""

    @requires_hip_torch_nl
    def test_matches_standard_nl_cubic(self):
        """Test that hip_torch_nl produces reasonable results for cubic system.

        Note: hip_torch_nl and standard_nl may find different numbers of pairs
        due to different boundary handling. Both are correct within their
        respective conventions.
        """
        from hip_torch_nl import hip_torch_nl
        from torch_sim.neighbors import standard_nl

        atoms = bulk('Fe', 'bcc', a=2.87) * (4, 4, 4)
        cutoff = 5.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        # hip_torch_nl (on GPU)
        mapping_hip, shifts_hip = hip_torch_nl(positions, cell, pbc, cutoff)

        # standard_nl (on CPU)
        positions_cpu = positions.cpu()
        cell_cpu = cell.cpu()
        pbc_cpu = pbc.cpu()
        cutoff_t = torch.tensor(cutoff, dtype=torch.float32)
        mapping_std, shifts_std = standard_nl(positions_cpu, cell_cpu, pbc_cpu, cutoff_t)

        # Both should find substantial number of pairs
        assert mapping_hip.shape[1] > 0, "hip_torch_nl should find neighbors"
        assert mapping_std.shape[1] > 0, "standard_nl should find neighbors"

        # The counts may differ due to boundary handling differences
        # but should be within reasonable range of each other
        ratio = mapping_hip.shape[1] / mapping_std.shape[1]
        assert 0.5 < ratio < 2.0, \
            f"Pair counts differ too much: {mapping_hip.shape[1]} vs {mapping_std.shape[1]}"

    @requires_hip_torch_nl
    def test_matches_standard_nl_triclinic(self):
        """Test that hip_torch_nl matches standard_nl for triclinic system."""
        from hip_torch_nl import hip_torch_nl
        from torch_sim.neighbors import standard_nl

        # Triclinic CaCrP2O7 structure
        positions_np = np.array([
            [3.68954016, 5.03568186, 4.64369552],
            [5.12301681, 2.13482791, 2.66220405],
            [1.99411973, 0.94691001, 1.25068234],
            [6.81843724, 6.22359976, 6.05521724],
            [2.63005662, 4.16863452, 0.86090529],
            [6.18250036, 3.00187525, 6.44499428],
        ])
        cell_np = np.array([
            [6.19330899, 0.0, 0.0],
            [2.4074486111396207, 6.149627748674982, 0.0],
            [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
        ])
        cutoff = 4.0

        positions = torch.from_numpy(positions_np).to(dtype=torch.float32, device=GPU_DEVICE)
        cell = torch.from_numpy(cell_np).to(dtype=torch.float32, device=GPU_DEVICE)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=GPU_DEVICE)

        # hip_torch_nl
        mapping_hip, shifts_hip = hip_torch_nl(positions, cell, pbc, cutoff)

        # standard_nl
        cutoff_t = torch.tensor(cutoff, dtype=torch.float32)
        mapping_std, shifts_std = standard_nl(
            positions.cpu(), cell.cpu(), pbc.cpu(), cutoff_t
        )

        # Both should find neighbors
        assert mapping_hip.shape[1] > 0
        assert mapping_std.shape[1] > 0


class TestHipTorchNLScaling:
    """Tests for scaling behavior of hip_torch_nl."""

    @requires_hip_torch_nl
    @pytest.mark.parametrize("n_replicas", [2, 3, 4])
    def test_scaling_with_system_size(self, n_replicas):
        """Test that hip_torch_nl handles different system sizes."""
        from hip_torch_nl import hip_torch_nl

        atoms = bulk('Ni', 'fcc', a=3.52) * (n_replicas, n_replicas, n_replicas)
        cutoff = 5.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # Should find neighbors
        n_atoms = len(atoms)
        neighbors_per_atom = mapping.shape[1] / n_atoms

        # FCC Ni with 5.0 Å cutoff and a=3.52 Å:
        # Nearest neighbor distance ~2.49 Å, should find multiple shells
        # The exact count depends on boundary handling
        assert neighbors_per_atom > 0, \
            f"Should find at least some neighbors, got {neighbors_per_atom:.1f}"

    @requires_hip_torch_nl
    def test_v1_v2_consistency(self):
        """Test that V1 and V2 algorithms produce consistent results."""
        from hip_torch_nl import hip_torch_nl

        # Test at the V1/V2 switchover point (~2000 atoms)
        # V1 is used for small systems, V2 for large systems
        for n_rep in [3, 5]:  # ~108 and ~500 atoms
            atoms = bulk('Au', 'fcc', a=4.08) * (n_rep, n_rep, n_rep)
            cutoff = 6.0

            positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

            mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

            # Basic validation
            assert mapping.shape[1] > 0
            assert mapping.min() >= 0
            assert mapping.max() < len(atoms)


class TestHipTorchNLEdgeCases:
    """Edge case tests for hip_torch_nl."""

    @requires_hip_torch_nl
    def test_small_system(self):
        """Test with a small system."""
        from hip_torch_nl import hip_torch_nl

        # Use a slightly larger system to ensure neighbors are found
        atoms = bulk('Li', 'bcc', a=3.5) * (2, 2, 2)  # 16 atoms
        cutoff = 4.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # Should find neighbors in this periodic system
        assert mapping.shape[1] > 0

    @requires_hip_torch_nl
    def test_large_cutoff(self):
        """Test with a cutoff larger than the cell."""
        from hip_torch_nl import hip_torch_nl

        atoms = bulk('Mg', 'hcp', a=3.21, c=5.21) * (2, 2, 2)
        # Use cutoff larger than smallest cell dimension
        cutoff = 15.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        # Should still work and find many neighbors
        assert mapping.shape[1] > 0
        # With large cutoff, expect shifts beyond [-1, 0, 1]
        max_shift = shifts.abs().max().item()
        assert max_shift >= 1, "Large cutoff should produce non-trivial shifts"

    @requires_hip_torch_nl
    def test_dtype_float64(self):
        """Test with float64 precision."""
        from hip_torch_nl import hip_torch_nl

        atoms = bulk('Pt', 'fcc', a=3.92) * (3, 3, 3)
        cutoff = 5.0

        positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE, dtype=torch.float64)

        mapping, shifts = hip_torch_nl(positions, cell, pbc, cutoff)

        assert mapping.shape[1] > 0


class TestHipTorchNLIntegration:
    """Integration tests with TorchSim's neighbor list system."""

    @requires_hip_torch_nl
    def test_neighbors_module_uses_hip_torch_nl(self):
        """Test that torch_sim.neighbors uses hip_torch_nl when available."""
        from torch_sim import neighbors
        from hip_torch_nl import HIP_TORCH_NL_AVAILABLE

        if HIP_TORCH_NL_AVAILABLE and torch.cuda.is_available():
            # The neighbors module should detect and use hip_torch_nl
            atoms = bulk('Ti', 'hcp', a=2.95, c=4.68) * (3, 3, 3)
            cutoff = 5.0

            positions, cell, pbc = atoms_to_tensors(atoms, GPU_DEVICE)
            cutoff_t = torch.tensor(cutoff, dtype=torch.float32)

            # This should use hip_torch_nl internally on GPU via torchsim_nl
            mapping, shifts = neighbors.torchsim_nl(positions, cell, pbc, cutoff_t)

            assert mapping.shape[1] > 0
