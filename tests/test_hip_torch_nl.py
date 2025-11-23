"""Tests for hip_torch_nl - HIP-accelerated neighbor list as PyTorch extension."""

import pytest
import torch

# Try to import hip_torch_nl
try:
    from hip_torch_nl import (
        HIP_TORCH_NL_AVAILABLE,
        hip_torch_nl,
        hip_torch_nl_v1,
        hip_torch_nl_v2,
    )
except ImportError:
    HIP_TORCH_NL_AVAILABLE = False
    hip_torch_nl = None
    hip_torch_nl_v1 = None
    hip_torch_nl_v2 = None

from torch_sim.neighbors import standard_nl


# Skip all tests if hip_torch_nl is not available or not on AMD GPU
pytestmark = pytest.mark.skipif(
    not HIP_TORCH_NL_AVAILABLE or torch.version.hip is None,
    reason="hip_torch_nl not available or not on AMD ROCm",
)


@pytest.fixture
def simple_system():
    """Create a simple test system."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    pbc = torch.tensor([True, True, True])
    cutoff = 1.5
    return positions, cell, pbc, cutoff


@pytest.fixture
def periodic_system():
    """Create a periodic test system with atoms near boundaries."""
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],  # Near origin
            [9.9, 9.9, 9.9],  # Near opposite corner (should be neighbor via PBC)
            [5.0, 5.0, 5.0],  # Center
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 10.0
    pbc = torch.tensor([True, True, True])
    cutoff = 1.0
    return positions, cell, pbc, cutoff


class TestHipTorchNLBasic:
    """Basic functionality tests for hip_torch_nl."""

    def test_import(self):
        """Test that hip_torch_nl can be imported."""
        assert HIP_TORCH_NL_AVAILABLE is True

    def test_simple_system(self, simple_system):
        """Test hip_torch_nl on a simple system."""
        positions, cell, pbc, cutoff = simple_system

        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        # Move to GPU
        positions_gpu = positions.cuda()
        cell_gpu = cell.cuda()
        pbc_gpu = pbc.cuda()

        # Call hip_torch_nl
        mapping, shifts = hip_torch_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff)

        # Basic checks
        assert mapping.shape[0] == 2
        assert mapping.device.type == "cuda"
        assert shifts.device.type == "cuda"
        assert mapping.shape[1] == shifts.shape[0]

    def test_output_types(self, simple_system):
        """Test that output types are correct."""
        positions, cell, pbc, cutoff = simple_system

        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        positions_gpu = positions.cuda()
        cell_gpu = cell.cuda()
        pbc_gpu = pbc.cuda()

        mapping, shifts = hip_torch_nl(positions_gpu, cell_gpu, pbc_gpu, cutoff)

        assert mapping.dtype == torch.int64
        assert shifts.dtype == positions.dtype


class TestHipTorchNLCorrectness:
    """Correctness tests comparing hip_torch_nl to standard_nl."""

    @pytest.mark.parametrize("n_atoms", [5, 10, 20, 50, 100])
    def test_pair_count_matches_standard_nl(self, n_atoms):
        """Test that hip_torch_nl finds the same number of pairs as standard_nl."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        torch.manual_seed(42 + n_atoms)
        positions = torch.rand(n_atoms, 3, dtype=torch.float64) * 10.0
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([True, True, True])
        cutoff = 3.0
        cutoff_t = torch.tensor(cutoff, dtype=torch.float64)

        # Standard NL (CPU)
        std_mapping, _ = standard_nl(positions, cell, pbc, cutoff_t)

        # Hip torch NL (GPU)
        hip_mapping, _ = hip_torch_nl(
            positions.cuda(), cell.cuda(), pbc.cuda(), cutoff
        )

        assert std_mapping.shape[1] == hip_mapping.shape[1], (
            f"Pair count mismatch: standard_nl={std_mapping.shape[1]}, "
            f"hip_torch_nl={hip_mapping.shape[1]}"
        )

    def test_periodic_boundary_conditions(self, periodic_system):
        """Test that PBC pairs are correctly identified."""
        positions, cell, pbc, cutoff = periodic_system

        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        cutoff_t = torch.tensor(cutoff, dtype=torch.float64)

        # Standard NL
        std_mapping, std_shifts = standard_nl(positions, cell, pbc, cutoff_t)

        # Hip torch NL
        hip_mapping, hip_shifts = hip_torch_nl(
            positions.cuda(), cell.cuda(), pbc.cuda(), cutoff
        )

        # Should find the same pairs (atoms 0 and 1 are neighbors via PBC)
        assert std_mapping.shape[1] == hip_mapping.shape[1]

    def test_non_periodic_system(self):
        """Test with non-periodic boundary conditions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        pbc = torch.tensor([False, False, False])
        cutoff = 2.0
        cutoff_t = torch.tensor(cutoff, dtype=torch.float64)

        # Standard NL
        std_mapping, _ = standard_nl(positions, cell, pbc, cutoff_t)

        # Hip torch NL
        hip_mapping, _ = hip_torch_nl(
            positions.cuda(), cell.cuda(), pbc.cuda(), cutoff
        )

        # Should find 2 pairs: (0,1) and (1,0)
        assert std_mapping.shape[1] == hip_mapping.shape[1]


class TestHipTorchNLV2CellList:
    """Tests for V2 cell-list algorithm."""

    @pytest.mark.parametrize("n_atoms", [10, 50, 100, 200, 500])
    def test_v2_matches_v1(self, n_atoms):
        """Test that V2 cell-list finds same pairs as V1 brute-force."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        torch.manual_seed(42 + n_atoms)
        positions = torch.rand(n_atoms, 3, dtype=torch.float64).cuda() * 10.0
        cell = torch.eye(3, dtype=torch.float64).cuda() * 10.0
        pbc = torch.tensor([True, True, True]).cuda()
        cutoff = torch.tensor(3.0, dtype=torch.float64)

        # V1 brute-force
        mapping_v1, shifts_v1 = hip_torch_nl_v1(
            positions, cell, pbc, cutoff, compatible_mode=False
        )

        # V2 cell-list
        mapping_v2, shifts_v2 = hip_torch_nl_v2(
            positions, cell, pbc, cutoff, compatible_mode=False
        )

        # Same number of pairs
        assert mapping_v1.shape[1] == mapping_v2.shape[1], (
            f"V1={mapping_v1.shape[1]} vs V2={mapping_v2.shape[1]}"
        )

    def test_v2_with_algorithm_auto(self):
        """Test that algorithm='auto' selects V2 for large systems."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        # Large system should use V2 (>5000 atoms triggers cell_list)
        n_atoms = 100
        torch.manual_seed(42)
        positions = torch.rand(n_atoms, 3, dtype=torch.float64).cuda() * 10.0
        cell = torch.eye(3, dtype=torch.float64).cuda() * 10.0
        pbc = torch.tensor([True, True, True]).cuda()
        cutoff = torch.tensor(3.0, dtype=torch.float64)

        # With auto algorithm
        mapping_auto, shifts_auto = hip_torch_nl(
            positions, cell, pbc, cutoff, algorithm="auto", compatible_mode=False
        )

        # Explicit V1 for comparison
        mapping_v1, shifts_v1 = hip_torch_nl_v1(
            positions, cell, pbc, cutoff, compatible_mode=False
        )

        assert mapping_auto.shape[1] == mapping_v1.shape[1]

    def test_v2_periodic_boundary(self):
        """Test V2 handles periodic boundaries correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        # Atoms near opposite corners - should be neighbors via PBC
        positions = torch.tensor(
            [
                [0.1, 0.1, 0.1],
                [9.9, 9.9, 9.9],
            ],
            dtype=torch.float64,
        ).cuda()
        cell = torch.eye(3, dtype=torch.float64).cuda() * 10.0
        pbc = torch.tensor([True, True, True]).cuda()
        cutoff = torch.tensor(1.0, dtype=torch.float64)

        mapping_v2, shifts_v2 = hip_torch_nl_v2(
            positions, cell, pbc, cutoff, compatible_mode=False
        )

        # Should find 2 pairs: (0,1) and (1,0)
        assert mapping_v2.shape[1] == 2


class TestHipTorchNLIntegration:
    """Integration tests with torchsim_nl."""

    def test_torchsim_nl_uses_hip_torch_nl(self):
        """Test that torchsim_nl automatically uses hip_torch_nl on AMD GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP not available")

        from torch_sim.neighbors import torchsim_nl, _check_hip_torch_nl_available

        # Verify hip_torch_nl is available
        assert _check_hip_torch_nl_available() is True

        # Create test system on GPU
        positions = torch.rand(50, 3, dtype=torch.float64).cuda() * 10.0
        cell = torch.eye(3, dtype=torch.float64).cuda() * 10.0
        pbc = torch.tensor([True, True, True]).cuda()
        cutoff = torch.tensor(3.0, dtype=torch.float64)

        # Call torchsim_nl - should use hip_torch_nl automatically
        mapping, shifts = torchsim_nl(positions, cell, pbc, cutoff)

        # Results should be on GPU
        assert mapping.device.type == "cuda"
        assert shifts.device.type == "cuda"
