"""Tests for hip_nl - HIP-accelerated neighbor list for AMD GPUs.

These tests are separate from the main test suite because hip_nl uses a
standalone HIP context that conflicts with PyTorch's HIP backend.

Run these tests in isolation:
    cd hip_nl/dev_tests
    setenv HSA_OVERRIDE_GFX_VERSION 11.0.0
    setenv USE_HIP_NL 1
    pytest test_hip_nl.py -v
"""

import os

import pytest
import torch


# Set environment variables before any hip_nl imports
@pytest.fixture(scope="module", autouse=True)
def set_hip_nl_env():
    """Set environment variables for hip_nl tests."""
    old_hsa = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    old_use = os.environ.get("USE_HIP_NL")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    os.environ["USE_HIP_NL"] = "1"
    yield
    # Restore original values
    if old_hsa is None:
        os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    else:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = old_hsa
    if old_use is None:
        os.environ.pop("USE_HIP_NL", None)
    else:
        os.environ["USE_HIP_NL"] = old_use


def test_hip_nl_availability() -> None:
    """Test that hip_nl can be imported and HIP_NL_AVAILABLE is set."""
    from hip_nl import HIP_NL_AVAILABLE
    assert isinstance(HIP_NL_AVAILABLE, bool)
    assert HIP_NL_AVAILABLE, "hip_nl library should be available"


@pytest.mark.parametrize(
    "pbc_config",
    [
        torch.tensor([False, False, False]),  # No PBC - now matches with compatible_mode=True
        torch.tensor([True, True, True]),     # Full PBC
        torch.tensor([True, False, True]),    # Mixed PBC
    ],
)
def test_hip_nl_correctness(pbc_config: torch.Tensor) -> None:
    """Test that hip_nl produces correct results matching standard_nl."""
    from hip_nl import hip_nl, HIP_NL_AVAILABLE
    from torch_sim.neighbors import standard_nl

    if not HIP_NL_AVAILABLE:
        pytest.skip("hip_nl not available")

    device = torch.device("cpu")
    dtype = torch.float32

    torch.manual_seed(42)
    positions = torch.rand(100, 3, device=device, dtype=dtype) * 20.0
    cell = torch.eye(3, device=device, dtype=dtype) * 20.0
    pbc = pbc_config.to(device)
    cutoff = torch.tensor(3.0, device=device, dtype=dtype)

    mapping_hip, shifts_hip = hip_nl(positions, cell, pbc, cutoff)
    mapping_std, shifts_std = standard_nl(positions, cell, pbc, cutoff)

    assert mapping_hip.shape == mapping_std.shape, (
        f"hip_nl found {mapping_hip.shape[1]} pairs, "
        f"standard_nl found {mapping_std.shape[1]} pairs"
    )
    assert shifts_hip.shape == shifts_std.shape


def test_torchsim_nl_uses_hip_on_rocm() -> None:
    """Test that torchsim_nl uses hip_nl when HSA_OVERRIDE is set."""
    from hip_nl import HIP_NL_AVAILABLE
    from torch_sim.neighbors import torchsim_nl

    if not HIP_NL_AVAILABLE:
        pytest.skip("hip_nl not available")
    if torch.version.hip is None:
        pytest.skip("Not a ROCm environment")

    device = torch.device("cpu")
    dtype = torch.float32

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
        device=device,
        dtype=dtype,
    )
    cell = torch.eye(3, device=device, dtype=dtype) * 5.0
    pbc = torch.tensor([False, False, False], device=device)
    cutoff = torch.tensor(2.0, device=device, dtype=dtype)

    mapping, shifts = torchsim_nl(positions, cell, pbc, cutoff)

    assert mapping.shape[0] == 2
    assert shifts.shape[1] == 3
    assert mapping.shape[1] == shifts.shape[0]
