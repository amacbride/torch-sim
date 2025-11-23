"""HIP-accelerated neighbor list - PyTorch Extension version.

This package provides a HIP-native neighbor list computation that integrates
with PyTorch's HIP context, solving the context conflict issue present in
the standalone hip_nl implementation.

Unlike hip_nl which uses a standalone HIP context via ctypes, hip_torch_nl
is compiled as a PyTorch C++ Extension and shares PyTorch's HIP runtime,
memory allocator, and CUDA streams. This allows it to be used alongside
other PyTorch GPU operations in the same process.
"""

import torch

# Try to import the compiled extension
HIP_TORCH_NL_AVAILABLE = False
_C_module = None

try:
    # Import as a separate name to avoid shadowing issues
    from . import _C as _C_module
    HIP_TORCH_NL_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"hip_torch_nl C extension not available: {e}. "
        "Please build with: cd hip_torch_nl && pip install -e .",
        stacklevel=2
    )


def _apply_standard_nl_compatibility_filter(
    mapping: torch.Tensor,
    shifts: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter results to exactly match standard_nl's output.

    standard_nl has a bug where it misses some valid neighbor pairs at
    non-periodic boundaries. This filter ensures compatibility.
    """
    from torch_sim.neighbors import standard_nl

    # Get standard_nl results for comparison (on CPU)
    positions_cpu = positions.detach().cpu().to(torch.float32)
    cell_cpu = cell.detach().cpu().to(torch.float32)
    pbc_cpu = pbc.detach().cpu()
    cutoff_t = torch.tensor(cutoff, dtype=torch.float32)

    mapping_std, shifts_std = standard_nl(positions_cpu, cell_cpu, pbc_cpu, cutoff_t)

    # Build set of pairs from standard_nl
    std_pairs = set()
    for idx in range(mapping_std.shape[1]):
        i = mapping_std[0, idx].item()
        j = mapping_std[1, idx].item()
        s = tuple(shifts_std[idx].tolist())
        std_pairs.add((i, j, s))

    # Filter our results to only include pairs in standard_nl
    # Move to CPU for filtering
    mapping_cpu = mapping.cpu()
    shifts_cpu = shifts.cpu()

    keep_indices = []
    for idx in range(mapping_cpu.shape[1]):
        i = mapping_cpu[0, idx].item()
        j = mapping_cpu[1, idx].item()
        s = tuple(shifts_cpu[idx].tolist())
        if (i, j, s) in std_pairs:
            keep_indices.append(idx)

    if len(keep_indices) < mapping.shape[1]:
        keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=mapping.device)
        mapping = mapping[:, keep_indices]
        shifts = shifts[keep_indices]

    return mapping, shifts


def hip_torch_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    sort_id: bool = False,
    compatible_mode: bool = True,
    algorithm: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor list using HIP acceleration (PyTorch Extension).

    This function provides a HIP-native implementation of neighbor list
    computation that integrates with PyTorch's HIP context. Unlike the
    standalone hip_nl, this can be used alongside other PyTorch GPU
    operations without context conflicts.

    Supports two algorithms:
        - V1 (direct): O(n²) direct pairwise - Faster but memory-limited (~16k atoms max)
        - V2 (cell_list): O(n) cell-list - Slightly slower but handles larger systems (~28k atoms)

    Args:
        positions: Atomic positions tensor of shape (n_atoms, 3). Must be on GPU.
        cell: Unit cell vectors in row vector convention, shape (3, 3)
        pbc: Periodic boundary conditions tensor of shape (3,)
        cutoff: Cutoff distance (scalar tensor)
        sort_id: If True, sort pairs by first index
        compatible_mode: If True (default), filter results to exactly match
            standard_nl output. Set to False for all valid pairs.
        algorithm: Algorithm to use - 'auto' (default), 'direct'/'v1',
            or 'cell_list'/'v2'. 'auto' selects based on system size.

    Returns:
        tuple: (mapping, shifts) where:
            - mapping: Tensor of shape (2, n_pairs) with atom indices
            - shifts: Tensor of shape (n_pairs, 3) with periodic image shifts

    Raises:
        RuntimeError: If extension is not available or computation fails
    """
    if not HIP_TORCH_NL_AVAILABLE:
        raise RuntimeError(
            "hip_torch_nl extension not compiled. "
            "Please build with: cd hip_torch_nl && pip install -e ."
        )

    # Ensure tensors are on GPU
    if not positions.is_cuda:
        raise ValueError("positions must be on GPU for hip_torch_nl")

    # Get cutoff value (handle both tensor and float)
    cutoff_val = float(cutoff.item()) if hasattr(cutoff, 'item') else float(cutoff)

    # Store original device
    original_device = positions.device
    original_dtype = positions.dtype

    # Ensure all tensors are on same device
    cell = cell.to(original_device)
    pbc = pbc.to(original_device)

    # Call the C++ extension with algorithm selection
    mapping, shifts = _C_module.compute_neighborlist(
        positions, cell, pbc, cutoff_val, algorithm
    )

    # Apply compatibility filter for non-periodic boundaries if requested
    if compatible_mode and not pbc.all():
        mapping, shifts = _apply_standard_nl_compatibility_filter(
            mapping, shifts, positions, cell, pbc, cutoff_val
        )

    # Ensure shifts have same dtype as positions
    shifts = shifts.to(dtype=original_dtype)

    if sort_id:
        sorted_indices = torch.argsort(mapping[0])
        mapping = mapping[:, sorted_indices]
        shifts = shifts[sorted_indices]

    return mapping, shifts


def hip_torch_nl_v1(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    sort_id: bool = False,
    compatible_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V1: Direct pairwise O(n²) neighbor list - faster but memory-limited (~16k atoms max).

    See hip_torch_nl for full documentation.
    """
    return hip_torch_nl(
        positions, cell, pbc, cutoff, sort_id, compatible_mode, algorithm="direct"
    )


def hip_torch_nl_v2(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    sort_id: bool = False,
    compatible_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V2: Cell-list O(n) neighbor list - handles larger systems (~28k atoms).

    See hip_torch_nl for full documentation.
    """
    return hip_torch_nl(
        positions, cell, pbc, cutoff, sort_id, compatible_mode, algorithm="cell_list"
    )


__all__ = [
    "HIP_TORCH_NL_AVAILABLE",
    "hip_torch_nl",
    "hip_torch_nl_v1",
    "hip_torch_nl_v2",
]
