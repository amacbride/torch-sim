"""HIP-accelerated neighbor list implementation for AMD GPUs.

This package provides a HIP-native neighbor list computation that can be used
as a drop-in replacement for standard_nl on AMD GPUs.
"""

import ctypes
import os
from pathlib import Path

import torch


# Try to load the compiled HIP library
HIP_NL_AVAILABLE = False
_lib = None

try:
    # Preload HSA runtime library to resolve symbols
    # This is needed because libamdhip64 depends on HSA runtime symbols
    try:
        _hsa = ctypes.CDLL("/opt/rocm/lib/libhsa-runtime64.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # Try versioned ROCm path
        _hsa = ctypes.CDLL(
            "/opt/rocm-7.0.1/lib/libhsa-runtime64.so.1", mode=ctypes.RTLD_GLOBAL
        )

    # Look for compiled library
    lib_path = Path(__file__).parent / "libhip_nl.so"
    if lib_path.exists():
        # Load with RTLD_GLOBAL so HIP/HSA symbols are visible
        _lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        # Define function signature
        _lib.hip_compute_neighborlist.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # positions
            ctypes.c_size_t,  # n_points
            ctypes.POINTER(ctypes.c_double),  # box
            ctypes.POINTER(ctypes.c_bool),  # periodic
            ctypes.c_double,  # cutoff
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),  # pairs_out
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # shifts_out
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # distances_out
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # vectors_out
            ctypes.POINTER(ctypes.c_size_t),  # n_pairs_out
            ctypes.c_bool,  # compute_distances
            ctypes.c_bool,  # compute_vectors
        ]
        _lib.hip_compute_neighborlist.restype = ctypes.c_int

        HIP_NL_AVAILABLE = True
except Exception as e:
    import warnings

    warnings.warn(f"hip_nl library not available: {e}", stacklevel=2)


def hip_nl(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: torch.Tensor,
    sort_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute neighbor list using HIP acceleration.

    This function provides a HIP-native implementation of neighbor list
    computation for AMD GPUs, offering better performance than PyTorch's
    generic operations.

    Args:
        positions: Atomic positions tensor of shape (n_atoms, 3)
        cell: Unit cell vectors in row vector convention, shape (3, 3)
        pbc: Periodic boundary conditions tensor of shape (3,)
        cutoff: Cutoff distance (scalar tensor)
        sort_id: If True, sort pairs by first index (not implemented yet)

    Returns:
        tuple: (mapping, shifts) where:
            - mapping: Tensor of shape (2, n_pairs) with atom indices
            - shifts: Tensor of shape (n_pairs, 3) with periodic image shifts

    Raises:
        RuntimeError: If HIP library is not available or computation fails
    """
    if not HIP_NL_AVAILABLE:
        raise RuntimeError(
            "hip_nl library not compiled. Please compile hip_neighborlist.hip first."
        )

    # Note: hip_nl accepts tensors on any device
    # The HIP kernel manages GPU memory internally via hipMalloc/hipMemcpy
    # This design allows hip_nl to work even when PyTorch GPU support is unavailable

    # Store original device to return results on same device
    original_device = positions.device

    # Convert to CPU and float64 for C interface (HIP kernel uses double precision)
    positions_cpu = positions.detach().cpu().to(torch.float64).numpy()
    cell_cpu = cell.detach().cpu().to(torch.float64).numpy()
    pbc_cpu = pbc.detach().cpu().numpy()
    cutoff_val = float(cutoff.item())

    n_points = positions_cpu.shape[0]

    # Prepare C arguments
    c_positions = positions_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_box = cell_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_periodic = pbc_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    c_pairs = ctypes.POINTER(ctypes.c_ulong)()
    c_shifts = ctypes.POINTER(ctypes.c_int)()
    c_distances = ctypes.POINTER(ctypes.c_double)()
    c_vectors = ctypes.POINTER(ctypes.c_double)()
    c_n_pairs = ctypes.c_size_t()

    # Call HIP kernel
    ret = _lib.hip_compute_neighborlist(
        c_positions,
        n_points,
        c_box,
        c_periodic,
        cutoff_val,
        ctypes.byref(c_pairs),
        ctypes.byref(c_shifts),
        ctypes.byref(c_distances),
        ctypes.byref(c_vectors),
        ctypes.byref(c_n_pairs),
        False,  # compute_distances
        False,  # compute_vectors
    )

    if ret != 0:
        raise RuntimeError(f"HIP kernel failed with error code {ret}")

    n_pairs = c_n_pairs.value

    # Convert results back to torch tensors
    import numpy as np

    # Read as flat arrays with explicit dtype
    pairs_flat = np.ctypeslib.as_array(c_pairs, shape=(n_pairs * 2,)).astype(np.uint64)
    shifts_flat = np.ctypeslib.as_array(c_shifts, shape=(n_pairs * 3,)).astype(np.int32)

    # Reshape
    pairs_np = pairs_flat.reshape(n_pairs, 2)
    shifts_np = shifts_flat.reshape(n_pairs, 3)

    # Convert to torch - uint64 becomes long, preserving values < 2^63
    mapping = torch.from_numpy(pairs_np.astype(np.int64)).t()
    shifts = torch.from_numpy(shifts_np).to(dtype=torch.long)

    # Move back to original device
    mapping = mapping.to(original_device)
    shifts = shifts.to(original_device)

    # Free C memory
    ctypes.CDLL(None).free(c_pairs)
    ctypes.CDLL(None).free(c_shifts)

    if sort_id:
        # Sort by first index
        sorted_indices = torch.argsort(mapping[0])
        mapping = mapping[:, sorted_indices]
        shifts = shifts[sorted_indices]

    return mapping, shifts


__all__ = ["HIP_NL_AVAILABLE", "hip_nl"]
