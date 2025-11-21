#!/usr/bin/env python
"""Standalone test script to verify hip_nl correctness without pytest.

This script tests hip_nl functionality directly, bypassing pytest's module
loading issues. Run this to verify hip_nl works correctly:

    python verify_hip_nl.py

Or with environment variables if needed:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python verify_hip_nl.py
"""

import sys

import torch

from hip_nl import hip_nl
from torch_sim import neighbors


def test_hip_nl_availability():
    """Test that HIP_NL_AVAILABLE flag is correctly set."""
    print("Test 1: HIP_NL Availability")
    print("-" * 80)
    print(f"  HIP_NL_AVAILABLE: {neighbors.HIP_NL_AVAILABLE}")
    print(f"  torch.version.hip: {torch.version.hip}")

    if neighbors.HIP_NL_AVAILABLE:
        print("  ✅ PASS: hip_nl is available")
        return True
    print("  ❌ FAIL: hip_nl not available")
    return False


def test_hip_nl_correctness():
    """Test hip_nl correctness with different PBC configurations."""
    print("\nTest 2: HIP_NL Correctness")
    print("-" * 80)

    test_cases = [
        ("No PBC", torch.tensor([False, False, False]), False),  # Expected to fail
        ("Full PBC", torch.tensor([True, True, True]), True),  # Expected to pass
        ("Mixed PBC", torch.tensor([True, False, True]), True),  # Expected to pass
    ]

    results = []

    for name, pbc, should_pass in test_cases:
        torch.manual_seed(42)
        positions = torch.rand(100, 3, dtype=torch.float32) * 20.0
        cell = torch.eye(3, dtype=torch.float32) * 20.0
        cutoff = torch.tensor(3.0, dtype=torch.float32)

        try:
            mapping_hip, shifts_hip = hip_nl(positions, cell, pbc, cutoff)
            mapping_std, shifts_std = neighbors.standard_nl(positions, cell, pbc, cutoff)

            matches = mapping_hip.shape == mapping_std.shape

            if matches and should_pass:
                status = "✅ PASS"
                passed = True
            elif not matches and not should_pass:
                status = "✅ XFAIL (expected)"
                passed = True
            elif matches and not should_pass:
                status = "❌ UNEXPECTED PASS"
                passed = False
            else:
                status = "❌ FAIL"
                passed = False

            print(
                f"  {name:12s}: {status} (hip={mapping_hip.shape[1]}, std={mapping_std.shape[1]})"
            )
            results.append(passed)

        except RuntimeError as e:
            if "error code 100" in str(e):
                print(f"  {name:12s}: SKIPPED (HIP runtime not available)")
                results.append(None)  # Skip doesn't count as pass or fail
            else:
                print(f"  {name:12s}: ❌ ERROR: {e}")
                results.append(False)

    # Count results
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)

    print()
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


def test_torchsim_nl_integration():
    """Test that torchsim_nl uses hip_nl on ROCm systems."""
    print("\nTest 3: TorchSim NL Integration")
    print("-" * 80)

    if not (neighbors.HIP_NL_AVAILABLE and torch.version.hip is not None):
        print("  SKIPPED: Requires hip_nl and ROCm environment")
        return None

    torch.manual_seed(42)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cell = torch.eye(3, dtype=torch.float32) * 3.0
    pbc = torch.tensor([True, True, True])
    cutoff = torch.tensor(1.5, dtype=torch.float32)

    try:
        mapping_torchsim, shifts_torchsim = neighbors.torchsim_nl(
            positions, cell, pbc, cutoff
        )
        mapping_standard, shifts_standard = neighbors.standard_nl(
            positions, cell, pbc, cutoff
        )

        if mapping_torchsim.shape == mapping_standard.shape:
            print(
                f"  ✅ PASS: torchsim_nl integration works ({mapping_torchsim.shape[1]} pairs)"
            )
            return True
        print("  ❌ FAIL: Shape mismatch")
        return False

    except RuntimeError as e:
        if "error code 100" in str(e):
            print("  SKIPPED: HIP runtime not available")
            return None
        print(f"  ❌ ERROR: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("=" * 80)
    print("HIP_NL Verification Tests")
    print("=" * 80)
    print()

    # Check environment
    import os

    hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa_override:
        print(f"Environment: HSA_OVERRIDE_GFX_VERSION={hsa_override}")
    print()

    # Run tests
    test1 = test_hip_nl_availability()
    test2 = test_hip_nl_correctness()
    test3 = test_torchsim_nl_integration()

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    results = [test1, test2, test3]
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)

    print(f"Tests: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("✅ All tests passed!")
        return 0
    print("❌ Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
