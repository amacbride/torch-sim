#!/usr/bin/env python
"""Staged performance comparison between standard_nl and hip_nl.

IMPORTANT: hip_nl and PyTorch GPU cannot coexist in the same process.
This script must be run in two stages:

Stage 1 - Benchmark standard_nl (uses PyTorch GPU):
    setenv HSA_OVERRIDE_GFX_VERSION 11.0.0
    python benchmark_comparison.py --stage standard

Stage 2 - Benchmark hip_nl (uses standalone HIP):
    setenv HSA_OVERRIDE_GFX_VERSION 11.0.0
    setenv USE_HIP_NL 1
    python benchmark_comparison.py --stage hip_nl

Compare results:
    python benchmark_comparison.py --compare
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from ase.build import bulk


def benchmark_nl(nl_fn, positions, cell, pbc, cutoff, n_warmup=5, n_iter=20):
    """Benchmark a neighbor list function."""
    # Warmup
    for _ in range(n_warmup):
        _ = nl_fn(positions, cell, pbc, cutoff)

    # Benchmark
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        mapping, shifts = nl_fn(positions, cell, pbc, cutoff)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    n_pairs = mapping.shape[1]
    mean_time = sum(times) / len(times)

    return {
        "mean_ms": mean_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5,
        "n_pairs": n_pairs,
        "pairs_per_ms": n_pairs / mean_time,
    }


def create_test_system(n_atoms, dtype=torch.float32):
    """Create a test system with specified number of atoms (CPU tensors)."""
    base = bulk("Si", "diamond", a=5.43, cubic=True)
    target_size = int((n_atoms / len(base)) ** (1 / 3)) + 1
    atoms = base.repeat([target_size, target_size, target_size])

    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(atoms.get_cell().array, dtype=dtype)
    pbc = torch.tensor([True, True, True])
    cutoff = torch.tensor(5.0, dtype=dtype)

    return positions, cell, pbc, cutoff, len(atoms)


def run_standard_benchmark(sizes, output_file):
    """Benchmark standard_nl."""
    from torch_sim.neighbors import standard_nl

    print("=" * 70)
    print("Benchmarking standard_nl (PyTorch implementation)")
    print("=" * 70)

    results = {}
    for target_n in sizes:
        positions, cell, pbc, cutoff, actual_n = create_test_system(target_n)

        print(f"\nSystem: {actual_n} atoms")
        stats = benchmark_nl(standard_nl, positions, cell, pbc, cutoff)

        print(f"  Mean time:   {stats['mean_ms']:.3f} ms")
        print(f"  Pairs found: {stats['n_pairs']}")
        print(f"  Pairs/ms:    {stats['pairs_per_ms']:.1f}")

        results[actual_n] = stats

    # Save results
    with open(output_file, "w") as f:
        json.dump({"implementation": "standard_nl", "results": results}, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return results


def run_hip_nl_benchmark(sizes, output_file):
    """Benchmark hip_nl."""
    # Check environment
    if os.environ.get("USE_HIP_NL") != "1":
        print("ERROR: USE_HIP_NL=1 must be set to run hip_nl benchmark")
        print("Run: setenv USE_HIP_NL 1")
        return None

    from hip_nl import hip_nl, HIP_NL_AVAILABLE

    if not HIP_NL_AVAILABLE:
        print("ERROR: hip_nl library not available")
        return None

    print("=" * 70)
    print("Benchmarking hip_nl (HIP/ROCm implementation)")
    print("=" * 70)

    results = {}
    for target_n in sizes:
        positions, cell, pbc, cutoff, actual_n = create_test_system(target_n)

        print(f"\nSystem: {actual_n} atoms")
        stats = benchmark_nl(hip_nl, positions, cell, pbc, cutoff)

        print(f"  Mean time:   {stats['mean_ms']:.3f} ms")
        print(f"  Pairs found: {stats['n_pairs']}")
        print(f"  Pairs/ms:    {stats['pairs_per_ms']:.1f}")

        results[actual_n] = stats

    # Save results
    with open(output_file, "w") as f:
        json.dump({"implementation": "hip_nl", "results": results}, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return results


def compare_results(standard_file, hip_nl_file):
    """Compare results from both benchmarks."""
    if not Path(standard_file).exists():
        print(f"ERROR: {standard_file} not found. Run --stage standard first.")
        return
    if not Path(hip_nl_file).exists():
        print(f"ERROR: {hip_nl_file} not found. Run --stage hip_nl first.")
        return

    with open(standard_file) as f:
        standard_data = json.load(f)
    with open(hip_nl_file) as f:
        hip_nl_data = json.load(f)

    print("=" * 70)
    print("Performance Comparison: standard_nl vs hip_nl")
    print("=" * 70)
    print()
    print(f"{'Atoms':>8} | {'standard_nl':>12} | {'hip_nl':>12} | {'Speedup':>10}")
    print("-" * 50)

    for n_atoms in sorted(standard_data["results"].keys(), key=int):
        if n_atoms in hip_nl_data["results"]:
            std_ms = standard_data["results"][n_atoms]["mean_ms"]
            hip_ms = hip_nl_data["results"][n_atoms]["mean_ms"]
            speedup = std_ms / hip_ms

            print(f"{n_atoms:>8} | {std_ms:>10.3f}ms | {hip_ms:>10.3f}ms | {speedup:>9.2f}x")

    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark standard_nl vs hip_nl")
    parser.add_argument(
        "--stage",
        choices=["standard", "hip_nl"],
        help="Which implementation to benchmark",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results from both stages",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="System sizes to benchmark",
    )

    args = parser.parse_args()

    output_dir = Path(__file__).parent
    standard_file = output_dir / "benchmark_standard_nl.json"
    hip_nl_file = output_dir / "benchmark_hip_nl.json"

    if args.compare:
        compare_results(standard_file, hip_nl_file)
    elif args.stage == "standard":
        run_standard_benchmark(args.sizes, standard_file)
    elif args.stage == "hip_nl":
        run_hip_nl_benchmark(args.sizes, hip_nl_file)
    else:
        parser.print_help()
        print("\nExample usage (tcsh):")
        print("  # Stage 1: Benchmark standard_nl")
        print("  setenv HSA_OVERRIDE_GFX_VERSION 11.0.0")
        print("  python benchmark_comparison.py --stage standard")
        print()
        print("  # Stage 2: Benchmark hip_nl (NEW SHELL or unsetenv first)")
        print("  setenv HSA_OVERRIDE_GFX_VERSION 11.0.0")
        print("  setenv USE_HIP_NL 1")
        print("  python benchmark_comparison.py --stage hip_nl")
        print()
        print("  # Compare results")
        print("  python benchmark_comparison.py --compare")


if __name__ == "__main__":
    main()
