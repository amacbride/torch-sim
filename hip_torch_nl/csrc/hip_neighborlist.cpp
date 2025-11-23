/*
 * PyTorch C++ Extension binding layer for HIP neighbor list
 *
 * This provides the Python-visible interface and delegates to the HIP kernel.
 * Supports two algorithms:
 *   - V1 (direct): O(n²) direct pairwise - Faster but memory-limited (~16k atoms max)
 *   - V2 (cell_list): O(n) cell-list - Slightly slower but handles larger systems (~28k atoms)
 */

#include <torch/extension.h>
#include <vector>
#include <string>

// Forward declaration of HIP kernel wrappers
// V1: Brute-force O(n²)
std::tuple<torch::Tensor, torch::Tensor> hip_compute_neighborlist_kernel(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff,
    bool compute_distances,
    bool compute_vectors
);

// V2: Cell-list O(n)
std::tuple<torch::Tensor, torch::Tensor> hip_compute_neighborlist_cell_list(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff
);

// Input validation helper
void validate_inputs(
    const torch::Tensor& positions,
    const torch::Tensor& cell,
    const torch::Tensor& pbc
) {
    TORCH_CHECK(positions.device().is_cuda(),
                "positions must be on GPU (CUDA/HIP device)");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must have shape (n_atoms, 3)");
    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "cell must have shape (3, 3)");
    TORCH_CHECK(pbc.dim() == 1 && pbc.size(0) == 3,
                "pbc must have shape (3,)");
    TORCH_CHECK(cell.device() == positions.device(),
                "cell must be on same device as positions");
    TORCH_CHECK(pbc.device() == positions.device(),
                "pbc must be on same device as positions");
}

// Python-visible function with algorithm selection
std::tuple<torch::Tensor, torch::Tensor> compute_neighborlist(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff,
    const std::string& algorithm = "auto"
) {
    validate_inputs(positions, cell, pbc);

    // Convert to double precision for kernel (matches original hip_nl)
    auto positions_d = positions.to(torch::kFloat64).contiguous();
    auto cell_d = cell.to(torch::kFloat64).contiguous();
    auto pbc_bool = pbc.to(torch::kBool).contiguous();

    // Select algorithm
    std::string algo = algorithm;
    if (algo == "auto") {
        // Use cell_list for larger systems where V1 runs out of memory
        // V1 is faster but OOMs at ~16k atoms; V2 handles up to ~28k atoms
        algo = (positions.size(0) > 15000) ? "cell_list" : "direct";
    }

    if (algo == "cell_list" || algo == "v2") {
        return hip_compute_neighborlist_cell_list(
            positions_d, cell_d, pbc_bool, cutoff
        );
    } else {
        // Default to direct pairwise (v1)
        // Accepts: "direct", "brute_force", "v1", or any other string
        return hip_compute_neighborlist_kernel(
            positions_d, cell_d, pbc_bool, cutoff, false, false
        );
    }
}

// Explicit V1 function for testing/benchmarking
std::tuple<torch::Tensor, torch::Tensor> compute_neighborlist_v1(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff
) {
    validate_inputs(positions, cell, pbc);

    auto positions_d = positions.to(torch::kFloat64).contiguous();
    auto cell_d = cell.to(torch::kFloat64).contiguous();
    auto pbc_bool = pbc.to(torch::kBool).contiguous();

    return hip_compute_neighborlist_kernel(
        positions_d, cell_d, pbc_bool, cutoff, false, false
    );
}

// Explicit V2 function for testing/benchmarking
std::tuple<torch::Tensor, torch::Tensor> compute_neighborlist_v2(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff
) {
    validate_inputs(positions, cell, pbc);

    auto positions_d = positions.to(torch::kFloat64).contiguous();
    auto cell_d = cell.to(torch::kFloat64).contiguous();
    auto pbc_bool = pbc.to(torch::kBool).contiguous();

    return hip_compute_neighborlist_cell_list(
        positions_d, cell_d, pbc_bool, cutoff
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_neighborlist", &compute_neighborlist,
          "HIP-accelerated neighbor list computation (PyTorch extension)\n\n"
          "Args:\n"
          "    positions: Atomic positions tensor (n_atoms, 3) on GPU\n"
          "    cell: Unit cell tensor (3, 3) on GPU\n"
          "    pbc: Periodic boundary conditions (3,) on GPU\n"
          "    cutoff: Neighbor cutoff distance\n"
          "    algorithm: 'auto' (default), 'direct'/'v1', or 'cell_list'/'v2'\n\n"
          "Returns:\n"
          "    mapping: Neighbor pairs (2, n_pairs)\n"
          "    shifts: Periodic shifts (n_pairs, 3)",
          py::arg("positions"),
          py::arg("cell"),
          py::arg("pbc"),
          py::arg("cutoff"),
          py::arg("algorithm") = "auto");

    m.def("compute_neighborlist_v1", &compute_neighborlist_v1,
          "V1: Direct pairwise O(n²) neighbor list - faster but memory-limited (~16k atoms max)",
          py::arg("positions"),
          py::arg("cell"),
          py::arg("pbc"),
          py::arg("cutoff"));

    m.def("compute_neighborlist_v2", &compute_neighborlist_v2,
          "V2: Cell-list O(n) neighbor list - handles larger systems (~28k atoms)",
          py::arg("positions"),
          py::arg("cell"),
          py::arg("pbc"),
          py::arg("cutoff"));
}
