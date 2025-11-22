/*
 * PyTorch C++ Extension binding layer for HIP neighbor list
 *
 * This provides the Python-visible interface and delegates to the HIP kernel.
 */

#include <torch/extension.h>
#include <vector>

// Forward declaration of HIP kernel wrapper
std::tuple<torch::Tensor, torch::Tensor> hip_compute_neighborlist_kernel(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff,
    bool compute_distances,
    bool compute_vectors
);

// Python-visible function
std::tuple<torch::Tensor, torch::Tensor> compute_neighborlist(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff
) {
    // Input validation
    TORCH_CHECK(positions.device().is_cuda(),
                "positions must be on GPU (CUDA/HIP device)");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must have shape (n_atoms, 3)");
    TORCH_CHECK(cell.dim() == 2 && cell.size(0) == 3 && cell.size(1) == 3,
                "cell must have shape (3, 3)");
    TORCH_CHECK(pbc.dim() == 1 && pbc.size(0) == 3,
                "pbc must have shape (3,)");

    // Ensure all tensors are on the same device
    TORCH_CHECK(cell.device() == positions.device(),
                "cell must be on same device as positions");
    TORCH_CHECK(pbc.device() == positions.device(),
                "pbc must be on same device as positions");

    // Convert to double precision for kernel (matches original hip_nl)
    auto positions_d = positions.to(torch::kFloat64).contiguous();
    auto cell_d = cell.to(torch::kFloat64).contiguous();
    auto pbc_bool = pbc.to(torch::kBool).contiguous();

    return hip_compute_neighborlist_kernel(
        positions_d, cell_d, pbc_bool, cutoff, false, false
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_neighborlist", &compute_neighborlist,
          "HIP-accelerated neighbor list computation (PyTorch extension)",
          py::arg("positions"),
          py::arg("cell"),
          py::arg("pbc"),
          py::arg("cutoff"));
}
