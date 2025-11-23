/*
 * HIP kernel for neighbor list computation - PyTorch Extension version
 *
 * This version integrates with PyTorch's HIP context and memory management,
 * solving the context conflict issue present in the standalone hip_nl.
 *
 * Based on vesin's CUDA implementation with minimum image convention.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <cmath>

// Wave size abstraction for AMD/NVIDIA portability
#ifdef __HIP_PLATFORM_AMD__
#define WAVESIZE 64
#else
#define WAVESIZE 32
#endif

#define NWARPS 4  // Number of warps per block
#define THREADS_PER_BLOCK (WAVESIZE * NWARPS)

// Helper functions
__device__ inline double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline double3 operator+(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline double3 operator-(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline double3 operator*(double3 a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__device__ inline double3 operator*(double s, double3 a) {
    return a * s;
}

// Compute inverse of 3x3 matrix
__host__ __device__ inline void invert_3x3(const double3 box[3], double3 inv[3]) {
    double det = box[0].x * (box[1].y * box[2].z - box[1].z * box[2].y) -
                 box[0].y * (box[1].x * box[2].z - box[1].z * box[2].x) +
                 box[0].z * (box[1].x * box[2].y - box[1].y * box[2].x);

    double inv_det = 1.0 / det;

    inv[0].x = (box[1].y * box[2].z - box[1].z * box[2].y) * inv_det;
    inv[0].y = (box[0].z * box[2].y - box[0].y * box[2].z) * inv_det;
    inv[0].z = (box[0].y * box[1].z - box[0].z * box[1].y) * inv_det;

    inv[1].x = (box[1].z * box[2].x - box[1].x * box[2].z) * inv_det;
    inv[1].y = (box[0].x * box[2].z - box[0].z * box[2].x) * inv_det;
    inv[1].z = (box[0].z * box[1].x - box[0].x * box[1].z) * inv_det;

    inv[2].x = (box[1].x * box[2].y - box[1].y * box[2].x) * inv_det;
    inv[2].y = (box[0].y * box[2].x - box[0].x * box[2].y) * inv_det;
    inv[2].z = (box[0].x * box[1].y - box[0].y * box[1].x) * inv_det;
}

// Apply minimum image convention
__device__ double3 apply_mic(double3 vec, const double3 box[3], const double3 inv_box[3],
                              const bool periodic[3], int3& shifts) {
    // Convert to fractional coordinates
    double3 frac = make_double3(
        dot(vec, inv_box[0]),
        dot(vec, inv_box[1]),
        dot(vec, inv_box[2])
    );

    // Apply periodic wrapping
    shifts = make_int3(0, 0, 0);

    if (periodic[0]) {
        shifts.x = (int)round(frac.x);
        frac.x -= shifts.x;
    }
    if (periodic[1]) {
        shifts.y = (int)round(frac.y);
        frac.y -= shifts.y;
    }
    if (periodic[2]) {
        shifts.z = (int)round(frac.z);
        frac.z -= shifts.z;
    }

    // Convert back to Cartesian
    return make_double3(
        frac.x * box[0].x + frac.y * box[1].x + frac.z * box[2].x,
        frac.x * box[0].y + frac.y * box[1].y + frac.z * box[2].y,
        frac.x * box[0].z + frac.y * box[1].z + frac.z * box[2].z
    );
}

// Main kernel for computing neighbor list (full pairs)
// Simplified version using per-thread atomic output (no wave ballot optimization)
// This avoids potential issues with wave-level operations on AMD GPUs
__global__ void compute_neighbors_full(
    const double3* positions,
    size_t n_points,
    const double3 box[3],
    const double3 inv_box[3],
    const bool periodic[3],
    double cutoff,
    ulong2* pairs,
    int3* shifts_out,
    size_t* n_pairs  // Global atomic counter for output indexing
) {
    // Shared memory for box matrices
    __shared__ double3 s_box[3];
    __shared__ double3 s_inv_box[3];
    __shared__ bool s_periodic[3];

    // Load box to shared memory
    if (threadIdx.x < 3) {
        s_box[threadIdx.x] = box[threadIdx.x];
        s_inv_box[threadIdx.x] = inv_box[threadIdx.x];
        s_periodic[threadIdx.x] = periodic[threadIdx.x];
    }
    __syncthreads();

    // Calculate which point this warp processes
    int warp_id = threadIdx.x / WAVESIZE;
    int lane_id = threadIdx.x % WAVESIZE;
    size_t i = blockIdx.x * NWARPS + warp_id;

    if (i >= n_points) return;

    double3 pos_i = positions[i];
    double cutoff_sq = cutoff * cutoff;

    // Each thread in warp checks different j values
    for (size_t j_base = 0; j_base < n_points; j_base += WAVESIZE) {
        size_t j = j_base + lane_id;

        if (j < n_points) {
            double3 pos_j = positions[j];
            double3 vec = pos_j - pos_i;

            // Apply minimum image convention
            int3 shift;
            vec = apply_mic(vec, s_box, s_inv_box, s_periodic, shift);

            double dist_sq = dot(vec, vec);
            bool valid = (dist_sq < cutoff_sq) && (i != j || shift.x != 0 || shift.y != 0 || shift.z != 0);

            // If valid pair found, atomically reserve output slot and write
            if (valid) {
                size_t out_idx = atomicAdd((unsigned long long*)n_pairs, 1ULL);
                pairs[out_idx] = make_ulong2(i, j);
                // Negate shift to match standard_nl convention:
                // standard_nl: D = pos[j] - pos[i] + S @ cell
                // Our shift is from MIC: we subtracted shift to get minimum image
                shifts_out[out_idx] = make_int3(-shift.x, -shift.y, -shift.z);
            }
        }
    }
}

// ============================================================================
// V2: Cell-List Algorithm O(n) - Better for large systems (>5000 atoms)
// ============================================================================

// Kernel to assign atoms to cells and count atoms per cell
__global__ void assign_atoms_to_cells(
    const double3* positions,
    size_t n_points,
    const double3 inv_box[3],
    int3 n_cells,
    int* cell_indices,      // Output: cell index for each atom
    int* cell_counts        // Output: number of atoms in each cell (atomically incremented)
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    double3 pos = positions[i];

    // Convert to fractional coordinates [0, 1)
    double3 frac = make_double3(
        dot(pos, inv_box[0]),
        dot(pos, inv_box[1]),
        dot(pos, inv_box[2])
    );

    // Wrap to [0, 1)
    frac.x = frac.x - floor(frac.x);
    frac.y = frac.y - floor(frac.y);
    frac.z = frac.z - floor(frac.z);

    // Compute cell indices
    int cx = min((int)(frac.x * n_cells.x), n_cells.x - 1);
    int cy = min((int)(frac.y * n_cells.y), n_cells.y - 1);
    int cz = min((int)(frac.z * n_cells.z), n_cells.z - 1);

    // Linear cell index
    int cell_idx = cx + n_cells.x * (cy + n_cells.y * cz);
    cell_indices[i] = cell_idx;

    // Atomically increment count for this cell
    atomicAdd(&cell_counts[cell_idx], 1);
}

// Kernel to build cell lists (scatter atoms into cell arrays)
__global__ void build_cell_lists(
    size_t n_points,
    const int* cell_indices,
    int* cell_offsets,      // Input: prefix sum of cell counts (start index for each cell)
    int* cell_counts,       // Input/Output: current count per cell (used as local offset)
    int* cell_atoms         // Output: atom indices organized by cell
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    int cell_idx = cell_indices[i];
    int local_idx = atomicAdd(&cell_counts[cell_idx], 1);
    int global_idx = cell_offsets[cell_idx] + local_idx;
    cell_atoms[global_idx] = (int)i;
}

// Main V2 kernel: search neighbors using cell lists
__global__ void compute_neighbors_cell_list(
    const double3* positions,
    size_t n_points,
    const double3 box[3],
    const double3 inv_box[3],
    const bool periodic[3],
    double cutoff,
    int3 n_cells,
    const int* cell_indices,
    const int* cell_offsets,
    const int* cell_atom_counts,  // Original counts (not modified)
    const int* cell_atoms,
    uint2* pairs,           // Changed from ulong2 to uint2 (int32) - saves 50% memory
    int3* shifts_out,
    size_t* n_pairs,
    size_t max_pairs  // Buffer size - pairs beyond this are counted but not written
) {
    // Shared memory for box matrices
    __shared__ double3 s_box[3];
    __shared__ double3 s_inv_box[3];
    __shared__ bool s_periodic[3];

    if (threadIdx.x < 3) {
        s_box[threadIdx.x] = box[threadIdx.x];
        s_inv_box[threadIdx.x] = inv_box[threadIdx.x];
        s_periodic[threadIdx.x] = periodic[threadIdx.x];
    }
    __syncthreads();

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_points) return;

    double3 pos_i = positions[i];
    double cutoff_sq = cutoff * cutoff;
    int cell_i = cell_indices[i];

    // Compute 3D cell coordinates for atom i
    int cz_i = cell_i / (n_cells.x * n_cells.y);
    int rem = cell_i % (n_cells.x * n_cells.y);
    int cy_i = rem / n_cells.x;
    int cx_i = rem % n_cells.x;

    // Iterate over 27 neighboring cells (including self)
    for (int dcz = -1; dcz <= 1; dcz++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            for (int dcx = -1; dcx <= 1; dcx++) {
                int cx_j = cx_i + dcx;
                int cy_j = cy_i + dcy;
                int cz_j = cz_i + dcz;

                // Cell shift for periodic boundaries
                int3 cell_shift = make_int3(0, 0, 0);

                // Handle periodic wrapping
                // Sign convention: shift = round(frac) where frac = (j-i) / cell
                // If we access cell -1 (wrapped to n-1), j is on the "far side"
                // The vector j-i spans most of the box, so shift should be +1
                if (s_periodic[0]) {
                    if (cx_j < 0) { cx_j += n_cells.x; cell_shift.x = +1; }
                    else if (cx_j >= n_cells.x) { cx_j -= n_cells.x; cell_shift.x = -1; }
                } else {
                    if (cx_j < 0 || cx_j >= n_cells.x) continue;
                }

                if (s_periodic[1]) {
                    if (cy_j < 0) { cy_j += n_cells.y; cell_shift.y = +1; }
                    else if (cy_j >= n_cells.y) { cy_j -= n_cells.y; cell_shift.y = -1; }
                } else {
                    if (cy_j < 0 || cy_j >= n_cells.y) continue;
                }

                if (s_periodic[2]) {
                    if (cz_j < 0) { cz_j += n_cells.z; cell_shift.z = +1; }
                    else if (cz_j >= n_cells.z) { cz_j -= n_cells.z; cell_shift.z = -1; }
                } else {
                    if (cz_j < 0 || cz_j >= n_cells.z) continue;
                }

                int cell_j = cx_j + n_cells.x * (cy_j + n_cells.y * cz_j);
                int start_j = cell_offsets[cell_j];
                int count_j = cell_atom_counts[cell_j];

                // Check all atoms in neighboring cell
                for (int k = 0; k < count_j; k++) {
                    int j = cell_atoms[start_j + k];

                    double3 pos_j = positions[j];
                    double3 vec = pos_j - pos_i;

                    // Apply MIC to get minimum image and shift
                    // MIC handles all periodic wrapping - cell_shift just tracks
                    // which periodic image we expect to find the pair in
                    int3 shift;
                    vec = apply_mic(vec, s_box, s_inv_box, s_periodic, shift);

                    double dist_sq = dot(vec, vec);

                    // Check if this is a valid pair
                    // For the same cell (cell_shift=0), accept the pair if within cutoff
                    // For wrapped cells (cell_shift!=0), only accept if the MIC shift matches
                    // to avoid counting pairs multiple times through different cells
                    bool shift_matches = (shift.x == cell_shift.x &&
                                         shift.y == cell_shift.y &&
                                         shift.z == cell_shift.z);
                    bool valid = (dist_sq < cutoff_sq) && shift_matches &&
                                 (i != j || shift.x != 0 || shift.y != 0 || shift.z != 0);

                    if (valid) {
                        size_t out_idx = atomicAdd((unsigned long long*)n_pairs, 1ULL);
                        // Only write if within buffer bounds (count overflow but don't write)
                        if (out_idx < max_pairs) {
                            pairs[out_idx] = make_uint2((unsigned int)i, (unsigned int)j);
                            // Negate shift to match standard_nl convention:
                            // standard_nl: D = pos[j] - pos[i] + S @ cell
                            shifts_out[out_idx] = make_int3(-shift.x, -shift.y, -shift.z);
                        }
                        // else: pair is counted but not written (overflow detected on host)
                    }
                }
            }
        }
    }
}

// V2 Host wrapper: Cell-list algorithm
std::tuple<torch::Tensor, torch::Tensor> hip_compute_neighborlist_cell_list(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff
) {
    auto device = positions.device();
    const size_t n_points = positions.size(0);

    // Prepare box matrices on host
    double3 h_box[3], h_inv_box[3];
    bool h_periodic[3];

    auto cell_cpu = cell.cpu();
    auto pbc_cpu = pbc.cpu();

    for (int i = 0; i < 3; i++) {
        h_box[i].x = cell_cpu[i][0].item<double>();
        h_box[i].y = cell_cpu[i][1].item<double>();
        h_box[i].z = cell_cpu[i][2].item<double>();
        h_periodic[i] = pbc_cpu[i].item<bool>();
    }

    invert_3x3(h_box, h_inv_box);

    // Compute cell dimensions: each cell should be at least cutoff in size
    // Use reciprocal lattice to get face distances
    double face_dist[3];
    for (int i = 0; i < 3; i++) {
        double len = sqrt(h_inv_box[i].x * h_inv_box[i].x +
                         h_inv_box[i].y * h_inv_box[i].y +
                         h_inv_box[i].z * h_inv_box[i].z);
        face_dist[i] = (len > 0) ? 1.0 / len : 1.0;
    }

    int3 n_cells;
    n_cells.x = std::max(1, (int)(face_dist[0] / cutoff));
    n_cells.y = std::max(1, (int)(face_dist[1] / cutoff));
    n_cells.z = std::max(1, (int)(face_dist[2] / cutoff));
    int total_cells = n_cells.x * n_cells.y * n_cells.z;

    // Allocate device memory
    auto options_d3 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(device);

    // Box matrices
    auto d_box = torch::empty({3, 3}, options_d3);
    auto d_inv_box = torch::empty({3, 3}, options_d3);
    auto d_periodic = torch::empty({3}, options_bool);

    {
        auto d_box_cpu = torch::empty({3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
        auto d_inv_box_cpu = torch::empty({3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
        auto d_periodic_cpu = torch::empty({3}, torch::TensorOptions().dtype(torch::kBool));

        for (int i = 0; i < 3; i++) {
            d_box_cpu[i][0] = h_box[i].x;
            d_box_cpu[i][1] = h_box[i].y;
            d_box_cpu[i][2] = h_box[i].z;
            d_inv_box_cpu[i][0] = h_inv_box[i].x;
            d_inv_box_cpu[i][1] = h_inv_box[i].y;
            d_inv_box_cpu[i][2] = h_inv_box[i].z;
            d_periodic_cpu[i] = h_periodic[i];
        }

        d_box.copy_(d_box_cpu);
        d_inv_box.copy_(d_inv_box_cpu);
        d_periodic.copy_(d_periodic_cpu);
    }

    // Cell data structures
    auto cell_indices = torch::empty({static_cast<long>(n_points)}, options_int);
    auto cell_counts = torch::zeros({total_cells}, options_int);
    auto cell_offsets = torch::empty({total_cells}, options_int);
    auto cell_atoms = torch::empty({static_cast<long>(n_points)}, options_int);

    hipStream_t stream = (hipStream_t)at::cuda::getDefaultCUDAStream();
    int threads = 256;
    int blocks = (n_points + threads - 1) / threads;

    // Phase 1: Assign atoms to cells
    hipLaunchKernelGGL(
        assign_atoms_to_cells,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<const double3*>(positions.data_ptr<double>()),
        n_points,
        reinterpret_cast<const double3*>(d_inv_box.data_ptr<double>()),
        n_cells,
        cell_indices.data_ptr<int>(),
        cell_counts.data_ptr<int>()
    );

    hipError_t err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "assign_atoms_to_cells error: ", hipGetErrorString(err));

    // Phase 2: Compute prefix sum for cell offsets (exclusive scan)
    // Use PyTorch's cumsum for simplicity
    hipDeviceSynchronize();
    auto cell_counts_copy = cell_counts.clone();  // Save original counts
    // cumsum might return Long, cast back to Int32
    cell_offsets = (torch::cumsum(cell_counts, 0) - cell_counts).to(torch::kInt32);
    cell_counts.zero_();  // Reset for scatter phase

    // Phase 3: Build cell lists (scatter atoms)
    hipLaunchKernelGGL(
        build_cell_lists,
        dim3(blocks), dim3(threads), 0, stream,
        n_points,
        cell_indices.data_ptr<int>(),
        cell_offsets.data_ptr<int>(),
        cell_counts.data_ptr<int>(),
        cell_atoms.data_ptr<int>()
    );

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "build_cell_lists error: ", hipGetErrorString(err));

    // Phase 4: Search neighbors
    // Memory-efficient pair estimation using density scaling:
    // For uniform density, pairs ≈ n * avg_neighbors where avg_neighbors = n * V_cutoff / V_box
    // V_cutoff = 4/3 * π * r³, so pairs ≈ n * n * (4π/3 * cutoff³) / V_box
    // This is O(n) for fixed density (n/V_box constant), O(n²) for fixed box size
    //
    // Calculate box volume from cell vectors
    double3 cross;
    cross.x = h_box[1].y * h_box[2].z - h_box[1].z * h_box[2].y;
    cross.y = h_box[1].z * h_box[2].x - h_box[1].x * h_box[2].z;
    cross.z = h_box[1].x * h_box[2].y - h_box[1].y * h_box[2].x;
    double box_volume = fabs(h_box[0].x * cross.x + h_box[0].y * cross.y + h_box[0].z * cross.z);

    // Cutoff sphere volume (this counts each pair twice since we output i->j and j->i)
    double cutoff_volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;

    // Expected pairs per atom = number_density * cutoff_volume * 2 (for both directions)
    double number_density = (double)n_points / box_volume;
    double avg_neighbors = number_density * cutoff_volume * 2.0;

    // Estimated pairs with 1.5x safety margin for non-uniformity
    // Minimum of 100 pairs per atom to handle sparse systems
    size_t estimated_pairs = (size_t)(n_points * std::max(avg_neighbors * 1.5, 100.0));

    // Cap estimation to avoid OOM - use incremental growth strategy instead
    // Each pair costs 20 bytes (8 for int32x2 + 12 for int32x3)
    // Start with reasonable allocation, rely on retry for large systems
    size_t bytes_per_pair = 20;
    size_t max_initial_bytes = 2UL * 1024 * 1024 * 1024;  // 2GB initial limit
    size_t max_initial_pairs = max_initial_bytes / bytes_per_pair;
    estimated_pairs = std::min(estimated_pairs, max_initial_pairs);

    // Use int32 for pair indices - sufficient for up to 2^31 atoms, saves 50% memory
    auto d_pairs = torch::empty({static_cast<long>(estimated_pairs), 2}, options_int);
    auto d_shifts = torch::empty({static_cast<long>(estimated_pairs), 3}, options_int);
    auto d_n_pairs = torch::zeros({1}, options_long);

    hipLaunchKernelGGL(
        compute_neighbors_cell_list,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<const double3*>(positions.data_ptr<double>()),
        n_points,
        reinterpret_cast<const double3*>(d_box.data_ptr<double>()),
        reinterpret_cast<const double3*>(d_inv_box.data_ptr<double>()),
        d_periodic.data_ptr<bool>(),
        cutoff,
        n_cells,
        cell_indices.data_ptr<int>(),
        cell_offsets.data_ptr<int>(),
        cell_counts_copy.data_ptr<int>(),
        cell_atoms.data_ptr<int>(),
        reinterpret_cast<uint2*>(d_pairs.data_ptr<int32_t>()),
        reinterpret_cast<int3*>(d_shifts.data_ptr<int32_t>()),
        reinterpret_cast<size_t*>(d_n_pairs.data_ptr<int64_t>()),
        estimated_pairs  // max_pairs for bounds checking
    );

    err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, "compute_neighbors_cell_list error: ", hipGetErrorString(err));

    hipDeviceSynchronize();

    size_t n_pairs_val = d_n_pairs.cpu().item<int64_t>();

    // Handle overflow: if we got more pairs than estimated, reallocate and retry
    if (n_pairs_val > estimated_pairs) {
        size_t new_size = n_pairs_val + 1000;
        d_pairs = torch::empty({static_cast<long>(new_size), 2}, options_int);
        d_shifts = torch::empty({static_cast<long>(new_size), 3}, options_int);
        d_n_pairs.zero_();

        hipLaunchKernelGGL(
            compute_neighbors_cell_list,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const double3*>(positions.data_ptr<double>()),
            n_points,
            reinterpret_cast<const double3*>(d_box.data_ptr<double>()),
            reinterpret_cast<const double3*>(d_inv_box.data_ptr<double>()),
            d_periodic.data_ptr<bool>(),
            cutoff,
            n_cells,
            cell_indices.data_ptr<int>(),
            cell_offsets.data_ptr<int>(),
            cell_counts_copy.data_ptr<int>(),
            cell_atoms.data_ptr<int>(),
            reinterpret_cast<uint2*>(d_pairs.data_ptr<int32_t>()),
            reinterpret_cast<int3*>(d_shifts.data_ptr<int32_t>()),
            reinterpret_cast<size_t*>(d_n_pairs.data_ptr<int64_t>()),
            new_size  // max_pairs for retry
        );

        hipDeviceSynchronize();
        n_pairs_val = d_n_pairs.cpu().item<int64_t>();
    }

    // Convert int32 pairs to int64 for output (PyTorch expects int64 indices)
    auto mapping = d_pairs.slice(0, 0, n_pairs_val).t().to(torch::kInt64).contiguous();
    auto shifts = d_shifts.slice(0, 0, n_pairs_val).to(torch::kFloat32);

    return std::make_tuple(mapping, shifts);
}

// ============================================================================
// V1: Brute-Force Algorithm O(n²) - Better for small systems (<5000 atoms)
// ============================================================================

// Host wrapper function that integrates with PyTorch
std::tuple<torch::Tensor, torch::Tensor> hip_compute_neighborlist_kernel(
    torch::Tensor positions,
    torch::Tensor cell,
    torch::Tensor pbc,
    double cutoff,
    bool compute_distances,
    bool compute_vectors
) {
    // Get device and stream from PyTorch
    auto device = positions.device();
    // Stream will be obtained later using getDefaultCUDAStream pattern

    const size_t n_points = positions.size(0);

    // Prepare box matrices on host
    double3 h_box[3], h_inv_box[3];
    bool h_periodic[3];

    // Copy cell data (already on GPU, need to access on CPU for matrix inversion)
    auto cell_cpu = cell.cpu();
    auto pbc_cpu = pbc.cpu();

    for (int i = 0; i < 3; i++) {
        h_box[i].x = cell_cpu[i][0].item<double>();
        h_box[i].y = cell_cpu[i][1].item<double>();
        h_box[i].z = cell_cpu[i][2].item<double>();
        h_periodic[i] = pbc_cpu[i].item<bool>();
    }

    // Compute inverse box on CPU
    invert_3x3(h_box, h_inv_box);

    // Allocate device memory for box matrices using PyTorch
    auto options_d3 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

    // Create tensors for box data
    auto d_box = torch::empty({3, 3}, options_d3);
    auto d_inv_box = torch::empty({3, 3}, options_d3);
    auto d_periodic = torch::empty({3}, options_bool);

    // Copy box data to device (as flat arrays, then reinterpret)
    {
        auto d_box_cpu = torch::empty({3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
        auto d_inv_box_cpu = torch::empty({3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
        auto d_periodic_cpu = torch::empty({3}, torch::TensorOptions().dtype(torch::kBool));

        for (int i = 0; i < 3; i++) {
            d_box_cpu[i][0] = h_box[i].x;
            d_box_cpu[i][1] = h_box[i].y;
            d_box_cpu[i][2] = h_box[i].z;
            d_inv_box_cpu[i][0] = h_inv_box[i].x;
            d_inv_box_cpu[i][1] = h_inv_box[i].y;
            d_inv_box_cpu[i][2] = h_inv_box[i].z;
            d_periodic_cpu[i] = h_periodic[i];
        }

        d_box.copy_(d_box_cpu);
        d_inv_box.copy_(d_inv_box_cpu);
        d_periodic.copy_(d_periodic_cpu);
    }

    // Allocate output buffers (over-allocate to be safe)
    // Using PyTorch's allocator ensures we share the same memory pool
    size_t max_pairs = n_points * n_points;

    auto options_ulong2 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto options_int3 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // pairs: [max_pairs, 2] as int64
    auto d_pairs = torch::empty({static_cast<long>(max_pairs), 2}, options_ulong2);
    // shifts: [max_pairs, 3] as int32
    auto d_shifts = torch::empty({static_cast<long>(max_pairs), 3}, options_int3);
    // counter
    auto d_n_pairs = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    // Launch kernel on PyTorch's stream
    int n_blocks = (n_points + NWARPS - 1) / NWARPS;

    // Get stream using the same pattern as the working example
    // PyTorch ROCm builds use the "cuda" namespace; it maps to HIP internally
    hipStream_t stream_h = (hipStream_t)at::cuda::getDefaultCUDAStream();

    hipLaunchKernelGGL(
        compute_neighbors_full,
        dim3(n_blocks), dim3(THREADS_PER_BLOCK),
        0, stream_h,
                       reinterpret_cast<const double3*>(positions.data_ptr<double>()),
                       n_points,
                       reinterpret_cast<const double3*>(d_box.data_ptr<double>()),
                       reinterpret_cast<const double3*>(d_inv_box.data_ptr<double>()),
                       d_periodic.data_ptr<bool>(),
                       cutoff,
                       reinterpret_cast<ulong2*>(d_pairs.data_ptr<int64_t>()),
                       reinterpret_cast<int3*>(d_shifts.data_ptr<int32_t>()),
                       reinterpret_cast<size_t*>(d_n_pairs.data_ptr<int64_t>()));

    // Check for launch errors
    hipError_t launch_err = hipGetLastError();
    TORCH_CHECK(launch_err == hipSuccess,
                "HIP kernel launch error: ", hipGetErrorString(launch_err));

    // Synchronize to get pair count
    hipError_t sync_err = hipDeviceSynchronize();
    TORCH_CHECK(sync_err == hipSuccess,
                "HIP sync error: ", hipGetErrorString(sync_err));

    // Get actual number of pairs
    size_t n_pairs = d_n_pairs.cpu().item<int64_t>();

    // Slice to actual size and transpose pairs to [2, n_pairs] format
    auto mapping = d_pairs.slice(0, 0, n_pairs).t().contiguous();
    auto shifts = d_shifts.slice(0, 0, n_pairs).to(torch::kFloat32);

    return std::make_tuple(mapping, shifts);
}
