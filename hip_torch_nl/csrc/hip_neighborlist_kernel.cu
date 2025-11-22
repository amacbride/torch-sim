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
                shifts_out[out_idx] = shift;
            }
        }
    }
}

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
