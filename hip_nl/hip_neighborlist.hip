/*
 * HIP implementation of neighbor list computation
 * Based on vesin's CUDA implementation with minimum image convention
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>

// Wave size abstraction for AMD/NVIDIA portability
#ifdef __HIP_PLATFORM_AMD__
#define WAVESIZE 64
#else
#define WAVESIZE 32
#endif

#define NWARPS 4  // Number of warps per block
#define THREADS_PER_BLOCK (WAVESIZE * NWARPS)

// Note: HIP already provides double3, int3, ulong2, and make_* functions

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
__global__ void compute_neighbors_full(
    const double3* positions,
    size_t n_points,
    const double3 box[3],
    const double3 inv_box[3],
    const bool periodic[3],
    double cutoff,
    ulong2* pairs,
    int3* shifts_out,
    double* distances_out,
    double3* vectors_out,
    size_t* n_pairs,
    bool compute_distances,
    bool compute_vectors
) {
    // Shared memory for box matrices
    __shared__ double3 s_box[3];
    __shared__ double3 s_inv_box[3];
    __shared__ bool s_periodic[3];
    __shared__ size_t s_counter;

    // Load box to shared memory
    if (threadIdx.x < 3) {
        s_box[threadIdx.x] = box[threadIdx.x];
        s_inv_box[threadIdx.x] = inv_box[threadIdx.x];
        s_periodic[threadIdx.x] = periodic[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_counter = 0;
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

        bool valid = false;
        double3 vec;
        int3 shift;
        double dist_sq = 0.0;

        if (j < n_points) {
            double3 pos_j = positions[j];
            vec = pos_j - pos_i;

            // Apply minimum image convention
            vec = apply_mic(vec, s_box, s_inv_box, s_periodic, shift);

            dist_sq = dot(vec, vec);
            valid = (dist_sq < cutoff_sq) && (i != j || shift.x != 0 || shift.y != 0 || shift.z != 0);
        }

        // Use wave ballot to find valid pairs
        unsigned long long ballot = __ballot(valid);
        int n_valid = __popcll(ballot);

        // Reserve space in output
        size_t output_base = 0;
        if (lane_id == 0 && n_valid > 0) {
            output_base = atomicAdd((unsigned long long*)&s_counter, (unsigned long long)n_valid);
        }
        // Broadcast base index to all threads in wave
        // On AMD, __shfl needs explicit wave size
        output_base = __shfl(output_base, 0, WAVESIZE);

        // Write outputs
        if (valid) {
            // Count how many valid pairs before this thread
            unsigned long long mask = (1ULL << lane_id) - 1;
            int offset = __popcll(ballot & mask);
            size_t out_idx = output_base + offset;

            pairs[out_idx] = make_ulong2(i, j);
            shifts_out[out_idx] = shift;

            if (compute_distances) {
                distances_out[out_idx] = sqrt(dist_sq);
            }
            if (compute_vectors) {
                vectors_out[out_idx] = vec;
            }
        }
    }

    // Write final count
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)n_pairs, (unsigned long long)s_counter);
    }
}

// Host function to launch kernel
extern "C" {

hipError_t hip_compute_neighborlist(
    const double* positions,     // [n_points, 3]
    size_t n_points,
    const double* box,           // [3, 3] row vectors
    const bool* periodic,        // [3]
    double cutoff,
    unsigned long** pairs_out,   // Output: [n_pairs, 2]
    int** shifts_out,            // Output: [n_pairs, 3]
    double** distances_out,      // Output: [n_pairs] or nullptr
    double** vectors_out,        // Output: [n_pairs, 3] or nullptr
    size_t* n_pairs_out,
    bool compute_distances,
    bool compute_vectors
) {
    // Prepare box matrices
    double3 h_box[3], h_inv_box[3];
    bool h_periodic[3];

    // Copy and convert box
    for (int i = 0; i < 3; i++) {
        h_box[i].x = box[i * 3 + 0];
        h_box[i].y = box[i * 3 + 1];
        h_box[i].z = box[i * 3 + 2];
        h_periodic[i] = periodic[i];
    }

    // Compute inverse box
    invert_3x3(h_box, h_inv_box);

    // Copy to device constant memory
    double3 *d_box, *d_inv_box;
    bool *d_periodic;
    hipMalloc(&d_box, 3 * sizeof(double3));
    hipMalloc(&d_inv_box, 3 * sizeof(double3));
    hipMalloc(&d_periodic, 3 * sizeof(bool));

    hipMemcpy(d_box, h_box, 3 * sizeof(double3), hipMemcpyHostToDevice);
    hipMemcpy(d_inv_box, h_inv_box, 3 * sizeof(double3), hipMemcpyHostToDevice);
    hipMemcpy(d_periodic, h_periodic, 3 * sizeof(bool), hipMemcpyHostToDevice);

    // Allocate device memory for positions
    double3* d_positions;
    hipMalloc(&d_positions, n_points * sizeof(double3));
    hipMemcpy(d_positions, positions, n_points * sizeof(double3), hipMemcpyHostToDevice);

    // Allocate output buffers (over-allocate to be safe)
    size_t max_pairs = n_points * n_points;  // Worst case
    ulong2* d_pairs;
    int3* d_shifts;
    double* d_distances = nullptr;
    double3* d_vectors = nullptr;
    size_t* d_n_pairs;

    hipMalloc(&d_pairs, max_pairs * sizeof(ulong2));
    hipMalloc(&d_shifts, max_pairs * sizeof(int3));
    hipMalloc(&d_n_pairs, sizeof(size_t));
    hipMemset(d_n_pairs, 0, sizeof(size_t));

    if (compute_distances) {
        hipMalloc(&d_distances, max_pairs * sizeof(double));
    }
    if (compute_vectors) {
        hipMalloc(&d_vectors, max_pairs * sizeof(double3));
    }

    // Launch kernel
    int n_blocks = (n_points + NWARPS - 1) / NWARPS;
    hipLaunchKernelGGL(compute_neighbors_full,
                       dim3(n_blocks), dim3(THREADS_PER_BLOCK), 0, 0,
                       d_positions, n_points,
                       d_box, d_inv_box, d_periodic,
                       cutoff,
                       d_pairs, d_shifts, d_distances, d_vectors,
                       d_n_pairs,
                       compute_distances, compute_vectors);

    // Get number of pairs
    size_t n_pairs;
    hipMemcpy(&n_pairs, d_n_pairs, sizeof(size_t), hipMemcpyDeviceToHost);
    *n_pairs_out = n_pairs;

    // Allocate and copy results
    *pairs_out = (unsigned long*)malloc(n_pairs * 2 * sizeof(unsigned long));
    *shifts_out = (int*)malloc(n_pairs * 3 * sizeof(int));

    hipMemcpy(*pairs_out, d_pairs, n_pairs * sizeof(ulong2), hipMemcpyDeviceToHost);
    hipMemcpy(*shifts_out, d_shifts, n_pairs * sizeof(int3), hipMemcpyDeviceToHost);

    if (compute_distances && distances_out) {
        *distances_out = (double*)malloc(n_pairs * sizeof(double));
        hipMemcpy(*distances_out, d_distances, n_pairs * sizeof(double), hipMemcpyDeviceToHost);
    }

    if (compute_vectors && vectors_out) {
        *vectors_out = (double*)malloc(n_pairs * 3 * sizeof(double));
        hipMemcpy(*vectors_out, d_vectors, n_pairs * sizeof(double3), hipMemcpyDeviceToHost);
    }

    // Cleanup
    hipFree(d_positions);
    hipFree(d_box);
    hipFree(d_inv_box);
    hipFree(d_periodic);
    hipFree(d_pairs);
    hipFree(d_shifts);
    hipFree(d_n_pairs);
    if (d_distances) hipFree(d_distances);
    if (d_vectors) hipFree(d_vectors);

    return hipGetLastError();
}

}  // extern "C"
