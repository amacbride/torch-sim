/*
 * Simplified HIP implementation without wave primitives
 * Each thread independently writes pairs using atomicAdd
 */

#include <hip/hip_runtime.h>
#include <cmath>

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

__device__ inline double3 apply_mic(double3 vec, const double3 box[3],
                                     const double3 inv_box[3], const bool periodic[3], int3& shift) {
    shift = make_int3(0, 0, 0);

    // Project onto box vectors
    double3 frac = make_double3(
        dot(vec, inv_box[0]),
        dot(vec, inv_box[1]),
        dot(vec, inv_box[2])
    );

    // Apply minimum image for periodic dimensions
    if (periodic[0]) {
        shift.x = -round(frac.x);
        frac.x += shift.x;
    }
    if (periodic[1]) {
        shift.y = -round(frac.y);
        frac.y += shift.y;
    }
    if (periodic[2]) {
        shift.z = -round(frac.z);
        frac.z += shift.z;
    }

    // Convert back to Cartesian
    return box[0] * frac.x + box[1] * frac.y + box[2] * frac.z;
}

// Simple kernel - one thread per (i,j) pair
__global__ void compute_neighbors_simple(
    const double3* positions,
    size_t n_points,
    const double3 box[3],
    const double3 inv_box[3],
    const bool periodic[3],
    double cutoff,
    ulong2* pairs,
    int3* shifts_out,
    size_t* n_pairs
) {
    // Each thread handles one potential pair
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_pairs = n_points * n_points;

    if (idx >= total_pairs) return;

    size_t i = idx / n_points;
    size_t j = idx % n_points;

    double3 pos_i = positions[i];
    double3 pos_j = positions[j];
    double3 vec = pos_j - pos_i;

    int3 shift;
    vec = apply_mic(vec, box, inv_box, periodic, shift);

    double dist_sq = dot(vec, vec);
    double cutoff_sq = cutoff * cutoff;

    // Check if this is a valid pair
    bool valid = (dist_sq < cutoff_sq) && (i != j || shift.x != 0 || shift.y != 0 || shift.z != 0);

    if (valid) {
        size_t out_idx = atomicAdd((unsigned long long*)n_pairs, 1ULL);
        pairs[out_idx] = make_ulong2(i, j);
        shifts_out[out_idx] = shift;
    }
}

extern "C" {

hipError_t hip_compute_neighborlist(
    const double* positions,      // [n_points, 3]
    size_t n_points,
    const double* box,            // [3, 3] row vectors
    const bool* periodic,         // [3]
    double cutoff,
    unsigned long** pairs_out,    // Output: [n_pairs, 2]
    int** shifts_out,             // Output: [n_pairs, 3]
    double** distances_out,       // Output: [n_pairs] or nullptr
    double** vectors_out,         // Output: [n_pairs, 3] or nullptr
    size_t* n_pairs_out,
    bool compute_distances,
    bool compute_vectors
) {
    // Convert input arrays to double3 format
    double3* h_positions = (double3*)malloc(n_points * sizeof(double3));
    for (size_t i = 0; i < n_points; i++) {
        h_positions[i] = make_double3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    }

    double3 h_box[3];
    for (int i = 0; i < 3; i++) {
        h_box[i] = make_double3(box[i*3], box[i*3+1], box[i*3+2]);
    }

    double3 h_inv_box[3];
    invert_3x3(h_box, h_inv_box);

    bool h_periodic[3] = {periodic[0], periodic[1], periodic[2]};

    // Allocate device memory
    double3* d_positions;
    double3* d_box;
    double3* d_inv_box;
    bool* d_periodic;
    ulong2* d_pairs;
    int3* d_shifts;
    size_t* d_n_pairs;

    size_t max_pairs = n_points * n_points;  // Worst case

    hipMalloc(&d_positions, n_points * sizeof(double3));
    hipMalloc(&d_box, 3 * sizeof(double3));
    hipMalloc(&d_inv_box, 3 * sizeof(double3));
    hipMalloc(&d_periodic, 3 * sizeof(bool));
    hipMalloc(&d_pairs, max_pairs * sizeof(ulong2));
    hipMalloc(&d_shifts, max_pairs * sizeof(int3));
    hipMalloc(&d_n_pairs, sizeof(size_t));

    // Copy to device
    hipMemcpy(d_positions, h_positions, n_points * sizeof(double3), hipMemcpyHostToDevice);
    hipMemcpy(d_box, h_box, 3 * sizeof(double3), hipMemcpyHostToDevice);
    hipMemcpy(d_inv_box, h_inv_box, 3 * sizeof(double3), hipMemcpyHostToDevice);
    hipMemcpy(d_periodic, h_periodic, 3 * sizeof(bool), hipMemcpyHostToDevice);
    hipMemset(d_n_pairs, 0, sizeof(size_t));

    // Launch kernel
    int threads = 256;
    size_t total_work = n_points * n_points;
    int blocks = (total_work + threads - 1) / threads;

    hipLaunchKernelGGL(compute_neighbors_simple,
                       dim3(blocks), dim3(threads), 0, 0,
                       d_positions, n_points, d_box, d_inv_box, d_periodic,
                       cutoff, d_pairs, d_shifts, d_n_pairs);

    // Copy results back
    size_t n_pairs;
    hipMemcpy(&n_pairs, d_n_pairs, sizeof(size_t), hipMemcpyDeviceToHost);
    *n_pairs_out = n_pairs;

    *pairs_out = (unsigned long*)malloc(n_pairs * 2 * sizeof(unsigned long));
    *shifts_out = (int*)malloc(n_pairs * 3 * sizeof(int));

    hipMemcpy(*pairs_out, d_pairs, n_pairs * sizeof(ulong2), hipMemcpyDeviceToHost);
    hipMemcpy(*shifts_out, d_shifts, n_pairs * sizeof(int3), hipMemcpyDeviceToHost);

    // Clean up
    free(h_positions);
    hipFree(d_positions);
    hipFree(d_box);
    hipFree(d_inv_box);
    hipFree(d_periodic);
    hipFree(d_pairs);
    hipFree(d_shifts);
    hipFree(d_n_pairs);

    return hipGetLastError();
}

} // extern "C"
