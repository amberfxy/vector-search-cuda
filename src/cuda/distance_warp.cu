// Phase 5A experiment: warp-per-vector L2 (NOT the default baseline).
//
// Mapping: one warp owns one candidate vector. Lanes cooperate along dim so
// consecutive lanes load consecutive floats of the same row (coalesced within
// a warp for that row). Partial sums are combined with __shfl_down_sync.
//
// Intentionally does NOT use shared memory — isolates the mapping change from
// the existing shared-query tiled baseline.
#include "gpu_vector_index.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while (0)

// Full warp mask for sm_75+ active warps (32 lanes).
constexpr unsigned kFullWarpMask = 0xffffffffu;

__device__ inline float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(kFullWarpMask, v, offset);
    }
    return v;
}

// Grid: one warp per candidate vector.
// blockDim.x must be a multiple of 32 (launch uses 256 = 8 warps/block).
__global__ void l2_warp_per_vector_kernel(const float* __restrict__ store,
                                           const float* __restrict__ query,
                                           size_t num_vectors, size_t dim,
                                           float* __restrict__ out_scores) {
    const int lane = threadIdx.x & 31;                 // 0..31
    const int warp_in_block = threadIdx.x >> 5;        // threadIdx.x / 32
    const int warps_per_block = blockDim.x >> 5;
    const size_t vec =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(warps_per_block) +
        static_cast<size_t>(warp_in_block);
    if (vec >= num_vectors) return;

    const float* candidate = store + vec * dim;
    float partial = 0.0f;

    // Lane k processes dimensions k, k+32, k+64, ... (handles dim not % 32)
    for (size_t d = static_cast<size_t>(lane); d < dim; d += 32) {
        const float diff = query[d] - candidate[d];
        partial += diff * diff;
    }

    const float total = warp_reduce_sum(partial);
    if (lane == 0) {
        out_scores[vec] = sqrtf(total);
    }
}

} // namespace

void launch_l2_warp_device(const float* d_store, const float* d_query,
                            size_t num_vectors, size_t dim, float* d_scores) {
    const int threads_per_block = 256;  // 8 warps
    const int warps_per_block = threads_per_block / 32;
    const int blocks = static_cast<int>(
        (num_vectors + static_cast<size_t>(warps_per_block) - 1) /
        static_cast<size_t>(warps_per_block));
    l2_warp_per_vector_kernel<<<blocks, threads_per_block>>>(
        d_store, d_query, num_vectors, dim, d_scores);
    CUDA_CHECK(cudaGetLastError());
}
