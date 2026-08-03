// Naive CUDA implementation: one thread per candidate vector. Each thread
// independently reads the ENTIRE query vector from global memory, plus its
// own candidate vector from global memory. This is intentionally the
// "obviously correct, obviously not optimized" baseline.
//
// Why this is slow (the thing to be able to explain in an interview):
// If a block has 256 threads, all 256 threads redundantly re-read the same
// `dim`-length query vector from global memory. Global memory bandwidth is
// the scarce resource on GPUs -- L2 cache helps some, but this is still a
// large amount of avoidable traffic. The tiled version (distance_tiled.cu)
// fixes exactly this by loading the query vector into on-chip shared
// memory ONCE per block, then having every thread in that block read from
// shared memory (which is roughly two orders of magnitude faster than
// global memory) instead of hitting global memory again.
#include "distance_cuda.cuh"
#include "cuda_timing.cuh"
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

__global__ void l2_naive_kernel(const float* __restrict__ store,
                                 const float* __restrict__ query,
                                 size_t num_vectors, size_t dim,
                                 float* __restrict__ out_scores) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float diff = query[d] - candidate[d];   // query[d] re-fetched from global memory by EVERY thread
        sum += diff * diff;
    }
    out_scores[i] = sqrtf(sum);
}

__global__ void cosine_naive_kernel(const float* __restrict__ store,
                                     const float* __restrict__ query,
                                     size_t num_vectors, size_t dim,
                                     float* __restrict__ out_scores) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float dot = 0.0f, norm_q = 0.0f, norm_c = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float q = query[d];
        float c = candidate[d];
        dot += q * c;
        norm_q += q * q;
        norm_c += c * c;
    }
    float denom = sqrtf(norm_q) * sqrtf(norm_c);
    out_scores[i] = denom > 0.0f ? dot / denom : 0.0f;
}

} // namespace

void batch_distance_naive_cuda(const float* h_store, size_t num_vectors, size_t dim,
                                const float* h_query, Metric metric,
                                float* h_out_scores) {
    float *d_store = nullptr, *d_query = nullptr, *d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_store, num_vectors * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, num_vectors * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_store, h_store, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = static_cast<int>((num_vectors + threads_per_block - 1) / threads_per_block);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    if (metric == Metric::L2) {
        l2_naive_kernel<<<blocks, threads_per_block>>>(d_store, d_query, num_vectors, dim, d_scores);
    } else {
        cosine_naive_kernel<<<blocks, threads_per_block>>>(d_store, d_query, num_vectors, dim, d_scores);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    set_last_kernel_time_ms(ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_out_scores, d_scores, num_vectors * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_store);
    cudaFree(d_query);
    cudaFree(d_scores);
}
