// Shared-memory-optimized version of the naive kernel.
//
// The optimization: in distance_naive.cu, every one of the `threads_per_block`
// threads independently re-reads the full `dim`-length query vector from
// global memory. Global memory bandwidth is the GPU's scarcest resource, and
// that traffic is 100% redundant -- every thread in a block wants the exact
// same query vector.
//
// Fix: have the block cooperatively load the query vector into `__shared__`
// memory ONCE (each thread loads a slice, then __syncthreads() before anyone
// reads), then have every thread read the query values from shared memory
// (on-chip, ~100x lower latency than global memory) for the rest of the
// kernel. This is the same "stage data on-chip, reuse across threads in the
// block" idea behind tiled matrix multiplication / cuBLAS-style GEMM kernels.
//
// What this does NOT yet do (documented honestly, not hidden):
// It does not tile the CANDIDATE vectors into shared memory -- each thread
// still streams its own candidate vector from global memory once. That's a
// reasonable next optimization if you want to push further (see README
// "Future work"), but the redundant-query-read is the single biggest win
// for this access pattern, since it's read `threads_per_block` times in the
// naive version vs. once per block here.
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

// dim is passed as a runtime value, so shared memory is allocated dynamically
// (third kernel-launch parameter) rather than as a compile-time-sized array.
__global__ void l2_tiled_kernel(const float* __restrict__ store,
                                 const float* __restrict__ query,
                                 size_t num_vectors, size_t dim,
                                 float* __restrict__ out_scores) {
    extern __shared__ float shared_query[];

    // Cooperative load: each thread loads a strided slice of the query
    // vector into shared memory. This runs once per block, not once per
    // thread's full distance computation.
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        shared_query[d] = query[d];
    }
    __syncthreads();  // every thread must wait until the full query vector is staged

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float diff = shared_query[d] - candidate[d];  // shared memory read instead of global
        sum += diff * diff;
    }
    out_scores[i] = sqrtf(sum);
}

__global__ void cosine_tiled_kernel(const float* __restrict__ store,
                                     const float* __restrict__ query,
                                     size_t num_vectors, size_t dim,
                                     float* __restrict__ out_scores) {
    extern __shared__ float shared_query[];

    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        shared_query[d] = query[d];
    }
    __syncthreads();

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float dot = 0.0f, norm_q = 0.0f, norm_c = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float q = shared_query[d];
        float c = candidate[d];
        dot += q * c;
        norm_q += q * q;
        norm_c += c * c;
    }
    float denom = sqrtf(norm_q) * sqrtf(norm_c);
    out_scores[i] = denom > 0.0f ? dot / denom : 0.0f;
}

} // namespace

void batch_distance_tiled_cuda(const float* h_store, size_t num_vectors, size_t dim,
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
    const size_t shared_mem_bytes = dim * sizeof(float);

    // Most GPUs cap default shared memory per block around 48KB. At dim=384
    // that's 1536 bytes -- comfortably under the limit. If you push `dim`
    // into the tens of thousands you'd need cudaFuncSetAttribute to opt into
    // the larger (up to ~228KB on recent architectures) shared memory carveout.

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    if (metric == Metric::L2) {
        l2_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
            d_store, d_query, num_vectors, dim, d_scores);
    } else {
        cosine_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
            d_store, d_query, num_vectors, dim, d_scores);
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
