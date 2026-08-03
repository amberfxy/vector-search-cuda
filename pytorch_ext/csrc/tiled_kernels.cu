// Same shared-memory-tiled optimization idea as src/cuda/distance_tiled.cu
// in the main project, but adapted for use as a PyTorch extension: these
// functions assume `query` and `store` are ALREADY on the GPU (as PyTorch
// CUDA tensors) and write results into an already-allocated output buffer.
// There is no cudaMalloc/cudaMemcpy/cudaFree here -- that's the whole
// point of wrapping this as a PyTorch op instead of the standalone
// benchmark harness. The standalone `bench_runner` in ../src/cuda/
// intentionally re-transfers data on every call (see the main README's
// "Known limitations" section); this extension exists specifically to
// show the alternative -- operating on device-resident tensors, the way
// a real inference-serving system would.
#include <cuda_runtime.h>
#include <cmath>

namespace vector_search_torch {

__global__ void l2_tiled_kernel(const float* __restrict__ store,
                                 const float* __restrict__ query,
                                 int64_t num_vectors, int64_t dim,
                                 float* __restrict__ out_scores) {
    extern __shared__ float shared_query[];

    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        shared_query[d] = query[d];
    }
    __syncthreads();

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float sum = 0.0f;
    for (int64_t d = 0; d < dim; ++d) {
        float diff = shared_query[d] - candidate[d];
        sum += diff * diff;
    }
    out_scores[i] = sqrtf(sum);
}

__global__ void cosine_tiled_kernel(const float* __restrict__ store,
                                     const float* __restrict__ query,
                                     int64_t num_vectors, int64_t dim,
                                     float* __restrict__ out_scores) {
    extern __shared__ float shared_query[];

    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        shared_query[d] = query[d];
    }
    __syncthreads();

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vectors) return;

    const float* candidate = store + i * dim;
    float dot = 0.0f, norm_q = 0.0f, norm_c = 0.0f;
    for (int64_t d = 0; d < dim; ++d) {
        float q = shared_query[d];
        float c = candidate[d];
        dot += q * c;
        norm_q += q * q;
        norm_c += c * c;
    }
    float denom = sqrtf(norm_q) * sqrtf(norm_c);
    out_scores[i] = denom > 0.0f ? dot / denom : 0.0f;
}

// Launchers -- these are called from binding.cpp with raw pointers already
// extracted from validated torch::Tensor objects (see binding.cpp for the
// dtype/device/contiguity checks; keeping those checks in the .cpp file
// and keeping this file focused on just the kernel launch is a deliberate
// separation between "PyTorch-facing validation" and "CUDA execution").
void launch_l2_tiled(const float* store, const float* query,
                      int64_t num_vectors, int64_t dim, float* out_scores) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((num_vectors + threads_per_block - 1) / threads_per_block);
    const size_t shared_mem_bytes = dim * sizeof(float);
    l2_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        store, query, num_vectors, dim, out_scores);
}

void launch_cosine_tiled(const float* store, const float* query,
                          int64_t num_vectors, int64_t dim, float* out_scores) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((num_vectors + threads_per_block - 1) / threads_per_block);
    const size_t shared_mem_bytes = dim * sizeof(float);
    cosine_tiled_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        store, query, num_vectors, dim, out_scores);
}

} // namespace vector_search_torch
