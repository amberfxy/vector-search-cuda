// Naive CUDA implementation: one thread per candidate vector. Each thread
// independently reads the ENTIRE query vector from global memory, plus its
// own candidate vector from global memory. This is intentionally the
// "obviously correct, obviously not optimized" baseline.
#include "distance_cuda.cuh"
#include "gpu_vector_index.cuh"
#include "cuda_timing.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <chrono>

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
        float diff = query[d] - candidate[d];
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

float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

} // namespace

void launch_l2_naive_device(const float* d_store, const float* d_query,
                             size_t num_vectors, size_t dim, float* d_scores) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((num_vectors + threads_per_block - 1) / threads_per_block);
    l2_naive_kernel<<<blocks, threads_per_block>>>(d_store, d_query, num_vectors, dim, d_scores);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cosine_naive_device(const float* d_store, const float* d_query,
                                 size_t num_vectors, size_t dim, float* d_scores) {
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((num_vectors + threads_per_block - 1) / threads_per_block);
    cosine_naive_kernel<<<blocks, threads_per_block>>>(d_store, d_query, num_vectors, dim, d_scores);
    CUDA_CHECK(cudaGetLastError());
}

void batch_distance_naive_cuda(const float* h_store, size_t num_vectors, size_t dim,
                                const float* h_query, Metric metric,
                                float* h_out_scores) {
    CudaStageTimes stages;
    batch_distance_naive_cuda_staged(h_store, num_vectors, dim, h_query, metric,
                                      h_out_scores, &stages);
    set_last_kernel_time_ms(static_cast<float>(stages.kernel_ms));
}

void batch_distance_naive_cuda_staged(const float* h_store, size_t num_vectors, size_t dim,
                                       const float* h_query, Metric metric,
                                       float* h_out_scores, CudaStageTimes* stages) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_e2e0 = Clock::now();

    CudaStageTimes local{};
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    float *d_store = nullptr, *d_query = nullptr, *d_scores = nullptr;

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMalloc(&d_store, num_vectors * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, num_vectors * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.alloc_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_store, h_store, num_vectors * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.corpus_h2d_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.query_h2d_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    if (metric == Metric::L2) {
        launch_l2_naive_device(d_store, d_query, num_vectors, dim, d_scores);
    } else {
        launch_cosine_naive_device(d_store, d_query, num_vectors, dim, d_scores);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.kernel_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(h_out_scores, d_scores, num_vectors * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.scores_d2h_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    cudaFree(d_store);
    cudaFree(d_query);
    cudaFree(d_scores);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.free_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    local.e2e_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_e2e0).count();
    set_last_kernel_time_ms(static_cast<float>(local.kernel_ms));
    if (stages) *stages = local;
}
