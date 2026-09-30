// Device-resident corpus index. Upload store once; reuse query/score buffers.
#include "gpu_vector_index.cuh"
#include <cuda_runtime.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
    } \
} while (0)

float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

} // namespace

GpuVectorIndex::GpuVectorIndex(const float* h_store, size_t num_vectors, size_t dim)
    : num_vectors_(num_vectors), dim_(dim) {
    if (h_store == nullptr || num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("GpuVectorIndex: empty store/dim");
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaMalloc(&d_store_, num_vectors_ * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query_, dim_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores_, num_vectors_ * sizeof(float)));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_store_, h_store, num_vectors_ * dim_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const double h2d_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    build_time_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    // Keep h2d visible inside build_time; unused variable silence if optimized:
    (void)h2d_ms;
}

GpuVectorIndex::~GpuVectorIndex() { release(); }

GpuVectorIndex::GpuVectorIndex(GpuVectorIndex&& other) noexcept
    : num_vectors_(other.num_vectors_),
      dim_(other.dim_),
      d_store_(other.d_store_),
      d_query_(other.d_query_),
      d_scores_(other.d_scores_),
      build_time_ms_(other.build_time_ms_) {
    other.num_vectors_ = 0;
    other.dim_ = 0;
    other.d_store_ = nullptr;
    other.d_query_ = nullptr;
    other.d_scores_ = nullptr;
    other.build_time_ms_ = 0.0;
}

GpuVectorIndex& GpuVectorIndex::operator=(GpuVectorIndex&& other) noexcept {
    if (this != &other) {
        release();
        num_vectors_ = other.num_vectors_;
        dim_ = other.dim_;
        d_store_ = other.d_store_;
        d_query_ = other.d_query_;
        d_scores_ = other.d_scores_;
        build_time_ms_ = other.build_time_ms_;
        other.num_vectors_ = 0;
        other.dim_ = 0;
        other.d_store_ = nullptr;
        other.d_query_ = nullptr;
        other.d_scores_ = nullptr;
        other.build_time_ms_ = 0.0;
    }
    return *this;
}

void GpuVectorIndex::release() noexcept {
    if (d_store_) { cudaFree(d_store_); d_store_ = nullptr; }
    if (d_query_) { cudaFree(d_query_); d_query_ = nullptr; }
    if (d_scores_) { cudaFree(d_scores_); d_scores_ = nullptr; }
}

void GpuVectorIndex::search(const float* h_query, Metric metric, float* h_out_scores,
                             GpuKernelKind kind, CudaStageTimes* stages) {
    if (!d_store_ || !h_query || !h_out_scores) {
        throw std::runtime_error("GpuVectorIndex::search: invalid state or null pointer");
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t_e2e0 = Clock::now();

    CudaStageTimes local{};
    // Resident path: no per-query corpus alloc/H2D/free.
    local.alloc_ms = 0.0;
    local.corpus_h2d_ms = 0.0;
    local.free_ms = 0.0;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_query_, h_query, dim_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.query_h2d_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    if (kind == GpuKernelKind::Naive) {
        if (metric == Metric::L2) {
            launch_l2_naive_device(d_store_, d_query_, num_vectors_, dim_, d_scores_);
        } else {
            launch_cosine_naive_device(d_store_, d_query_, num_vectors_, dim_, d_scores_);
        }
    } else {
        if (metric == Metric::L2) {
            launch_l2_tiled_device(d_store_, d_query_, num_vectors_, dim_, d_scores_);
        } else {
            launch_cosine_tiled_device(d_store_, d_query_, num_vectors_, dim_, d_scores_);
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.kernel_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(h_out_scores, d_scores_, num_vectors_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    local.scores_d2h_ms = event_ms(ev0, ev1);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    local.e2e_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_e2e0).count();
    if (stages) *stages = local;
}
