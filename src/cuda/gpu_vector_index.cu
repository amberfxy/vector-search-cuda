// Device-resident corpus index. Upload store once; reuse query/score buffers.
#include "gpu_vector_index.cuh"
#include "cuda_error.hpp"
#include "nvtx_ranges.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    VSC_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

void validate_finite_query(const float* q, size_t dim, const char* what) {
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(q[i])) {
            throw std::invalid_argument(
                std::string(what) + ": non-finite value at index " + std::to_string(i));
        }
    }
}

} // namespace

GpuVectorIndex::GpuVectorIndex(const float* h_store, size_t num_vectors, size_t dim, int device)
    : num_vectors_(num_vectors), dim_(dim) {
    if (h_store == nullptr) {
        throw std::invalid_argument("GpuVectorIndex::build: h_store is null");
    }
    if (num_vectors == 0 || dim == 0) {
        throw std::invalid_argument("GpuVectorIndex::build: num_vectors and dim must be > 0");
    }

    int count = 0;
    VSC_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 1) {
        throw std::runtime_error("GpuVectorIndex::build: no CUDA devices available");
    }
    if (device < 0) {
        VSC_CUDA_CHECK(cudaGetDevice(&device_));
    } else {
        if (device >= count) {
            throw std::invalid_argument(
                "GpuVectorIndex::build: device " + std::to_string(device) +
                " out of range (count=" + std::to_string(count) + ")");
        }
        device_ = device;
        VSC_CUDA_CHECK(cudaSetDevice(device_));
    }

    cudaDeviceProp prop{};
    VSC_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    device_name_ = prop.name;

    const size_t corpus_bytes = num_vectors_ * dim_ * sizeof(float);
    size_t free_b = 0, total_b = 0;
    VSC_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    // Need corpus + at least 1 query + 1 score row (+ small slack).
    const size_t need = corpus_bytes + dim_ * sizeof(float) + num_vectors_ * sizeof(float) +
                        (16ull << 20);
    if (free_b < need) {
        throw std::runtime_error(
            "GpuVectorIndex::build: insufficient GPU memory (need ~" +
            std::to_string(need / (1024 * 1024)) + " MiB free, have " +
            std::to_string(free_b / (1024 * 1024)) + " MiB)");
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    cudaEvent_t ev0, ev1;
    VSC_CUDA_CHECK(cudaEventCreate(&ev0));
    VSC_CUDA_CHECK(cudaEventCreate(&ev1));

    VSC_CUDA_CHECK(cudaMalloc(&d_store_, corpus_bytes));
    query_capacity_ = 1;
    VSC_CUDA_CHECK(cudaMalloc(&d_queries_, query_capacity_ * dim_ * sizeof(float)));
    VSC_CUDA_CHECK(cudaMalloc(&d_scores_, query_capacity_ * num_vectors_ * sizeof(float)));

    VSC_CUDA_CHECK(cudaEventRecord(ev0));
    VSC_CUDA_CHECK(cudaMemcpy(d_store_, h_store, corpus_bytes, cudaMemcpyHostToDevice));
    VSC_CUDA_CHECK(cudaEventRecord(ev1));
    VSC_CUDA_CHECK(cudaEventSynchronize(ev1));
    (void)event_ms(ev0, ev1);

    VSC_CUDA_CHECK(cudaEventDestroy(ev0));
    VSC_CUDA_CHECK(cudaEventDestroy(ev1));

    build_time_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

GpuVectorIndex::~GpuVectorIndex() { release(); }

GpuVectorIndex::GpuVectorIndex(GpuVectorIndex&& other) noexcept
    : num_vectors_(other.num_vectors_),
      dim_(other.dim_),
      device_(other.device_),
      query_capacity_(other.query_capacity_),
      d_store_(other.d_store_),
      d_queries_(other.d_queries_),
      d_scores_(other.d_scores_),
      build_time_ms_(other.build_time_ms_),
      device_name_(std::move(other.device_name_)) {
    other.num_vectors_ = 0;
    other.dim_ = 0;
    other.device_ = -1;
    other.query_capacity_ = 0;
    other.d_store_ = nullptr;
    other.d_queries_ = nullptr;
    other.d_scores_ = nullptr;
    other.build_time_ms_ = 0.0;
}

GpuVectorIndex& GpuVectorIndex::operator=(GpuVectorIndex&& other) noexcept {
    if (this != &other) {
        release();
        num_vectors_ = other.num_vectors_;
        dim_ = other.dim_;
        device_ = other.device_;
        query_capacity_ = other.query_capacity_;
        d_store_ = other.d_store_;
        d_queries_ = other.d_queries_;
        d_scores_ = other.d_scores_;
        build_time_ms_ = other.build_time_ms_;
        device_name_ = std::move(other.device_name_);
        other.num_vectors_ = 0;
        other.dim_ = 0;
        other.device_ = -1;
        other.query_capacity_ = 0;
        other.d_store_ = nullptr;
        other.d_queries_ = nullptr;
        other.d_scores_ = nullptr;
        other.build_time_ms_ = 0.0;
    }
    return *this;
}

void GpuVectorIndex::release() noexcept {
    if (d_store_) { cudaFree(d_store_); d_store_ = nullptr; }
    if (d_queries_) { cudaFree(d_queries_); d_queries_ = nullptr; }
    if (d_scores_) { cudaFree(d_scores_); d_scores_ = nullptr; }
    query_capacity_ = 0;
}

void GpuVectorIndex::reset() noexcept {
    release();
    num_vectors_ = 0;
    dim_ = 0;
    device_ = -1;
    build_time_ms_ = 0.0;
    device_name_.clear();
}

GpuIndexStats GpuVectorIndex::stats() const {
    GpuIndexStats s;
    s.num_vectors = num_vectors_;
    s.dim = dim_;
    s.device = device_;
    s.device_corpus_bytes = deviceCorpusBytes();
    s.device_query_capacity_bytes = query_capacity_ * dim_ * sizeof(float);
    s.device_score_capacity_bytes = query_capacity_ * num_vectors_ * sizeof(float);
    s.query_capacity = query_capacity_;
    s.build_time_ms = build_time_ms_;
    s.device_name = device_name_;
    return s;
}

void GpuVectorIndex::ensure_query_capacity(size_t num_queries) {
    if (num_queries == 0) {
        throw std::invalid_argument("GpuVectorIndex: num_queries must be > 0");
    }
    if (num_queries <= query_capacity_) return;

    float* new_q = nullptr;
    float* new_s = nullptr;
    VSC_CUDA_CHECK(cudaMalloc(&new_q, num_queries * dim_ * sizeof(float)));
    VSC_CUDA_CHECK(cudaMalloc(&new_s, num_queries * num_vectors_ * sizeof(float)));
    if (d_queries_) cudaFree(d_queries_);
    if (d_scores_) cudaFree(d_scores_);
    d_queries_ = new_q;
    d_scores_ = new_s;
    query_capacity_ = num_queries;
}

void GpuVectorIndex::launch_one(const float* d_query, float* d_scores, Metric metric,
                                GpuKernelKind kind) {
    if (kind == GpuKernelKind::Warp) {
        if (metric != Metric::L2) {
            throw std::invalid_argument(
                "GpuVectorIndex: GpuKernelKind::Warp currently supports L2 only");
        }
        launch_l2_warp_device(d_store_, d_query, num_vectors_, dim_, d_scores);
    } else if (kind == GpuKernelKind::Naive) {
        if (metric == Metric::L2) {
            launch_l2_naive_device(d_store_, d_query, num_vectors_, dim_, d_scores);
        } else {
            launch_cosine_naive_device(d_store_, d_query, num_vectors_, dim_, d_scores);
        }
    } else {
        if (metric == Metric::L2) {
            launch_l2_tiled_device(d_store_, d_query, num_vectors_, dim_, d_scores);
        } else {
            launch_cosine_tiled_device(d_store_, d_query, num_vectors_, dim_, d_scores);
        }
    }
}

void GpuVectorIndex::search(const float* h_query, Metric metric, float* h_out_scores,
                             GpuKernelKind kind, CudaStageTimes* stages) {
    search_batch(h_query, /*num_queries=*/1, metric, h_out_scores, kind, stages);
}

void GpuVectorIndex::search_batch(const float* h_queries, size_t num_queries, Metric metric,
                                  float* h_out_scores, GpuKernelKind kind,
                                  CudaStageTimes* stages) {
    if (!d_store_) {
        throw std::runtime_error("GpuVectorIndex::search_batch: index is empty (call build())");
    }
    if (!h_queries || !h_out_scores) {
        throw std::invalid_argument("GpuVectorIndex::search_batch: null pointer");
    }
    if (num_queries == 0) {
        throw std::invalid_argument("GpuVectorIndex::search_batch: num_queries must be > 0");
    }

    VSC_CUDA_CHECK(cudaSetDevice(device_));
    for (size_t q = 0; q < num_queries; ++q) {
        validate_finite_query(h_queries + q * dim_, dim_, "GpuVectorIndex::search_batch query");
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t_e2e0 = Clock::now();

    CudaStageTimes local{};
    local.alloc_ms = 0.0;
    local.corpus_h2d_ms = 0.0;
    local.free_ms = 0.0;

    ensure_query_capacity(num_queries);

    cudaEvent_t ev0, ev1;
    VSC_CUDA_CHECK(cudaEventCreate(&ev0));
    VSC_CUDA_CHECK(cudaEventCreate(&ev1));

    {
        NvtxRange r("query_h2d");
        VSC_CUDA_CHECK(cudaEventRecord(ev0));
        VSC_CUDA_CHECK(cudaMemcpy(d_queries_, h_queries,
                                 num_queries * dim_ * sizeof(float),
                                 cudaMemcpyHostToDevice));
        VSC_CUDA_CHECK(cudaEventRecord(ev1));
        VSC_CUDA_CHECK(cudaEventSynchronize(ev1));
        local.query_h2d_ms = event_ms(ev0, ev1);
    }

    {
        NvtxRange r("distance_kernel");
        VSC_CUDA_CHECK(cudaEventRecord(ev0));
        for (size_t q = 0; q < num_queries; ++q) {
            const float* dq = d_queries_ + q * dim_;
            float* ds = d_scores_ + q * num_vectors_;
            launch_one(dq, ds, metric, kind);
        }
        VSC_CUDA_CHECK(cudaEventRecord(ev1));
        VSC_CUDA_CHECK(cudaEventSynchronize(ev1));
        local.kernel_ms = event_ms(ev0, ev1);
    }

    {
        NvtxRange r("scores_d2h");
        VSC_CUDA_CHECK(cudaEventRecord(ev0));
        VSC_CUDA_CHECK(cudaMemcpy(h_out_scores, d_scores_,
                                 num_queries * num_vectors_ * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        VSC_CUDA_CHECK(cudaEventRecord(ev1));
        VSC_CUDA_CHECK(cudaEventSynchronize(ev1));
        local.scores_d2h_ms = event_ms(ev0, ev1);
    }

    VSC_CUDA_CHECK(cudaEventDestroy(ev0));
    VSC_CUDA_CHECK(cudaEventDestroy(ev1));

    local.e2e_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_e2e0).count();
    if (stages) *stages = local;
}
