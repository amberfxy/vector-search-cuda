#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "distance_cpu.hpp"

// Device-resident corpus index: upload N×D store once, run many searches
// without re-allocating or re-uploading the corpus.
//
// Contrast with batch_distance_*_cuda (legacy), which cudaMalloc + H2D store
// + free on every call. Keep both paths so A/B benchmarks stay honest.

struct CudaStageTimes {
    double alloc_ms = 0.0;
    double corpus_h2d_ms = 0.0;
    double query_h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double scores_d2h_ms = 0.0;
    double free_ms = 0.0;
    double e2e_ms = 0.0;
};

struct GpuIndexStats {
    size_t num_vectors = 0;
    size_t dim = 0;
    int device = -1;
    size_t device_corpus_bytes = 0;
    size_t device_query_capacity_bytes = 0;
    size_t device_score_capacity_bytes = 0;
    size_t query_capacity = 0;  // max queries that fit without realloc
    double build_time_ms = 0.0;
    std::string device_name;
};

enum class GpuKernelKind { Naive, Tiled, Warp };

// Legacy path with per-stage cudaEvent timing (same lifecycle as
// batch_distance_*_cuda: alloc + full corpus H2D every call).
void batch_distance_naive_cuda_staged(const float* h_store, size_t num_vectors, size_t dim,
                                       const float* h_query, Metric metric,
                                       float* h_out_scores, CudaStageTimes* stages);

void batch_distance_tiled_cuda_staged(const float* h_store, size_t num_vectors, size_t dim,
                                       const float* h_query, Metric metric,
                                       float* h_out_scores, CudaStageTimes* stages);

// Device-pointer kernel launches (store/query/scores already on device).
void launch_l2_naive_device(const float* d_store, const float* d_query,
                             size_t num_vectors, size_t dim, float* d_scores);
void launch_cosine_naive_device(const float* d_store, const float* d_query,
                                 size_t num_vectors, size_t dim, float* d_scores);
void launch_l2_tiled_device(const float* d_store, const float* d_query,
                             size_t num_vectors, size_t dim, float* d_scores);
void launch_cosine_tiled_device(const float* d_store, const float* d_query,
                                 size_t num_vectors, size_t dim, float* d_scores);

// Phase 5A experiment: warp-per-vector L2 only (see docs/PHASE5_WARP_PER_VECTOR.md).
void launch_l2_warp_device(const float* d_store, const float* d_query,
                            size_t num_vectors, size_t dim, float* d_scores);

// Production-oriented device-resident index (RAII).
//
// Lifecycle:
//   GpuVectorIndex idx = GpuVectorIndex::build(h_store, N, dim, device);
//   idx.search(...) / idx.search_batch(...);
//   idx.reset();  // frees device memory
//
// add() is not supported (immutable corpus after build). Rebuild via build().
class GpuVectorIndex {
public:
    GpuVectorIndex() = default;

    // Uploads corpus once onto `device` (-1 = current device). Throws on error.
    GpuVectorIndex(const float* h_store, size_t num_vectors, size_t dim, int device = -1);

    static GpuVectorIndex build(const float* h_store, size_t num_vectors, size_t dim,
                               int device = -1) {
        return GpuVectorIndex(h_store, num_vectors, dim, device);
    }

    ~GpuVectorIndex();

    GpuVectorIndex(const GpuVectorIndex&) = delete;
    GpuVectorIndex& operator=(const GpuVectorIndex&) = delete;
    GpuVectorIndex(GpuVectorIndex&& other) noexcept;
    GpuVectorIndex& operator=(GpuVectorIndex&& other) noexcept;

    // Single-query search. Reuses device buffers. Does not free corpus.
    // h_out_scores: length num_vectors_.
    void search(const float* h_query, Metric metric, float* h_out_scores,
                GpuKernelKind kind = GpuKernelKind::Tiled,
                CudaStageTimes* stages = nullptr);

    // Batched search: h_queries is [num_queries, dim] row-major;
    // h_out_scores is [num_queries, num_vectors] row-major.
    // Corpus stays resident; queries uploaded once; one kernel launch per query
    // (scores written into contiguous device buffer; one D2H).
    void search_batch(const float* h_queries, size_t num_queries, Metric metric,
                      float* h_out_scores, GpuKernelKind kind = GpuKernelKind::Tiled,
                      CudaStageTimes* stages = nullptr);

    // Release all device memory (safe to call multiple times).
    void reset() noexcept;

    bool empty() const { return d_store_ == nullptr; }
    size_t numVectors() const { return num_vectors_; }
    size_t dim() const { return dim_; }
    int device() const { return device_; }
    size_t deviceCorpusBytes() const { return num_vectors_ * dim_ * sizeof(float); }
    size_t deviceBufferBytes() const {
        return query_capacity_ * dim_ * sizeof(float) +
               query_capacity_ * num_vectors_ * sizeof(float);
    }
    double build_time_ms() const { return build_time_ms_; }
    GpuIndexStats stats() const;

private:
    void release() noexcept;
    void ensure_query_capacity(size_t num_queries);
    void launch_one(const float* d_query, float* d_scores, Metric metric,
                    GpuKernelKind kind);

    size_t num_vectors_ = 0;
    size_t dim_ = 0;
    int device_ = -1;
    size_t query_capacity_ = 0;
    float* d_store_ = nullptr;
    float* d_queries_ = nullptr;  // [query_capacity_, dim]
    float* d_scores_ = nullptr;   // [query_capacity_, num_vectors]
    double build_time_ms_ = 0.0;
    std::string device_name_;
};
