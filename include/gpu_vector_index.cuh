#pragma once
#include <cstddef>
#include <cstdint>
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

enum class GpuKernelKind { Naive, Tiled };

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

class GpuVectorIndex {
public:
    // Uploads corpus once. Throws on CUDA error.
    GpuVectorIndex(const float* h_store, size_t num_vectors, size_t dim);

    ~GpuVectorIndex();

    GpuVectorIndex(const GpuVectorIndex&) = delete;
    GpuVectorIndex& operator=(const GpuVectorIndex&) = delete;
    GpuVectorIndex(GpuVectorIndex&& other) noexcept;
    GpuVectorIndex& operator=(GpuVectorIndex&& other) noexcept;

    // Per-query search. Reuses device query/score buffers. Does not free corpus.
    // If stages != nullptr, fills per-stage timings for this call (e2e excludes
    // index construction — that is build_time_ms()).
    void search(const float* h_query, Metric metric, float* h_out_scores,
                GpuKernelKind kind = GpuKernelKind::Tiled,
                CudaStageTimes* stages = nullptr);

    size_t numVectors() const { return num_vectors_; }
    size_t dim() const { return dim_; }
    size_t deviceCorpusBytes() const { return num_vectors_ * dim_ * sizeof(float); }
    size_t deviceBufferBytes() const {
        return dim_ * sizeof(float) + num_vectors_ * sizeof(float);
    }

    // One-time construction cost (alloc + corpus H2D), milliseconds.
    double build_time_ms() const { return build_time_ms_; }

private:
    void release() noexcept;

    size_t num_vectors_ = 0;
    size_t dim_ = 0;
    float* d_store_ = nullptr;
    float* d_query_ = nullptr;
    float* d_scores_ = nullptr;
    double build_time_ms_ = 0.0;
};
