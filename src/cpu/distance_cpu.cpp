#include "distance_cpu.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

float l2_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float cosine_similarity(const float* a, const float* b, size_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        dot += a[d] * b[d];
        norm_a += a[d] * a[d];
        norm_b += b[d] * b[d];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    return denom > 0.0f ? dot / denom : 0.0f;
}

void batch_distance_singlethread(const VectorStore& store, const float* query,
                                  Metric metric, float* out_scores) {
    const size_t n = store.numVectors();
    const size_t dim = store.dim();
    for (size_t i = 0; i < n; ++i) {
        const float* candidate = store.vectorAt(i);
        out_scores[i] = (metric == Metric::L2)
            ? l2_distance(query, candidate, dim)
            : cosine_similarity(query, candidate, dim);
    }
}

void batch_distance_openmp(const VectorStore& store, const float* query,
                            Metric metric, float* out_scores) {
    const size_t n = store.numVectors();
    const size_t dim = store.dim();
    // Each thread handles a disjoint slice of candidate vectors -- no
    // synchronization needed since every iteration writes to a distinct
    // out_scores[i] and only reads query (read-only, shared safely).
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const float* candidate = store.vectorAt(static_cast<size_t>(i));
        out_scores[i] = (metric == Metric::L2)
            ? l2_distance(query, candidate, dim)
            : cosine_similarity(query, candidate, dim);
    }
}
