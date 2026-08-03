#pragma once
#include "vector_store.hpp"

// Distance metric selector.
enum class Metric { L2, Cosine };

// Single-vector-pair distance (used for correctness checks and as the
// building block the batch functions call in a loop).
float l2_distance(const float* a, const float* b, size_t dim);
float cosine_similarity(const float* a, const float* b, size_t dim);

// Batch: compute distance/similarity from a single query vector against
// every vector in `store`. out_scores must have space for store.numVectors().
//
// Three variants, in increasing order of sophistication — these are the
// three "rungs" the CUDA implementation will be benchmarked against:
//   1. Single-threaded, naive loop.               (batch_distance_singlethread)
//   2. Multi-threaded via OpenMP.                  (batch_distance_openmp)
// The GPU kernels (naive + shared-memory tiled) live in the cuda/ directory
// and share this same query-vs-all-candidates contract.
void batch_distance_singlethread(const VectorStore& store, const float* query,
                                  Metric metric, float* out_scores);

void batch_distance_openmp(const VectorStore& store, const float* query,
                            Metric metric, float* out_scores);
