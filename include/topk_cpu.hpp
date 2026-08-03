#pragma once
#include <vector>
#include <cstddef>

struct ScoredIndex {
    size_t index;
    float score;
};

// Returns the k indices with the smallest score (use this for L2 distance,
// where "closest" means smallest) or largest score (flip the comparator
// for cosine similarity, where "most similar" means largest) out of
// `scores[0..n)`. Implemented with std::partial_sort, which is the
// standard CPU baseline for top-k and also serves as the correctness
// ground truth that the GPU top-k implementation is checked against.
std::vector<ScoredIndex> top_k_smallest(const float* scores, size_t n, size_t k);
std::vector<ScoredIndex> top_k_largest(const float* scores, size_t n, size_t k);
