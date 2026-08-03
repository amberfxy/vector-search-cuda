#include "topk_cpu.hpp"
#include <algorithm>

std::vector<ScoredIndex> top_k_smallest(const float* scores, size_t n, size_t k) {
    k = std::min(k, n);
    std::vector<ScoredIndex> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = {i, scores[i]};

    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                       [](const ScoredIndex& a, const ScoredIndex& b) {
                           return a.score < b.score;
                       });
    all.resize(k);
    return all;
}

std::vector<ScoredIndex> top_k_largest(const float* scores, size_t n, size_t k) {
    k = std::min(k, n);
    std::vector<ScoredIndex> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = {i, scores[i]};

    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                       [](const ScoredIndex& a, const ScoredIndex& b) {
                           return a.score > b.score;
                       });
    all.resize(k);
    return all;
}
