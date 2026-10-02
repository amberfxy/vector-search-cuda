// Ensures the profiling executable's intended API surface stays wired to the
// same resident entry points as the harness. This is a compile/link-oriented
// guard on CUDA builds; full numerical coverage remains in test_correctness_gpu.
#include "gpu_vector_index.cuh"
#include "vector_store.hpp"
#include <cstdio>
#include <vector>

int main() {
    // Tiny smoke: construct index + one search through GpuVectorIndex::search
    // (identical call shape to profile_kernel / bench_harness resident mode).
    const size_t n = 257;  // non-multiple of block size
    const size_t dim = 64;
    VectorStore store(n, dim);
    store.fillRandom(42, true);
    VectorStore q(1, dim);
    q.fillRandom(43, true);

    GpuVectorIndex index(store.raw(), n, dim);
    std::vector<float> scores(n);
    index.search(q.vectorAt(0), Metric::L2, scores.data(), GpuKernelKind::Tiled, nullptr);
    index.search(q.vectorAt(0), Metric::L2, scores.data(), GpuKernelKind::Naive, nullptr);
    index.search(q.vectorAt(0), Metric::L2, scores.data(), GpuKernelKind::Warp, nullptr);

    // Batched API smoke
    VectorStore qs(4, dim);
    qs.fillRandom(44, true);
    std::vector<float> batch_scores(4 * n);
    index.search_batch(qs.raw(), 4, Metric::L2, batch_scores.data(), GpuKernelKind::Tiled,
                       nullptr);
    auto st = index.stats();
    (void)st;
    index.reset();

    std::printf("passed: profile path API (search/search_batch/reset; n=%zu dim=%zu)\n",
                n, dim);
    return 0;
}
