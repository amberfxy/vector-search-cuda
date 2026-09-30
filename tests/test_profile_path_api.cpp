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

    std::printf("passed: profile path API uses GpuVectorIndex::search (n=%zu dim=%zu)\n", n, dim);
    return 0;
}
