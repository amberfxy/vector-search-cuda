// GPU correctness tests -- only meaningful (and only buildable) on a
// machine with a CUDA-capable GPU and the CUDA toolkit installed.
//
// This is the test you should run FIRST on real hardware, before trusting
// any latency number out of bench_runner. It checks both CUDA kernels
// (naive and shared-memory-tiled) against the CPU single-threaded
// implementation, which is the established ground truth from
// tests/test_correctness.cpp.
//
// What "correct" means here: every element of the GPU output score array
// must be within `kTolerance` of the corresponding CPU output. Floating
// point operations on GPU and CPU are not bit-identical (different
// instruction scheduling, possible fused-multiply-add differences), so an
// exact match is the wrong bar -- a small numeric tolerance is the right
// one, same as the CPU-vs-CPU (single-thread vs. OpenMP) checks in
// test_correctness.cpp.
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "distance_cuda.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

namespace {

constexpr float kTolerance = 1e-4f;
int g_failures = 0;

bool approx_eq(float a, float b, float eps = kTolerance) {
    return std::fabs(a - b) < eps;
}

// Compares two score arrays element-wise, reports the max absolute
// difference and how many elements exceeded tolerance (not just pass/fail),
// since "3 out of 1,000,000 elements are slightly off" and "everything is
// wrong" call for very different debugging next steps.
void compare_and_report(const std::string& label,
                         const std::vector<float>& cpu,
                         const std::vector<float>& gpu) {
    if (cpu.size() != gpu.size()) {
        std::fprintf(stderr, "FAILED: %s -- size mismatch (cpu=%zu, gpu=%zu)\n",
                      label.c_str(), cpu.size(), gpu.size());
        g_failures++;
        return;
    }

    float max_diff = 0.0f;
    size_t num_mismatches = 0;
    size_t worst_index = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > max_diff) { max_diff = diff; worst_index = i; }
        if (diff >= kTolerance) num_mismatches++;
    }

    if (num_mismatches == 0) {
        std::printf("passed: %s (max abs diff = %.6f at index %zu, tolerance = %.6f)\n",
                     label.c_str(), max_diff, worst_index, kTolerance);
    } else {
        std::fprintf(stderr,
            "FAILED: %s -- %zu / %zu elements exceeded tolerance "
            "(max abs diff = %.6f at index %zu, cpu=%.6f, gpu=%.6f)\n",
            label.c_str(), num_mismatches, cpu.size(), max_diff, worst_index,
            cpu[worst_index], gpu[worst_index]);
        g_failures++;
    }
}

void run_check_for_size(size_t n, size_t dim, Metric metric, const std::string& metric_name) {
    VectorStore store(n, dim);
    store.fillRandom(/*seed=*/321, /*normalize=*/true);

    VectorStore query_store(1, dim);
    query_store.fillRandom(/*seed=*/654, /*normalize=*/true);
    const float* query = query_store.vectorAt(0);

    std::vector<float> cpu_scores(n);
    batch_distance_singlethread(store, query, metric, cpu_scores.data());

    std::vector<float> naive_scores(n);
    batch_distance_naive_cuda(store.raw(), n, dim, query, metric, naive_scores.data());
    compare_and_report("naive CUDA vs CPU  [" + metric_name + ", n=" + std::to_string(n) + "]",
                        cpu_scores, naive_scores);

    std::vector<float> tiled_scores(n);
    batch_distance_tiled_cuda(store.raw(), n, dim, query, metric, tiled_scores.data());
    compare_and_report("tiled CUDA vs CPU  [" + metric_name + ", n=" + std::to_string(n) + "]",
                        cpu_scores, tiled_scores);

    // The two GPU kernels should also agree with EACH OTHER, independent of
    // the CPU comparison -- this catches bugs that happen to cancel out
    // against CPU rounding but are still real bugs (e.g. an off-by-one in
    // the shared memory load loop that happens to not matter for this
    // particular random seed).
    compare_and_report("naive CUDA vs tiled CUDA  [" + metric_name + ", n=" + std::to_string(n) + "]",
                        naive_scores, tiled_scores);
}

} // namespace

int main() {
    // Deliberately include a size smaller than one block (256 threads) and
    // a size that isn't a clean multiple of the block size, since those are
    // the boundary conditions most likely to expose an off-by-one in the
    // `i >= num_vectors` guard or the shared-memory cooperative load loop.
    const std::vector<size_t> sizes = {100, 257, 10000, 100000};
    const size_t dim = 384;

    for (size_t n : sizes) {
        run_check_for_size(n, dim, Metric::L2, "L2");
        run_check_for_size(n, dim, Metric::Cosine, "Cosine");
    }

    if (g_failures == 0) {
        std::printf("\nAll GPU correctness checks passed.\n");
        return 0;
    } else {
        std::printf("\n%d GPU correctness check(s) FAILED -- do not trust bench_runner numbers until this is fixed.\n",
                     g_failures);
        return 1;
    }
}
