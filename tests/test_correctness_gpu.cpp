// GPU correctness + stress tests.
//
// Ground truth: CPU single-threaded batch_distance_singlethread.
// Tolerance: kTolerance = 1e-4 absolute difference per score element.
// Why not bit-identical: GPU vs CPU may differ in FMA / reduction order;
// 1e-4 is appropriate for fp32 L2/cosine on normalized unit vectors at
// dim<=1024 in this codebase (same bar as historical Colab T4 runs).
// Do NOT loosen this solely to make a broken kernel pass.
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "distance_cuda.cuh"
#include "gpu_vector_index.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-4f;
int g_failures = 0;

bool approx_eq(float a, float b, float eps = kTolerance) {
    return std::fabs(a - b) < eps;
}

struct CompareSummary {
    float max_diff = 0.0f;
    size_t num_mismatches = 0;
    size_t worst_index = 0;
};

CompareSummary compare_scores(const std::vector<float>& a, const std::vector<float>& b) {
    CompareSummary s;
    if (a.size() != b.size()) {
        s.num_mismatches = a.size() + b.size();
        return s;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > s.max_diff) { s.max_diff = diff; s.worst_index = i; }
        if (diff >= kTolerance) s.num_mismatches++;
    }
    return s;
}

void report(const std::string& label, const std::vector<float>& ref,
            const std::vector<float>& got) {
    if (ref.size() != got.size()) {
        std::fprintf(stderr, "FAILED: %s -- size mismatch (ref=%zu, got=%zu)\n",
                      label.c_str(), ref.size(), got.size());
        g_failures++;
        return;
    }
    auto s = compare_scores(ref, got);
    if (s.num_mismatches == 0) {
        std::printf("passed: %s (max abs diff = %.6f at %zu, tol = %.6f)\n",
                     label.c_str(), s.max_diff, s.worst_index, kTolerance);
    } else {
        std::fprintf(stderr,
            "FAILED: %s -- %zu / %zu exceeded tol (max abs diff = %.6f at %zu, "
            "ref=%.6f got=%.6f)\n",
            label.c_str(), s.num_mismatches, ref.size(), s.max_diff, s.worst_index,
            ref[s.worst_index], got[s.worst_index]);
        g_failures++;
    }
}

void run_size_dim(size_t n, size_t dim, Metric metric, const char* metric_name) {
    VectorStore store(n, dim);
    store.fillRandom(/*seed=*/321, /*normalize=*/true);
    VectorStore query_store(1, dim);
    query_store.fillRandom(/*seed=*/654, /*normalize=*/true);
    const float* query = query_store.vectorAt(0);

    std::vector<float> cpu(n), naive(n), tiled(n), resident(n);
    batch_distance_singlethread(store, query, metric, cpu.data());
    batch_distance_naive_cuda(store.raw(), n, dim, query, metric, naive.data());
    batch_distance_tiled_cuda(store.raw(), n, dim, query, metric, tiled.data());

    report(std::string("naive vs CPU [") + metric_name + " n=" + std::to_string(n) +
           " d=" + std::to_string(dim) + "]", cpu, naive);
    report(std::string("tiled vs CPU [") + metric_name + " n=" + std::to_string(n) +
           " d=" + std::to_string(dim) + "]", cpu, tiled);
    report(std::string("naive vs tiled [") + metric_name + " n=" + std::to_string(n) +
           " d=" + std::to_string(dim) + "]", naive, tiled);

    GpuVectorIndex index(store.raw(), n, dim);
    index.search(query, metric, resident.data(), GpuKernelKind::Tiled, nullptr);
    report(std::string("resident-tiled vs CPU [") + metric_name + " n=" +
           std::to_string(n) + " d=" + std::to_string(dim) + "]", cpu, resident);

    std::vector<float> resident_naive(n);
    index.search(query, metric, resident_naive.data(), GpuKernelKind::Naive, nullptr);
    report(std::string("resident-naive vs CPU [") + metric_name + " n=" +
           std::to_string(n) + " d=" + std::to_string(dim) + "]", cpu, resident_naive);

    report(std::string("legacy-tiled vs resident-tiled [") + metric_name + " n=" +
           std::to_string(n) + " d=" + std::to_string(dim) + "]", tiled, resident);
}

void test_zero_vector_cosine() {
    // Zero query against non-zero candidates → cosine should be 0.
    const size_t n = 128;
    const size_t dim = 384;
    VectorStore store(n, dim);
    store.fillRandom(/*seed=*/7, /*normalize=*/true);
    std::vector<float> query(dim, 0.0f);

    std::vector<float> cpu(n), tiled(n), resident(n);
    batch_distance_singlethread(store, query.data(), Metric::Cosine, cpu.data());
    batch_distance_tiled_cuda(store.raw(), n, dim, query.data(), Metric::Cosine, tiled.data());
    GpuVectorIndex index(store.raw(), n, dim);
    index.search(query.data(), Metric::Cosine, resident.data(), GpuKernelKind::Tiled, nullptr);

    for (size_t i = 0; i < n; ++i) {
        if (!approx_eq(cpu[i], 0.0f) || !approx_eq(tiled[i], 0.0f) ||
            !approx_eq(resident[i], 0.0f)) {
            std::fprintf(stderr,
                "FAILED: zero-query cosine — expected ~0 at %zu (cpu=%.6f tiled=%.6f resident=%.6f)\n",
                i, cpu[i], tiled[i], resident[i]);
            g_failures++;
            return;
        }
    }
    std::printf("passed: zero-query cosine edge case (n=%zu dim=%zu)\n", n, dim);
}

void test_repeated_resident_stable() {
    const size_t n = 1000;
    const size_t dim = 384;
    VectorStore store(n, dim);
    store.fillRandom(99, true);
    VectorStore qstore(3, dim);
    qstore.fillRandom(100, true);

    GpuVectorIndex index(store.raw(), n, dim);
    std::vector<float> first(n), again(n);
    index.search(qstore.vectorAt(0), Metric::L2, first.data(), GpuKernelKind::Tiled, nullptr);

    for (int rep = 0; rep < 20; ++rep) {
        index.search(qstore.vectorAt(static_cast<size_t>(rep % 3)), Metric::L2,
                     again.data(), GpuKernelKind::Tiled, nullptr);
    }
    // Re-run first query — must match original result (no state corruption).
    index.search(qstore.vectorAt(0), Metric::L2, again.data(), GpuKernelKind::Tiled, nullptr);
    report("resident repeated-run stability (L2 n=1000)", first, again);
}

void test_memory_stability_resident() {
    const size_t n = 5000;
    const size_t dim = 384;
    VectorStore store(n, dim);
    store.fillRandom(11, true);
    VectorStore qstore(1, dim);
    qstore.fillRandom(12, true);

    size_t free0 = 0, total0 = 0, free1 = 0, total1 = 0;
    cudaMemGetInfo(&free0, &total0);

    {
        GpuVectorIndex index(store.raw(), n, dim);
        std::vector<float> scores(n);
        for (int i = 0; i < 50; ++i) {
            index.search(qstore.vectorAt(0), Metric::L2, scores.data(),
                         GpuKernelKind::Tiled, nullptr);
        }
        cudaMemGetInfo(&free1, &total1);
        const long long used_during =
            static_cast<long long>(free0) - static_cast<long long>(free1);
        std::printf("info: memory during resident stress — free_before=%zu free_during=%zu "
                    "delta_used≈%lld bytes (index holds corpus)\n",
                    free0, free1, used_during);
    }

    size_t free2 = 0, total2 = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free2, &total2);
    // After destructor, free memory should return near baseline (allow 16MB slack
    // for driver/allocator caching — flag only large unreclaimed growth).
    const long long leaked =
        static_cast<long long>(free0) - static_cast<long long>(free2);
    constexpr long long kSlack = 16ll * 1024 * 1024;
    if (leaked > kSlack) {
        std::fprintf(stderr,
            "FAILED: possible GPU memory leak after GpuVectorIndex destroy "
            "(free_before=%zu free_after=%zu unreclaimed≈%lld bytes)\n",
            free0, free2, leaked);
        g_failures++;
    } else {
        std::printf("passed: resident memory stability (unreclaimed≈%lld bytes, slack=%lld)\n",
                     leaked, kSlack);
    }
}

void test_optional_large_n() {
    // 1M × 384 ≈ 1.5 GiB store — skip if device lacks headroom.
    const size_t n = 1000000;
    const size_t dim = 384;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t need = n * dim * sizeof(float) * 2 + (64ull << 20); // store+scores+slop
    if (free_b < need) {
        std::printf("skipped: 1M-vector validation (need ~%zu MiB free, have %zu MiB)\n",
                     need / (1024 * 1024), free_b / (1024 * 1024));
        return;
    }
    std::printf("running: 1M-vector L2 validation (dim=384)...\n");
    run_size_dim(n, dim, Metric::L2, "L2");
}

} // namespace

int main() {
    std::printf("GPU correctness/stress — tolerance = %.1e (absolute per element)\n",
                static_cast<double>(kTolerance));

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        std::fprintf(stderr, "No CUDA device — cannot run GPU tests.\n");
        return 1;
    }

    // Boundary + common dims. Include non-multiple of block size (257).
    const std::vector<size_t> sizes = {100, 257, 10000, 100000};
    const std::vector<size_t> dims = {384, 768, 1024};

    for (size_t dim : dims) {
        for (size_t n : sizes) {
            // Keep 100K×1024 runnable; still meaningful coverage.
            run_size_dim(n, dim, Metric::L2, "L2");
            run_size_dim(n, dim, Metric::Cosine, "Cosine");
        }
    }

    test_zero_vector_cosine();
    test_repeated_resident_stable();
    test_memory_stability_resident();
    test_optional_large_n();

    if (g_failures == 0) {
        std::printf("\nAll GPU correctness/stress checks passed.\n");
        return 0;
    }
    std::printf("\n%d GPU check(s) FAILED — do not trust benchmarks until fixed.\n",
                g_failures);
    return 1;
}
