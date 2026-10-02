// Expanded GPU correctness matrix vs CPU ground truth.
//
// Tolerance policy (absolute per score element):
//   kTolerance = 1e-4f
// Rationale: fp32 L2/cosine with different reduction orders (serial CPU vs
// warp shuffle / FMA) can differ at ~1e-5–1e-4 on normalized unit vectors at
// dim<=1024. Do NOT loosen solely to pass a broken kernel.
//
// FAISS reference is optional (not linked); CPU single-thread is the trusted
// baseline in this binary.
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "distance_cuda.cuh"
#include "gpu_vector_index.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-4f;
int g_failures = 0;

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
    auto s = compare_scores(ref, got);
    if (ref.size() != got.size()) {
        std::fprintf(stderr, "FAILED: %s — size mismatch\n", label.c_str());
        g_failures++;
        return;
    }
    if (s.num_mismatches == 0) {
        std::printf("passed: %s (max abs diff=%.6g)\n", label.c_str(), s.max_diff);
    } else {
        std::fprintf(stderr,
            "FAILED: %s — %zu/%zu over tol (max=%.6g at %zu)\n",
            label.c_str(), s.num_mismatches, ref.size(), s.max_diff, s.worst_index);
        g_failures++;
    }
}

void expect_throw(const char* label, bool threw) {
    if (threw) {
        std::printf("passed: %s (threw as expected)\n", label);
    } else {
        std::fprintf(stderr, "FAILED: %s (expected exception)\n", label);
        g_failures++;
    }
}

void run_single(size_t n, size_t dim, unsigned seed, GpuKernelKind kind, const char* kind_name) {
    VectorStore store(n, dim);
    store.fillRandom(seed, true);
    VectorStore q(1, dim);
    q.fillRandom(seed + 7u, true);

    std::vector<float> cpu(n), gpu(n);
    batch_distance_singlethread(store, q.vectorAt(0), Metric::L2, cpu.data());

    GpuVectorIndex index = GpuVectorIndex::build(store.raw(), n, dim);
    index.search(q.vectorAt(0), Metric::L2, gpu.data(), kind, nullptr);
    report(std::string("L2 ") + kind_name + " vs CPU n=" + std::to_string(n) +
           " d=" + std::to_string(dim) + " seed=" + std::to_string(seed),
           cpu, gpu);

    if (kind != GpuKernelKind::Warp) {
        std::vector<float> cpu_c(n), gpu_c(n);
        batch_distance_singlethread(store, q.vectorAt(0), Metric::Cosine, cpu_c.data());
        index.search(q.vectorAt(0), Metric::Cosine, gpu_c.data(), kind, nullptr);
        report(std::string("Cosine ") + kind_name + " vs CPU n=" + std::to_string(n) +
               " d=" + std::to_string(dim),
               cpu_c, gpu_c);
    }
}

void run_batch(size_t n, size_t dim, size_t batch, unsigned seed) {
    VectorStore store(n, dim);
    store.fillRandom(seed, true);
    VectorStore queries(batch, dim);
    queries.fillRandom(seed + 11u, true);

    std::vector<float> cpu(batch * n), gpu(batch * n);
    for (size_t q = 0; q < batch; ++q) {
        batch_distance_singlethread(store, queries.vectorAt(q), Metric::L2,
                                    cpu.data() + q * n);
    }

    GpuVectorIndex index = GpuVectorIndex::build(store.raw(), n, dim);
    index.search_batch(queries.raw(), batch, Metric::L2, gpu.data(),
                       GpuKernelKind::Tiled, nullptr);
    report("batch tiled L2 vs CPU n=" + std::to_string(n) + " d=" + std::to_string(dim) +
               " Q=" + std::to_string(batch),
           cpu, gpu);

    index.search_batch(queries.raw(), batch, Metric::L2, gpu.data(),
                       GpuKernelKind::Warp, nullptr);
    report("batch warp L2 vs CPU n=" + std::to_string(n) + " d=" + std::to_string(dim) +
               " Q=" + std::to_string(batch),
           cpu, gpu);
}

void test_invalid_inputs() {
    VectorStore store(64, 32);
    store.fillRandom(1, true);
    try {
        GpuVectorIndex::build(nullptr, 64, 32);
        expect_throw("null store", false);
    } catch (const std::exception&) {
        expect_throw("null store", true);
    }
    try {
        GpuVectorIndex::build(store.raw(), 0, 32);
        expect_throw("zero N", false);
    } catch (const std::exception&) {
        expect_throw("zero N", true);
    }
    try {
        GpuVectorIndex::build(store.raw(), 64, 0);
        expect_throw("zero dim", false);
    } catch (const std::exception&) {
        expect_throw("zero dim", true);
    }

    GpuVectorIndex index = GpuVectorIndex::build(store.raw(), 64, 32);
    std::vector<float> scores(64);
    std::vector<float> bad_q(32, std::numeric_limits<float>::quiet_NaN());
    try {
        index.search(bad_q.data(), Metric::L2, scores.data(), GpuKernelKind::Tiled, nullptr);
        expect_throw("NaN query", false);
    } catch (const std::exception&) {
        expect_throw("NaN query", true);
    }

    try {
        index.search(store.vectorAt(0), Metric::Cosine, scores.data(), GpuKernelKind::Warp,
                     nullptr);
        expect_throw("warp+cosine unsupported", false);
    } catch (const std::exception&) {
        expect_throw("warp+cosine unsupported", true);
    }

    index.reset();
    if (index.empty()) {
        std::printf("passed: reset() empties index\n");
    } else {
        std::fprintf(stderr, "FAILED: reset() did not empty index\n");
        g_failures++;
    }
}

void test_stats_and_topk_shape() {
    VectorStore store(100, 64);
    store.fillRandom(3, true);
    auto index = GpuVectorIndex::build(store.raw(), 100, 64);
    auto st = index.stats();
    if (st.num_vectors == 100 && st.dim == 64 && st.query_capacity >= 1 &&
        !st.device_name.empty()) {
        std::printf("passed: stats() (device=%s corpus_bytes=%zu)\n",
                    st.device_name.c_str(), st.device_corpus_bytes);
    } else {
        std::fprintf(stderr, "FAILED: stats() unexpected values\n");
        g_failures++;
    }
}

void test_optional_1m() {
    const size_t n = 1000000;
    const size_t dim = 384;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t need = n * dim * sizeof(float) * 2 + (64ull << 20);
    if (free_b < need) {
        std::printf("skipped: 1M correctness (need ~%zu MiB free)\n",
                    need / (1024 * 1024));
        return;
    }
    std::printf("running: 1M×384 L2 tiled+warp vs CPU (may take a while)...\n");
    run_single(n, dim, 42, GpuKernelKind::Tiled, "tiled");
    run_single(n, dim, 42, GpuKernelKind::Warp, "warp");
}

} // namespace

int main() {
    std::printf("Correctness matrix — absolute tol = %.1e\n",
                static_cast<double>(kTolerance));
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc < 1) {
        std::fprintf(stderr, "No CUDA device\n");
        return 1;
    }

    const size_t sizes[] = {1, 7, 32, 33, 100, 257, 1000, 10000};
    const size_t dims[] = {1, 31, 32, 33, 383, 384, 385, 768, 1024};
    for (size_t dim : dims) {
        for (size_t n : sizes) {
            // Skip absurd tiny-host cases that still allocate fine.
            if (n > 10000 && dim > 384) continue;
            run_single(n, dim, 321u, GpuKernelKind::Tiled, "tiled");
            run_single(n, dim, 321u, GpuKernelKind::Warp, "warp");
            if (n <= 1000 && dim <= 384) {
                run_single(n, dim, 999u, GpuKernelKind::Naive, "naive");
            }
        }
    }

    for (size_t batch : {size_t(1), size_t(8), size_t(32), size_t(128)}) {
        run_batch(/*n=*/1000, /*dim=*/384, batch, /*seed=*/55);
        run_batch(/*n=*/10000, /*dim=*/128, batch, /*seed=*/56);
    }

    test_invalid_inputs();
    test_stats_and_topk_shape();
    test_optional_1m();

    if (g_failures == 0) {
        std::printf("\nAll correctness-matrix checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) FAILED\n", g_failures);
    return 1;
}
