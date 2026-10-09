// Controlled, Colab-safe stress / failure cases for GpuVectorIndex.
// Prints one STRESS_JSON=... line per case. Does not crash the process on
// expected failures. Does not overwrite historical benchmark CSVs.
#include "gpu_vector_index.cuh"
#include "vector_store.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

void emit(const std::string& experiment, const std::string& status,
          size_t n, size_t dim, size_t batch, const std::string& detail) {
    // Minimal JSON (no external deps). Escape backslash and quotes in detail.
    std::string d;
    for (char c : detail) {
        if (c == '\\' || c == '"') d.push_back('\\');
        if (c == '\n' || c == '\r') { d += ' '; continue; }
        d.push_back(c);
    }
    if (d.size() > 400) d = d.substr(0, 400) + "...";
    std::printf(
        "STRESS_JSON={\"experiment\":\"%s\",\"status\":\"%s\",\"num_vectors\":%zu,"
        "\"dim\":%zu,\"batch\":%zu,\"detail\":\"%s\"}\n",
        experiment.c_str(), status.c_str(), n, dim, batch, d.c_str());
    std::fflush(stdout);
}

} // namespace

int main() {
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc < 1) {
        emit("gpu_present", "error", 0, 0, 0, "no CUDA device");
        return 1;
    }

    // --- invalid: null / zero dims ---
    try {
        GpuVectorIndex::build(nullptr, 10, 32);
        emit("stress_invalid_null_store", "error", 10, 32, 0, "did not throw");
    } catch (const std::exception& ex) {
        emit("stress_invalid_null_store", "expected_fail", 10, 32, 0, ex.what());
    }
    try {
        std::vector<float> dummy(32, 0.1f);
        GpuVectorIndex::build(dummy.data(), 0, 32);
        emit("stress_invalid_zero_n", "error", 0, 32, 0, "did not throw");
    } catch (const std::exception& ex) {
        emit("stress_invalid_zero_n", "expected_fail", 0, 32, 0, ex.what());
    }

    // --- NaN query ---
    {
        VectorStore store(256, 64);
        store.fillRandom(7, true);
        auto idx = GpuVectorIndex::build(store.raw(), 256, 64);
        std::vector<float> scores(256);
        std::vector<float> q(64, std::numeric_limits<float>::quiet_NaN());
        try {
            idx.search(q.data(), Metric::L2, scores.data(), GpuKernelKind::Tiled, nullptr);
            emit("stress_nan_query", "error", 256, 64, 1, "did not throw");
        } catch (const std::exception& ex) {
            emit("stress_nan_query", "expected_fail", 256, 64, 1, ex.what());
        }
    }

    // --- oversized batch relative to tiny free memory: grow score buffer ---
    // Safe: use moderate N but huge batch request; may OOM on score buffer.
    {
        VectorStore store(50000, 384);
        store.fillRandom(9, true);
        auto idx = GpuVectorIndex::build(store.raw(), 50000, 384);
        const size_t huge_batch = 4096;  // 4096 * 50000 * 4 ≈ 800 MiB scores
        std::vector<float> queries(huge_batch * 384, 0.01f);
        // normalize-ish first query only; rest dummy
        std::vector<float> scores;
        try {
            scores.resize(huge_batch * 50000);
            idx.search_batch(queries.data(), huge_batch, Metric::L2, scores.data(),
                             GpuKernelKind::Tiled, nullptr);
            emit("stress_oversized_batch", "ok", 50000, 384, huge_batch,
                 "completed (GPU had enough memory)");
        } catch (const std::exception& ex) {
            emit("stress_oversized_batch", "expected_fail", 50000, 384, huge_batch, ex.what());
        }
        idx.reset();
        cudaDeviceSynchronize();
    }

    // --- OOM: request more than free device memory via cudaMalloc (no huge host alloc) ---
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        void* p = nullptr;
        const size_t ask = free_b + (64ull << 20);  // free + 64 MiB
        cudaError_t err = cudaMalloc(&p, ask);
        if (err != cudaSuccess) {
            emit("stress_near_oom_alloc", "expected_fail", 0, 0, 0,
                 std::string("cudaMalloc(") + std::to_string(ask) + "): " +
                     cudaGetErrorString(err));
            cudaGetLastError();  // clear sticky error
        } else {
            cudaFree(p);
            emit("stress_near_oom_alloc", "ok", 0, 0, 0,
                 "unexpected success allocating free+64MiB");
        }
        cudaDeviceSynchronize();
    }

    emit("diagnostics_stress_done", "ok", 0, 0, 0, "all cases executed");
    return 0;
}
