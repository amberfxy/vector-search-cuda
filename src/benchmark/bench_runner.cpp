// Benchmark driver. Builds a VectorStore of the requested size, runs a
// query against every implementation available in this build, and appends
// one CSV row per (method, num_vectors) pair to results/benchmark.csv.
//
// Usage:
//   ./bench_runner [dim] [num_queries_averaged]
// (num_vectors is swept internally across 10K / 100K / 1M -- edit
//  kDatasetSizes below to change that.)
//
// Build with CUDA (see CMakeLists.txt) to also exercise the naive and
// tiled GPU kernels; without CUDA, only the two CPU baselines run, which
// is still useful on its own (e.g. to double check OpenMP scaling before
// ever touching a GPU).
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "topk_cpu.hpp"
#include "data_gen.hpp"

#ifdef USE_CUDA
#include "distance_cuda.cuh"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
    const size_t dim = (argc > 1) ? std::atoi(argv[1]) : 384;   // matches common sentence-embedding dims
    const int num_queries = (argc > 2) ? std::atoi(argv[2]) : 10; // average over N queries per size

    const std::vector<size_t> kDatasetSizes = {10000, 100000, 1000000};

    const std::string csv_path = "results/benchmark.csv";
    bool write_header = true;
    {
        std::ifstream check(csv_path);
        write_header = !check.good();
    }
    std::ofstream csv(csv_path, std::ios::app);
    if (write_header) {
        csv << "method,num_vectors,dim,avg_latency_ms\n";
    }

    for (size_t n : kDatasetSizes) {
        std::printf("\n=== num_vectors = %zu, dim = %zu ===\n", n, dim);

        VectorStore store(n, dim);
        store.fillRandom(/*seed=*/123, /*normalize=*/true);

        VectorStore query_store(num_queries, dim);
        query_store.fillRandom(/*seed=*/456, /*normalize=*/true);

        std::vector<float> scores(n);

        // --- CPU single-threaded ---
        {
            auto t0 = Clock::now();
            for (int q = 0; q < num_queries; ++q) {
                batch_distance_singlethread(store, query_store.vectorAt(q), Metric::L2, scores.data());
            }
            double avg_ms = ms_since(t0) / num_queries;
            std::printf("CPU single-thread:      %.3f ms/query\n", avg_ms);
            csv << "cpu_singlethread," << n << "," << dim << "," << avg_ms << "\n";
        }

        // --- CPU OpenMP ---
        {
            auto t0 = Clock::now();
            for (int q = 0; q < num_queries; ++q) {
                batch_distance_openmp(store, query_store.vectorAt(q), Metric::L2, scores.data());
            }
            double avg_ms = ms_since(t0) / num_queries;
            std::printf("CPU OpenMP:              %.3f ms/query\n", avg_ms);
            csv << "cpu_openmp," << n << "," << dim << "," << avg_ms << "\n";
        }

#ifdef USE_CUDA
        // --- Naive CUDA (wall-clock, includes host<->device transfer) ---
        {
            auto t0 = Clock::now();
            float kernel_only_ms = 0.0f;
            for (int q = 0; q < num_queries; ++q) {
                batch_distance_naive_cuda(store.raw(), n, dim, query_store.vectorAt(q), Metric::L2, scores.data());
                kernel_only_ms += last_kernel_time_ms();
            }
            double avg_wall_ms = ms_since(t0) / num_queries;
            double avg_kernel_ms = kernel_only_ms / num_queries;
            std::printf("GPU naive (wall/kernel): %.3f / %.3f ms/query\n", avg_wall_ms, avg_kernel_ms);
            csv << "gpu_naive_wallclock," << n << "," << dim << "," << avg_wall_ms << "\n";
            csv << "gpu_naive_kernel_only," << n << "," << dim << "," << avg_kernel_ms << "\n";
        }

        // --- Shared-memory tiled CUDA ---
        {
            auto t0 = Clock::now();
            float kernel_only_ms = 0.0f;
            for (int q = 0; q < num_queries; ++q) {
                batch_distance_tiled_cuda(store.raw(), n, dim, query_store.vectorAt(q), Metric::L2, scores.data());
                kernel_only_ms += last_kernel_time_ms();
            }
            double avg_wall_ms = ms_since(t0) / num_queries;
            double avg_kernel_ms = kernel_only_ms / num_queries;
            std::printf("GPU tiled (wall/kernel): %.3f / %.3f ms/query\n", avg_wall_ms, avg_kernel_ms);
            csv << "gpu_tiled_wallclock," << n << "," << dim << "," << avg_wall_ms << "\n";
            csv << "gpu_tiled_kernel_only," << n << "," << dim << "," << avg_kernel_ms << "\n";
        }
#else
        std::printf("(built without CUDA -- skipping GPU benchmarks; see CMakeLists.txt)\n");
#endif
    }

    std::printf("\nResults appended to %s\n", csv_path.c_str());
    std::printf("Run 'python3 scripts/plot_results.py' to generate the latency chart.\n");
    return 0;
}
