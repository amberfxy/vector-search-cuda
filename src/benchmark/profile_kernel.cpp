// Focused resident-kernel workload for Nsight Compute / Systems.
//
// Corpus is uploaded once via GpuVectorIndex BEFORE warm-up and BEFORE the
// marked profile region. Measured iterations call the same resident search
// path used by bench_harness (GpuVectorIndex::search).
//
// Usage (CUDA machine):
//   ./build/profile_kernel --num-vectors 1000000 --dim 384 --metric l2 \
//       --warmup 10 --iterations 20
//
// See docs/NSIGHT_PROFILING.md for ncu/nsys wrap commands.
#include "vector_store.hpp"
#include "distance_cpu.hpp"

#ifdef USE_CUDA
#include "gpu_vector_index.cuh"
#include "nvtx_ranges.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "  --num-vectors N     (default 1000000)\n"
        "  --dim D             (default 384)\n"
        "  --metric l2|cosine  (default l2)\n"
        "  --method tiled|naive|warp  (default tiled)\n"
        "                      warp = Phase 5A warp-per-vector L2 (resident)\n"
        "  --warmup W          (default 10)\n"
        "  --iterations I      (default 20)\n"
        "  --seed S            (default 123)\n"
        "  --help\n",
        argv0);
}

struct Config {
    size_t num_vectors = 1000000;
    size_t dim = 384;
    Metric metric = Metric::L2;
    std::string metric_name = "l2";
    std::string method_name = "tiled";
    int warmup = 10;
    int iterations = 20;
    unsigned seed = 123;
};

Config parse(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", n);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            std::exit(0);
        } else if (!std::strcmp(argv[i], "--num-vectors")) {
            cfg.num_vectors = static_cast<size_t>(std::atoll(need("--num-vectors")));
        } else if (!std::strcmp(argv[i], "--dim")) {
            cfg.dim = static_cast<size_t>(std::atoll(need("--dim")));
        } else if (!std::strcmp(argv[i], "--metric")) {
            cfg.metric_name = need("--metric");
            if (cfg.metric_name == "l2") cfg.metric = Metric::L2;
            else if (cfg.metric_name == "cosine") cfg.metric = Metric::Cosine;
            else {
                std::fprintf(stderr, "Unknown metric\n");
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--method")) {
            cfg.method_name = need("--method");
            if (cfg.method_name != "tiled" && cfg.method_name != "naive" &&
                cfg.method_name != "warp") {
                std::fprintf(stderr, "Unknown method (tiled|naive|warp)\n");
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup = std::atoi(need("--warmup"));
        } else if (!std::strcmp(argv[i], "--iterations")) {
            cfg.iterations = std::atoi(need("--iterations"));
        } else if (!std::strcmp(argv[i], "--seed")) {
            cfg.seed = static_cast<unsigned>(std::atoi(need("--seed")));
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (cfg.warmup < 0 || cfg.iterations < 1) {
        std::fprintf(stderr, "warmup >= 0 and iterations >= 1 required\n");
        std::exit(2);
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }

#ifndef USE_CUDA
    (void)parse;
    std::fprintf(stderr,
        "profile_kernel requires a CUDA build (nvcc + GPU).\n"
        "See docs/NSIGHT_PROFILING.md\n");
    return 1;
#else
    Config cfg = parse(argc, argv);
    if (cfg.method_name == "warp" && cfg.metric != Metric::L2) {
        std::fprintf(stderr, "--method warp currently supports --metric l2 only\n");
        return 2;
    }
    GpuKernelKind kind = GpuKernelKind::Tiled;
    if (cfg.method_name == "naive") kind = GpuKernelKind::Naive;
    else if (cfg.method_name == "warp") kind = GpuKernelKind::Warp;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        std::fprintf(stderr, "No CUDA device available.\n");
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    std::printf("=== profile_kernel workload metadata ===\n");
    std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("mode: resident (GpuVectorIndex)\n");
    std::printf("num_vectors: %zu\n", cfg.num_vectors);
    std::printf("dim: %zu\n", cfg.dim);
    std::printf("metric: %s\n", cfg.metric_name.c_str());
    std::printf("method: %s\n", cfg.method_name.c_str());
    std::printf("warmup: %d\n", cfg.warmup);
    std::printf("iterations: %d\n", cfg.iterations);
    std::printf("seed: %u\n", cfg.seed);
    std::printf("same_api_as_harness: GpuVectorIndex::search\n");
#if defined(VECTOR_SEARCH_USE_NVTX)
    std::printf("nvtx: enabled (query_h2d / distance_kernel / scores_d2h)\n");
#else
    std::printf("nvtx: disabled\n");
#endif

    VectorStore store(cfg.num_vectors, cfg.dim);
    store.fillRandom(/*seed=*/cfg.seed, /*normalize=*/true);
    const size_t nq = static_cast<size_t>(std::max(cfg.warmup + cfg.iterations, 1));
    VectorStore queries(nq, cfg.dim);
    queries.fillRandom(/*seed=*/cfg.seed + 333u, /*normalize=*/true);

    std::printf("Building GpuVectorIndex (corpus H2D once)...\n");
    GpuVectorIndex index(store.raw(), cfg.num_vectors, cfg.dim);
    std::printf("index_build_ms: %.3f\n", index.build_time_ms());
    std::printf("device_corpus_bytes: %zu\n", index.deviceCorpusBytes());

    std::vector<float> scores(cfg.num_vectors);

    std::printf("Warm-up (%d iterations)...\n", cfg.warmup);
    {
        NvtxRange warm("warmup");
        for (int i = 0; i < cfg.warmup; ++i) {
            index.search(queries.vectorAt(static_cast<size_t>(i % queries.numVectors())),
                         cfg.metric, scores.data(), kind, nullptr);
        }
        cudaDeviceSynchronize();
    }

    std::printf("=== PROFILE REGION BEGIN ===\n");
    std::fflush(stdout);
    {
        NvtxRange region("profile_region");
        for (int i = 0; i < cfg.iterations; ++i) {
            NvtxRange iter("profile_iteration");
            index.search(
                queries.vectorAt(static_cast<size_t>((cfg.warmup + i) % queries.numVectors())),
                cfg.metric, scores.data(), kind, nullptr);
        }
        cudaDeviceSynchronize();
    }
    std::printf("=== PROFILE REGION END ===\n");
    std::printf(
        "Done. Filter ncu with kernel names: l2_tiled_kernel, cosine_tiled_kernel, "
        "l2_naive_kernel, cosine_naive_kernel, l2_warp_per_vector_kernel.\n");
    return 0;
#endif
}
