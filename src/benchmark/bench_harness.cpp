// Phase 1 / Phase 4 benchmark harness.
// Compares legacy (re-upload corpus every query) vs device-resident GpuVectorIndex.
//
// Requires CUDA. Exits non-zero if no GPU / no USE_CUDA build.
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "bench_stats.hpp"

#ifdef USE_CUDA
#include "distance_cuda.cuh"
#include "gpu_vector_index.cuh"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {

void ensure_parent_dir(const std::string& path) {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return;
    const std::string dir = path.substr(0, slash);
    if (!dir.empty()) mkdir(dir.c_str(), 0755);
}

struct RunConfig {
    size_t num_vectors = 10000;
    size_t dim = 384;
    int warmup = 5;
    int iterations = 50;
    Metric metric = Metric::L2;
    std::string metric_name = "l2";
    std::string mode = "both";       // legacy | resident | both
    std::string method = "tiled";    // naive | tiled | warp | both | tiled,warp
    std::string csv_path = "results/phase1_benchmark.csv";
    std::string json_path = "results/phase1_benchmark.json";
    bool append_csv = true;
};

struct AggregateRow {
    std::string mode;
    std::string method;
    size_t num_vectors = 0;
    size_t dim = 0;
    std::string metric;
    int batch_size = 1;
    int concurrency = 1;
    int warmup = 0;
    int iterations = 0;
    double index_build_ms = 0.0;
    double alloc_mean_ms = 0.0;
    double corpus_h2d_mean_ms = 0.0;
    double query_h2d_mean_ms = 0.0;
    double kernel_mean_ms = 0.0;
    double kernel_p50_ms = 0.0;
    double kernel_p95_ms = 0.0;
    double kernel_p99_ms = 0.0;
    double scores_d2h_mean_ms = 0.0;
    double free_mean_ms = 0.0;
    double e2e_mean_ms = 0.0;
    double e2e_p50_ms = 0.0;
    double e2e_p95_ms = 0.0;
    double e2e_p99_ms = 0.0;
    double e2e_min_ms = 0.0;
    double e2e_max_ms = 0.0;
    double qps_from_e2e_mean = 0.0;
    long long gpu_mem_used_bytes = 0;
};

void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "  --num-vectors N       corpus size (default 10000)\n"
        "  --dim D               embedding dim (default 384)\n"
        "  --warmup W            discarded warm-up iters (default 5)\n"
        "  --iterations I        measured iters (default 50)\n"
        "  --metric l2|cosine    (default l2)\n"
        "  --mode legacy|resident|both  (default both)\n"
        "  --method naive|tiled|warp|both|tiled,warp  (default tiled)\n"
        "                        warp = Phase 5A warp-per-vector L2 (resident)\n"
        "  --csv PATH            (default results/phase1_benchmark.csv)\n"
        "  --json PATH           (default results/phase1_benchmark.json)\n"
        "  --no-append           overwrite CSV instead of append\n",
        argv0);
}

RunConfig parse_args(int argc, char** argv) {
    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (!std::strcmp(argv[i], "--num-vectors")) {
            cfg.num_vectors = static_cast<size_t>(std::atoll(need("--num-vectors")));
        } else if (!std::strcmp(argv[i], "--dim")) {
            cfg.dim = static_cast<size_t>(std::atoll(need("--dim")));
        } else if (!std::strcmp(argv[i], "--warmup")) {
            cfg.warmup = std::atoi(need("--warmup"));
        } else if (!std::strcmp(argv[i], "--iterations")) {
            cfg.iterations = std::atoi(need("--iterations"));
        } else if (!std::strcmp(argv[i], "--metric")) {
            cfg.metric_name = need("--metric");
            if (cfg.metric_name == "l2") cfg.metric = Metric::L2;
            else if (cfg.metric_name == "cosine") cfg.metric = Metric::Cosine;
            else {
                std::fprintf(stderr, "Unknown metric '%s'\n", cfg.metric_name.c_str());
                std::exit(2);
            }
        } else if (!std::strcmp(argv[i], "--mode")) {
            cfg.mode = need("--mode");
        } else if (!std::strcmp(argv[i], "--method")) {
            cfg.method = need("--method");
        } else if (!std::strcmp(argv[i], "--csv")) {
            cfg.csv_path = need("--csv");
        } else if (!std::strcmp(argv[i], "--json")) {
            cfg.json_path = need("--json");
        } else if (!std::strcmp(argv[i], "--no-append")) {
            cfg.append_csv = false;
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    if (cfg.warmup < 0 || cfg.iterations < 1) {
        std::fprintf(stderr, "warmup >= 0 and iterations >= 1 required\n");
        std::exit(2);
    }
    return cfg;
}

long long gpu_mem_used_bytes() {
#ifdef USE_CUDA
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return static_cast<long long>(total_b - free_b);
#else
    return 0;
#endif
}

AggregateRow finalize_row(const std::string& mode, const std::string& method,
                           const RunConfig& cfg, double index_build_ms,
                           const BenchStats& e2e, const BenchStats& kernel,
                           const BenchStats& alloc, const BenchStats& corpus_h2d,
                           const BenchStats& query_h2d, const BenchStats& d2h,
                           const BenchStats& freest) {
    AggregateRow r;
    r.mode = mode;
    r.method = method;
    r.num_vectors = cfg.num_vectors;
    r.dim = cfg.dim;
    r.metric = cfg.metric_name;
    r.warmup = cfg.warmup;
    r.iterations = cfg.iterations;
    r.index_build_ms = index_build_ms;
    r.alloc_mean_ms = alloc.count() ? alloc.mean() : 0.0;
    r.corpus_h2d_mean_ms = corpus_h2d.count() ? corpus_h2d.mean() : 0.0;
    r.query_h2d_mean_ms = query_h2d.count() ? query_h2d.mean() : 0.0;
    r.kernel_mean_ms = kernel.mean();
    r.kernel_p50_ms = kernel.p50();
    r.kernel_p95_ms = kernel.p95();
    r.kernel_p99_ms = kernel.p99();
    r.scores_d2h_mean_ms = d2h.count() ? d2h.mean() : 0.0;
    r.free_mean_ms = freest.count() ? freest.mean() : 0.0;
    r.e2e_mean_ms = e2e.mean();
    r.e2e_p50_ms = e2e.p50();
    r.e2e_p95_ms = e2e.p95();
    r.e2e_p99_ms = e2e.p99();
    r.e2e_min_ms = e2e.min();
    r.e2e_max_ms = e2e.max();
    r.qps_from_e2e_mean = e2e.qps_from_mean();
    r.gpu_mem_used_bytes = gpu_mem_used_bytes();
    return r;
}

void print_row(const AggregateRow& r) {
    std::printf(
        "[%s/%s] N=%zu dim=%zu metric=%s\n"
        "  index_build_ms=%.3f\n"
        "  stages mean ms: alloc=%.3f corpus_h2d=%.3f query_h2d=%.3f "
        "kernel=%.3f d2h=%.3f free=%.3f\n"
        "  kernel-only: mean=%.3f p50=%.3f p95=%.3f p99=%.3f\n"
        "  end-to-end:  mean=%.3f p50=%.3f p95=%.3f p99=%.3f  QPS~%.2f\n"
        "  gpu_mem_used_bytes=%lld\n",
        r.mode.c_str(), r.method.c_str(), r.num_vectors, r.dim, r.metric.c_str(),
        r.index_build_ms,
        r.alloc_mean_ms, r.corpus_h2d_mean_ms, r.query_h2d_mean_ms,
        r.kernel_mean_ms, r.scores_d2h_mean_ms, r.free_mean_ms,
        r.kernel_mean_ms, r.kernel_p50_ms, r.kernel_p95_ms, r.kernel_p99_ms,
        r.e2e_mean_ms, r.e2e_p50_ms, r.e2e_p95_ms, r.e2e_p99_ms, r.qps_from_e2e_mean,
        r.gpu_mem_used_bytes);
}

const char* CSV_HEADER =
    "mode,method,num_vectors,dim,metric,batch_size,concurrency,warmup,iterations,"
    "index_build_ms,alloc_mean_ms,corpus_h2d_mean_ms,query_h2d_mean_ms,"
    "kernel_mean_ms,kernel_p50_ms,kernel_p95_ms,kernel_p99_ms,"
    "scores_d2h_mean_ms,free_mean_ms,"
    "e2e_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_p99_ms,e2e_min_ms,e2e_max_ms,"
    "qps_from_e2e_mean,gpu_mem_used_bytes\n";

void write_csv_row(std::ostream& os, const AggregateRow& r) {
    os << r.mode << ',' << r.method << ',' << r.num_vectors << ',' << r.dim << ','
       << r.metric << ',' << r.batch_size << ',' << r.concurrency << ','
       << r.warmup << ',' << r.iterations << ','
       << r.index_build_ms << ',' << r.alloc_mean_ms << ',' << r.corpus_h2d_mean_ms << ','
       << r.query_h2d_mean_ms << ','
       << r.kernel_mean_ms << ',' << r.kernel_p50_ms << ',' << r.kernel_p95_ms << ','
       << r.kernel_p99_ms << ','
       << r.scores_d2h_mean_ms << ',' << r.free_mean_ms << ','
       << r.e2e_mean_ms << ',' << r.e2e_p50_ms << ',' << r.e2e_p95_ms << ','
       << r.e2e_p99_ms << ',' << r.e2e_min_ms << ',' << r.e2e_max_ms << ','
       << r.qps_from_e2e_mean << ',' << r.gpu_mem_used_bytes << '\n';
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

void write_json(const std::string& path, const std::string& hardware_note,
                 const std::vector<AggregateRow>& rows) {
    ensure_parent_dir(path);
    std::ofstream os(path);
    os << std::setprecision(6);
    os << "{\n  \"schema_version\": 1,\n  \"hardware_note\": \""
       << json_escape(hardware_note) << "\",\n  \"runs\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        os << "    {"
           << "\"mode\":\"" << json_escape(r.mode) << "\","
           << "\"method\":\"" << json_escape(r.method) << "\","
           << "\"num_vectors\":" << r.num_vectors << ","
           << "\"dim\":" << r.dim << ","
           << "\"metric\":\"" << json_escape(r.metric) << "\","
           << "\"batch_size\":" << r.batch_size << ","
           << "\"concurrency\":" << r.concurrency << ","
           << "\"warmup\":" << r.warmup << ","
           << "\"iterations\":" << r.iterations << ","
           << "\"index_build_ms\":" << r.index_build_ms << ","
           << "\"alloc_mean_ms\":" << r.alloc_mean_ms << ","
           << "\"corpus_h2d_mean_ms\":" << r.corpus_h2d_mean_ms << ","
           << "\"query_h2d_mean_ms\":" << r.query_h2d_mean_ms << ","
           << "\"kernel_mean_ms\":" << r.kernel_mean_ms << ","
           << "\"kernel_p50_ms\":" << r.kernel_p50_ms << ","
           << "\"kernel_p95_ms\":" << r.kernel_p95_ms << ","
           << "\"kernel_p99_ms\":" << r.kernel_p99_ms << ","
           << "\"scores_d2h_mean_ms\":" << r.scores_d2h_mean_ms << ","
           << "\"free_mean_ms\":" << r.free_mean_ms << ","
           << "\"e2e_mean_ms\":" << r.e2e_mean_ms << ","
           << "\"e2e_p50_ms\":" << r.e2e_p50_ms << ","
           << "\"e2e_p95_ms\":" << r.e2e_p95_ms << ","
           << "\"e2e_p99_ms\":" << r.e2e_p99_ms << ","
           << "\"e2e_min_ms\":" << r.e2e_min_ms << ","
           << "\"e2e_max_ms\":" << r.e2e_max_ms << ","
           << "\"qps_from_e2e_mean\":" << r.qps_from_e2e_mean << ","
           << "\"gpu_mem_used_bytes\":" << r.gpu_mem_used_bytes
           << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    os << "  ]\n}\n";
}

#ifdef USE_CUDA

GpuKernelKind parse_kernel_kind(const std::string& method) {
    if (method == "naive") return GpuKernelKind::Naive;
    if (method == "tiled") return GpuKernelKind::Tiled;
    if (method == "warp") return GpuKernelKind::Warp;
    throw std::runtime_error("Unknown method: " + method);
}

const char* method_csv_name(const std::string& method) {
    if (method == "naive") return "gpu_naive";
    if (method == "tiled") return "gpu_tiled";
    if (method == "warp") return "gpu_warp";
    return method.c_str();
}

AggregateRow run_legacy(const RunConfig& cfg, const VectorStore& store,
                         const VectorStore& queries, const std::string& method) {
    if (method == "warp") {
        throw std::runtime_error(
            "legacy mode does not support --method warp (use --mode resident)");
    }
    const std::string csv_method = method_csv_name(method);
    std::vector<float> scores(cfg.num_vectors);
    BenchStats e2e, kernel, alloc, corpus_h2d, query_h2d, d2h, freest;

    const int total = cfg.warmup + cfg.iterations;
    for (int i = 0; i < total; ++i) {
        const float* q = queries.vectorAt(static_cast<size_t>(i % queries.numVectors()));
        CudaStageTimes st{};
        if (method == "tiled") {
            batch_distance_tiled_cuda_staged(store.raw(), cfg.num_vectors, cfg.dim, q,
                                              cfg.metric, scores.data(), &st);
        } else {
            batch_distance_naive_cuda_staged(store.raw(), cfg.num_vectors, cfg.dim, q,
                                              cfg.metric, scores.data(), &st);
        }
        if (i < cfg.warmup) continue;
        e2e.add(st.e2e_ms);
        kernel.add(st.kernel_ms);
        alloc.add(st.alloc_ms);
        corpus_h2d.add(st.corpus_h2d_ms);
        query_h2d.add(st.query_h2d_ms);
        d2h.add(st.scores_d2h_ms);
        freest.add(st.free_ms);
    }

    return finalize_row("legacy", csv_method, cfg, /*index_build_ms=*/0.0,
                        e2e, kernel, alloc, corpus_h2d, query_h2d, d2h, freest);
}

AggregateRow run_resident(const RunConfig& cfg, const VectorStore& store,
                           const VectorStore& queries, const std::string& method) {
    if (method == "warp" && cfg.metric != Metric::L2) {
        throw std::runtime_error("--method warp currently supports --metric l2 only");
    }
    const std::string csv_method = method_csv_name(method);
    GpuVectorIndex index(store.raw(), cfg.num_vectors, cfg.dim);
    const double build_ms = index.build_time_ms();

    std::vector<float> scores(cfg.num_vectors);
    BenchStats e2e, kernel, alloc, corpus_h2d, query_h2d, d2h, freest;
    const GpuKernelKind kind = parse_kernel_kind(method);

    const int total = cfg.warmup + cfg.iterations;
    for (int i = 0; i < total; ++i) {
        const float* q = queries.vectorAt(static_cast<size_t>(i % queries.numVectors()));
        CudaStageTimes st{};
        index.search(q, cfg.metric, scores.data(), kind, &st);
        if (i < cfg.warmup) continue;
        e2e.add(st.e2e_ms);
        kernel.add(st.kernel_ms);
        alloc.add(st.alloc_ms);
        corpus_h2d.add(st.corpus_h2d_ms);
        query_h2d.add(st.query_h2d_ms);
        d2h.add(st.scores_d2h_ms);
        freest.add(st.free_ms);
    }

    return finalize_row("resident", csv_method, cfg, build_ms,
                        e2e, kernel, alloc, corpus_h2d, query_h2d, d2h, freest);
}

std::vector<std::string> expand_methods(const std::string& method) {
    if (method == "both") return {"naive", "tiled"};
    if (method == "tiled,warp" || method == "warp,tiled") return {"tiled", "warp"};
    if (method == "naive" || method == "tiled" || method == "warp") return {method};
    throw std::runtime_error("Invalid --method " + method);
}

#endif // USE_CUDA

} // namespace

int main(int argc, char** argv) {
    // Allow --help even in CPU-only builds.
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        }
    }
#ifndef USE_CUDA
    std::fprintf(stderr,
        "bench_harness requires a CUDA build (nvcc + GPU).\n"
        "This binary was built without USE_CUDA. Reconfigure with a CUDA toolkit.\n");
    return 1;
#else
    RunConfig cfg = parse_args(argc, argv);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count < 1) {
        std::fprintf(stderr,
            "No CUDA-capable GPU available (cudaGetDeviceCount failed or returned 0).\n"
            "Harness implemented; re-run on a machine with a GPU (e.g. Colab T4).\n"
            "Do not invent benchmark numbers.\n");
        return 1;
    }

    cudaDeviceProp prop{};
    std::ostringstream hw;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        hw << "unknown CUDA device";
    } else {
        hw << prop.name << " (sm_" << prop.major << prop.minor << ", "
           << (prop.totalGlobalMem / (1024ull * 1024ull)) << " MiB)";
    }

    std::printf("Hardware: %s\n", hw.str().c_str());
    std::printf("Config: N=%zu dim=%zu warmup=%d iters=%d metric=%s mode=%s method=%s\n",
                cfg.num_vectors, cfg.dim, cfg.warmup, cfg.iterations,
                cfg.metric_name.c_str(), cfg.mode.c_str(), cfg.method.c_str());

    // Memory sanity: float32 store bytes
    const double store_gb =
        (static_cast<double>(cfg.num_vectors) * cfg.dim * 4.0) / (1024.0 * 1024.0 * 1024.0);
    std::printf("Corpus size on host: %.3f GiB (float32)\n", store_gb);

    VectorStore store(cfg.num_vectors, cfg.dim);
    store.fillRandom(/*seed=*/123, /*normalize=*/true);
    // Enough distinct queries to rotate through warm-up + measured iters
    const size_t nq = static_cast<size_t>(std::max(cfg.warmup + cfg.iterations, 1));
    VectorStore queries(nq, cfg.dim);
    queries.fillRandom(/*seed=*/456, /*normalize=*/true);

    std::vector<std::string> modes;
    if (cfg.mode == "both") { modes = {"legacy", "resident"}; }
    else if (cfg.mode == "legacy" || cfg.mode == "resident") { modes = {cfg.mode}; }
    else {
        std::fprintf(stderr, "Invalid --mode %s\n", cfg.mode.c_str());
        return 2;
    }

    std::vector<std::string> methods;
    try {
        methods = expand_methods(cfg.method);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 2;
    }

    std::vector<AggregateRow> rows;
    try {
        for (const auto& mode : modes) {
            for (const auto& method : methods) {
                if (mode == "legacy" && method == "warp") {
                    std::printf("skip: legacy + warp not supported\n");
                    continue;
                }
                AggregateRow row = (mode == "legacy")
                    ? run_legacy(cfg, store, queries, method)
                    : run_resident(cfg, store, queries, method);
                print_row(row);
                rows.push_back(row);
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Benchmark failed: %s\n", ex.what());
        return 1;
    }

    ensure_parent_dir(cfg.csv_path);
    bool write_header = !cfg.append_csv;
    if (cfg.append_csv) {
        std::ifstream check(cfg.csv_path);
        write_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream csv(cfg.csv_path, cfg.append_csv ? std::ios::app : std::ios::trunc);
    if (write_header) csv << CSV_HEADER;
    for (const auto& r : rows) write_csv_row(csv, r);
    std::printf("Wrote CSV: %s\n", cfg.csv_path.c_str());

    write_json(cfg.json_path, hw.str(), rows);
    std::printf("Wrote JSON: %s\n", cfg.json_path.c_str());
    return 0;
#endif
}
