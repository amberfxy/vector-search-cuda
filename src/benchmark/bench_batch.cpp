// Batched resident search microbenchmark → CSV.
#include "vector_store.hpp"
#include "distance_cpu.hpp"
#include "bench_stats.hpp"
#include "gpu_vector_index.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

void ensure_parent(const std::string& path) {
    auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return;
    mkdir(path.substr(0, slash).c_str(), 0755);
}

struct Cfg {
    size_t num_vectors = 100000;
    size_t dim = 384;
    size_t batch = 8;
    int warmup = 5;
    int iterations = 30;
    std::string method = "tiled";
    std::string csv = "results/batch_benchmark.csv";
};

Cfg parse(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* n) {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing %s\n", n); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--num-vectors")) c.num_vectors = std::atoll(need("--num-vectors"));
        else if (!std::strcmp(argv[i], "--dim")) c.dim = std::atoll(need("--dim"));
        else if (!std::strcmp(argv[i], "--batch")) c.batch = std::atoll(need("--batch"));
        else if (!std::strcmp(argv[i], "--warmup")) c.warmup = std::atoi(need("--warmup"));
        else if (!std::strcmp(argv[i], "--iterations")) c.iterations = std::atoi(need("--iterations"));
        else if (!std::strcmp(argv[i], "--method")) c.method = need("--method");
        else if (!std::strcmp(argv[i], "--csv")) c.csv = need("--csv");
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf("Usage: bench_batch --num-vectors N --dim D --batch Q --method tiled|warp\n");
            std::exit(0);
        }
    }
    return c;
}

} // namespace

int main(int argc, char** argv) {
    Cfg cfg = parse(argc, argv);
    int dc = 0;
    if (cudaGetDeviceCount(&dc) != cudaSuccess || dc < 1) {
        std::fprintf(stderr, "No CUDA device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    GpuKernelKind kind = GpuKernelKind::Tiled;
    if (cfg.method == "warp") kind = GpuKernelKind::Warp;
    else if (cfg.method == "naive") kind = GpuKernelKind::Naive;

    VectorStore store(cfg.num_vectors, cfg.dim);
    store.fillRandom(123, true);
    VectorStore queries(cfg.batch, cfg.dim);
    queries.fillRandom(456, true);

    auto index = GpuVectorIndex::build(store.raw(), cfg.num_vectors, cfg.dim);
    auto st = index.stats();
    std::vector<float> scores(cfg.batch * cfg.num_vectors);

    BenchStats e2e, kernel;
    const int total = cfg.warmup + cfg.iterations;
    for (int i = 0; i < total; ++i) {
        CudaStageTimes stages{};
        index.search_batch(queries.raw(), cfg.batch, Metric::L2, scores.data(), kind, &stages);
        if (i < cfg.warmup) continue;
        e2e.add(stages.e2e_ms);
        kernel.add(stages.kernel_ms);
    }

    const double qps = e2e.qps_from_mean() * static_cast<double>(cfg.batch);
    std::printf(
        "[batch] device=%s method=%s N=%zu dim=%zu batch=%zu\n"
        "  kernel mean=%.4f p50=%.4f p95=%.4f p99=%.4f\n"
        "  e2e mean=%.4f p50=%.4f p95=%.4f p99=%.4f\n"
        "  query_QPS~%.2f  device_mem_bytes=%zu\n",
        prop.name, cfg.method.c_str(), cfg.num_vectors, cfg.dim, cfg.batch,
        kernel.mean(), kernel.p50(), kernel.p95(), kernel.p99(),
        e2e.mean(), e2e.p50(), e2e.p95(), e2e.p99(),
        qps, st.device_corpus_bytes + st.device_query_capacity_bytes +
                 st.device_score_capacity_bytes);

    ensure_parent(cfg.csv);
    bool write_header = true;
    {
        std::ifstream in(cfg.csv);
        write_header = !in.good() || in.peek() == EOF;
    }
    std::ofstream out(cfg.csv, std::ios::app);
    if (write_header) {
        out << "device,method,num_vectors,dim,batch,warmup,iterations,"
               "kernel_mean_ms,kernel_p50_ms,kernel_p95_ms,kernel_p99_ms,"
               "e2e_mean_ms,e2e_p50_ms,e2e_p95_ms,e2e_p99_ms,"
               "query_qps,device_mem_bytes\n";
    }
    out << prop.name << ',' << cfg.method << ',' << cfg.num_vectors << ',' << cfg.dim << ','
        << cfg.batch << ',' << cfg.warmup << ',' << cfg.iterations << ','
        << kernel.mean() << ',' << kernel.p50() << ',' << kernel.p95() << ',' << kernel.p99() << ','
        << e2e.mean() << ',' << e2e.p50() << ',' << e2e.p95() << ',' << e2e.p99() << ','
        << qps << ','
        << (st.device_corpus_bytes + st.device_query_capacity_bytes +
            st.device_score_capacity_bytes)
        << '\n';
    std::printf("Wrote %s\n", cfg.csv.c_str());
    return 0;
}
