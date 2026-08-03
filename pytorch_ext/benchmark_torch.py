"""
Benchmark: custom tiled CUDA kernel (via the PyTorch extension) vs. native
PyTorch operators (torch.cdist, F.cosine_similarity) for the same query-vs-
corpus distance computation, on data that's already resident on the GPU
(no host<->device transfer in the timed region -- that's the point of this
extension vs. the standalone bench_runner in ../src/benchmark/).

Honest expectation, stated up front rather than discovered after writing a
misleading bullet point: torch.cdist and F.cosine_similarity are backed by
cuBLAS/highly-tuned CUDA kernels that Meta's PyTorch team has spent years
optimizing. A hand-written kernel from a single optimization pass (shared
memory tiling on the query vector only) is NOT expected to beat them --
and if it does on some shape, that's more likely an artifact of a
specific case than a general win. The point of this benchmark is not "my
kernel beats PyTorch's" -- it's to have a real, measured number for
"how close does a hand-rolled kernel with one specific optimization get to
a production-grade library implementation," which is exactly the kind of
comparison a DevTech engineer would be asked to reason about.

Run with: python benchmark_torch.py
"""
import csv
import os
import time
import torch
import torch.nn.functional as F

import vector_search_torch as vst

RESULTS_CSV = "../results/benchmark_torch.csv"
NUM_QUERIES = 20
DATASET_SIZES = [10_000, 100_000, 1_000_000]
DIM = 384


def timed_gpu(fn, num_repeats):
    """Times `fn` (a zero-arg callable) using CUDA events, which measure
    actual GPU execution time rather than host-side wall clock -- wall
    clock would be misleading here since CUDA kernel launches are
    asynchronous from the host's perspective."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(num_repeats):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_repeats  # milliseconds


def bench_one_size(n, dim, num_queries):
    store = torch.randn(n, dim, device="cuda", dtype=torch.float32)
    query = torch.randn(dim, device="cuda", dtype=torch.float32)

    results = {}

    results["custom_cuda_l2"] = timed_gpu(
        lambda: vst.l2_distance(query, store), num_queries)
    results["pytorch_cdist_l2"] = timed_gpu(
        lambda: torch.cdist(query.unsqueeze(0), store), num_queries)

    results["custom_cuda_cosine"] = timed_gpu(
        lambda: vst.cosine_similarity(query, store), num_queries)
    results["pytorch_cosine_similarity"] = timed_gpu(
        lambda: F.cosine_similarity(query.unsqueeze(0), store), num_queries)

    return results


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available -- this benchmark requires a GPU.")
        return

    write_header = not os.path.exists(RESULTS_CSV)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "num_vectors", "dim", "avg_latency_ms"])

        for n in DATASET_SIZES:
            print(f"\n=== num_vectors = {n}, dim = {DIM} ===")
            results = bench_one_size(n, DIM, NUM_QUERIES)
            for method, avg_ms in results.items():
                print(f"{method:<28} {avg_ms:.4f} ms/query")
                writer.writerow([method, n, DIM, avg_ms])

    print(f"\nResults appended to {RESULTS_CSV}")
    print("Merge with the main results/benchmark.csv and re-run "
          "scripts/plot_results.py to include these on the same chart, "
          "or plot separately -- your call depending on how you want to "
          "present 'custom kernel vs. PyTorch' vs. 'GPU vs. CPU'.")


if __name__ == "__main__":
    main()
