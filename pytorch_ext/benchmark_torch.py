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
memory tiling on the query vector only) is NOT expected to beat them in
general -- and if it does on some shape, treat that as a measured data
point for that T4 / dtype / dim / device-resident setup, not a blanket win.
The point is to have a real number for "how close does a hand-rolled kernel
with one specific optimization get to a production-grade library op."

Run with:
    python benchmark_torch.py
    python benchmark_torch.py --dim 1024
"""
import argparse
import csv
import os
import torch
import torch.nn.functional as F

import vector_search_torch as vst

RESULTS_CSV = "../results/benchmark_torch.csv"
NUM_QUERIES = 20
DATASET_SIZES = [10_000, 100_000, 1_000_000]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=384,
                        help="Embedding dim (use 1024 to match BGE-large).")
    parser.add_argument("--queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--csv", default=RESULTS_CSV)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device available -- this benchmark requires a GPU.")
        return

    write_header = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "num_vectors", "dim", "avg_latency_ms"])

        for n in DATASET_SIZES:
            print(f"\n=== num_vectors = {n}, dim = {args.dim} ===")
            results = bench_one_size(n, args.dim, args.queries)
            for method, avg_ms in results.items():
                print(f"{method:<28} {avg_ms:.4f} ms/query")
                writer.writerow([method, n, args.dim, avg_ms])

    print(f"\nResults appended to {args.csv}")
    print("Plot with: python3 ../scripts/plot_results.py "
          f"--csv {args.csv} --dim {args.dim} "
          "--out ../results/latency_chart_torch.png")


if __name__ == "__main__":
    main()
