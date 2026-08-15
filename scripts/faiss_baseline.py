#!/usr/bin/env python3
"""FAISS CPU baseline using IndexFlatL2 (exact brute-force L2 on CPU).

This is the fair apples-to-apples line for this repo's exact L2 kernels --
not IVF/HNSW, and not a claim about FAISS approximate indexes.

Usage:
    python3 scripts/faiss_baseline.py [--dim 384] [--queries 50]
    python3 scripts/faiss_baseline.py --dim 1024 --queries 50 --csv results/benchmark_dim1024.csv

Requires: faiss-cpu, numpy (pip install faiss-cpu numpy)
"""
import argparse
import csv
import time
import os

try:
    import numpy as np
    import faiss
except ImportError:
    print("This script needs faiss-cpu and numpy: pip install faiss-cpu numpy")
    raise SystemExit(1)


def bench_one_size(n, dim, num_queries, rng):
    store = rng.standard_normal((n, dim)).astype("float32")
    store /= np.linalg.norm(store, axis=1, keepdims=True)
    queries = rng.standard_normal((num_queries, dim)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    index = faiss.IndexFlatL2(dim)   # brute-force exact search -- the fair
    index.add(store)                 # comparison point for our own brute-force kernels

    start = time.perf_counter()
    for q in queries:
        index.search(q.reshape(1, -1), 5)  # top-5, mirrors a typical RAG retrieval call
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / num_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--csv", default="results/benchmark.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=123)
    sizes = [10_000, 100_000, 1_000_000]

    write_header = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["method", "num_vectors", "dim", "avg_latency_ms"])
        for n in sizes:
            avg_ms = bench_one_size(n, args.dim, args.queries, rng)
            print(f"FAISS CPU  n={n:>8}  avg_latency_ms={avg_ms:.3f}")
            writer.writerow(["faiss_cpu", n, args.dim, avg_ms])

    print(f"\nAppended results to {args.csv}")


if __name__ == "__main__":
    main()
