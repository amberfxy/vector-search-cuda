#!/usr/bin/env python3
"""Plot latency-vs-dataset-size for every method found in results/benchmark.csv.

Usage:
    python3 scripts/plot_results.py [--csv results/benchmark.csv] [--out results/latency_chart.png]

Requires: pandas, matplotlib (pip install pandas matplotlib)
"""
import argparse
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("This script needs pandas and matplotlib: pip install pandas matplotlib", file=sys.stderr)
    sys.exit(1)

METHOD_LABELS = {
    "cpu_singlethread": "CPU (single-thread)",
    "cpu_openmp": "CPU (OpenMP, multi-thread)",
    "gpu_naive_wallclock": "GPU naive (wall-clock, incl. transfer)",
    "gpu_naive_kernel_only": "GPU naive (kernel-only)",
    "gpu_tiled_wallclock": "GPU tiled/shared-mem (wall-clock, incl. transfer)",
    "gpu_tiled_kernel_only": "GPU tiled/shared-mem (kernel-only)",
    "faiss_cpu": "FAISS (CPU)",
    "custom_cuda_l2": "Custom CUDA L2 (PyTorch ext, device-resident)",
    "pytorch_cdist_l2": "torch.cdist L2",
    "custom_cuda_cosine": "Custom CUDA cosine (PyTorch ext, device-resident)",
    "pytorch_cosine_similarity": "F.cosine_similarity",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--out", default="results/latency_chart.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in df.groupby("method"):
        group = group.sort_values("num_vectors")
        label = METHOD_LABELS.get(method, method)
        ax.plot(group["num_vectors"], group["avg_latency_ms"], marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of vectors (dataset size)")
    ax.set_ylabel("Average query latency (ms, log scale)")
    ax.set_title("Vector Similarity Search: Latency vs. Dataset Size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved chart to {args.out}")


if __name__ == "__main__":
    main()
