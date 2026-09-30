#!/usr/bin/env python3
"""Plot Phase 4 legacy vs device-resident charts from measured harness CSV.

Reads:  results/phase1_benchmark.csv
Writes: results/phase4_resident_vs_legacy.png
        results/phase4_speedup.png

Values are taken from the CSV only — nothing is hardcoded.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"mode", "num_vectors", "dim", "e2e_mean_ms", "method", "metric"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    # Default harness matrix: tiled L2
    df = df[(df["method"] == "gpu_tiled") & (df["metric"] == "l2")].copy()
    rows = []
    for (n, dim), g in df.groupby(["num_vectors", "dim"]):
        leg = g[g["mode"] == "legacy"]
        res = g[g["mode"] == "resident"]
        if leg.empty or res.empty:
            continue
        leg = leg.iloc[0]
        res = res.iloc[0]
        rows.append(
            {
                "num_vectors": int(n),
                "dim": int(dim),
                "legacy_e2e_mean_ms": float(leg["e2e_mean_ms"]),
                "resident_e2e_mean_ms": float(res["e2e_mean_ms"]),
                "speedup": float(leg["e2e_mean_ms"]) / float(res["e2e_mean_ms"]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dim", "num_vectors"])
    if out.empty:
        raise SystemExit("No legacy/resident pairs found in CSV")
    return out


def plot_latency(pairs: pd.DataFrame, out_path: str) -> None:
    dims = sorted(pairs["dim"].unique())
    ns = sorted(pairs["num_vectors"].unique())
    x = np.arange(len(ns))
    width = 0.12

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Group bars: for each N, show legacy/resident for each dim
    colors_legacy = {384: "#4C78A8", 768: "#F58518", 1024: "#54A24B"}
    colors_resident = {384: "#9ECAE1", 768: "#FDC086", 1024: "#A1D99B"}

    offset = 0
    for dim in dims:
        sub = pairs[pairs["dim"] == dim].set_index("num_vectors").reindex(ns)
        ax.bar(
            x + offset,
            sub["legacy_e2e_mean_ms"].values,
            width,
            label=f"legacy dim={dim}",
            color=colors_legacy[dim],
            edgecolor="black",
            linewidth=0.4,
        )
        offset += width
        ax.bar(
            x + offset,
            sub["resident_e2e_mean_ms"].values,
            width,
            label=f"resident dim={dim}",
            color=colors_resident[dim],
            edgecolor="black",
            linewidth=0.4,
            hatch="//",
        )
        offset += width

    ax.set_yscale("log")
    ax.set_ylabel("End-to-end latency (ms, log scale)")
    ax.set_xlabel("Corpus size N")
    ax.set_xticks(x + width * (len(dims) - 0.5))
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_title(
        "NVIDIA Tesla T4 — Legacy vs device-resident GpuVectorIndex\n"
        "Mean end-to-end latency (gpu_tiled, L2; warm-up excluded)"
    )
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, which="both", axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_speedup(pairs: pd.DataFrame, out_path: str) -> None:
    dims = sorted(pairs["dim"].unique())
    ns = sorted(pairs["num_vectors"].unique())
    x = np.arange(len(ns))
    width = 0.25
    colors = {384: "#4C78A8", 768: "#F58518", 1024: "#54A24B"}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, dim in enumerate(dims):
        sub = pairs[pairs["dim"] == dim].set_index("num_vectors").reindex(ns)
        bars = ax.bar(
            x + (i - 1) * width,
            sub["speedup"].values,
            width,
            label=f"dim={dim}",
            color=colors[dim],
            edgecolor="black",
            linewidth=0.4,
        )
        for rect, val in zip(bars, sub["speedup"].values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() * 1.02,
                f"{val:.1f}×",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("E2E speedup (legacy mean / resident mean)")
    ax.set_xlabel("Corpus size N")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_title(
        "NVIDIA Tesla T4 — End-to-end speedup from device-resident index\n"
        "(not a claim that the distance kernel itself got faster)"
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join("results", "phase1_benchmark.csv"),
        help="Path to Phase 1/4 harness CSV",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Missing CSV: {args.csv}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.outdir, exist_ok=True)

    pairs = load_pairs(args.csv)
    latency_path = os.path.join(args.outdir, "phase4_resident_vs_legacy.png")
    speedup_path = os.path.join(args.outdir, "phase4_speedup.png")
    plot_latency(pairs, latency_path)
    plot_speedup(pairs, speedup_path)

    print("Wrote", latency_path)
    print("Wrote", speedup_path)
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
