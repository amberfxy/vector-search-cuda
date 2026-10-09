#!/usr/bin/env python3
"""
Colab-friendly diagnostics orchestrator for vector-search-cuda.

- Collects GPU telemetry (pynvml / nvidia-smi)
- Sweeps batch sizes via ./build/bench_batch when present
- Runs safe controlled stress / failure cases
- Emits CSV + JSON + a short markdown report with symptom→cause mapping

Does NOT overwrite results/phase1_benchmark.csv or invent Phase 4/5 numbers.
Does NOT claim multi-GPU / NVLink / PCIe topology expertise.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_telemetry import (  # noqa: E402
    GpuTelemetry,
    TelemetrySampler,
    gpu_name,
    write_samples_csv,
)


@dataclass
class ExperimentRow:
    experiment: str
    status: str
    num_vectors: Optional[int] = None
    dim: Optional[int] = None
    batch: Optional[int] = None
    method: Optional[str] = None
    kernel_mean_ms: Optional[float] = None
    e2e_mean_ms: Optional[float] = None
    query_qps: Optional[float] = None
    device_mem_bytes: Optional[int] = None
    gpu_util_pct_mean: Optional[float] = None
    gpu_util_pct_max: Optional[float] = None
    mem_used_mib_max: Optional[float] = None
    mem_total_mib: Optional[float] = None
    temperature_c_max: Optional[float] = None
    power_w_max: Optional[float] = None
    failure_mode: Optional[str] = None
    likely_cause: Optional[str] = None
    detail: str = ""


def run_cmd(cmd: List[str], cwd: Path, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def ensure_build(root: Path, jobs: int = 2) -> Dict[str, Any]:
    build = root / "build"
    info: Dict[str, Any] = {"built": False, "log": ""}
    need = [
        build / "bench_batch",
        build / "bench_harness",
        build / "diagnostics_stress",
    ]
    if all(p.exists() and os.access(p, os.X_OK) for p in need):
        info["built"] = True
        info["log"] = "binaries already present"
        return info

    cfg = run_cmd(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=root,
        timeout=600,
    )
    info["log"] += cfg.stdout
    if cfg.returncode != 0:
        info["error"] = "cmake configure failed"
        return info
    bld = run_cmd(
        ["cmake", "--build", "build", "-j", str(jobs),
         "--target", "bench_batch", "bench_harness", "diagnostics_stress"],
        cwd=root,
        timeout=1800,
    )
    info["log"] += bld.stdout
    info["built"] = bld.returncode == 0 and (build / "bench_batch").exists()
    if not info["built"]:
        info["error"] = "cmake build failed or CUDA unavailable"
    return info


def parse_bench_batch_stdout(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(
        r"kernel mean=([0-9.]+).*e2e mean=([0-9.]+).*query_QPS~([0-9.]+).*device_mem_bytes=([0-9]+)",
        text,
        re.S,
    )
    if m:
        out["kernel_mean_ms"] = float(m.group(1))
        out["e2e_mean_ms"] = float(m.group(2))
        out["query_qps"] = float(m.group(3))
        out["device_mem_bytes"] = int(m.group(4))
    return out


def diagnose_row(row: ExperimentRow, baseline_e2e: Optional[float] = None) -> ExperimentRow:
    """Map symptoms → likely cause (heuristic, single-GPU Colab scope)."""
    detail = (row.detail or "").lower()
    if row.status == "expected_fail":
        if "nan" in detail or "non-finite" in detail:
            row.failure_mode = "invalid_input"
            row.likely_cause = "Host-side validation rejected non-finite query values"
        elif "shape" in detail or "dim" in detail or "null" in detail or "zero" in detail:
            row.failure_mode = "invalid_input"
            row.likely_cause = "Invalid dimensions / null / empty arguments"
        elif "oom" in detail or "out of memory" in detail or "memory allocation" in detail:
            row.failure_mode = "memory_pressure_oom"
            row.likely_cause = "cudaMalloc failed under oversized N/dim/batch working set"
        elif "transfer" in row.experiment or "legacy" in row.experiment:
            row.failure_mode = "transfer_bottleneck"
            row.likely_cause = "Per-query corpus H2D dominates E2E (legacy path)"
        return row

    if row.status != "ok":
        if "oom" in detail or "out of memory" in detail:
            row.failure_mode = "memory_pressure_oom"
            row.likely_cause = "Device memory exhausted"
        elif "invalid" in detail or "nan" in detail:
            row.failure_mode = "invalid_input"
            row.likely_cause = "Bad input rejected or propagated"
        else:
            row.failure_mode = "runtime_error"
            row.likely_cause = "See detail / stdout"
        return row

    # Healthy runs: optional soft signals
    if (
        row.mem_total_mib
        and row.mem_used_mib_max
        and row.mem_used_mib_max / row.mem_total_mib > 0.92
    ):
        row.failure_mode = "memory_pressure_near_limit"
        row.likely_cause = "High device memory occupancy during workload"
    elif baseline_e2e and row.e2e_mean_ms and row.e2e_mean_ms > 3.0 * baseline_e2e:
        row.failure_mode = "kernel_or_e2e_regression"
        row.likely_cause = "E2E much slower than baseline batch=1 on same machine"
    elif row.experiment.startswith("stress_transfer") and row.e2e_mean_ms and row.kernel_mean_ms:
        if row.e2e_mean_ms > 5.0 * max(row.kernel_mean_ms, 1e-6):
            row.failure_mode = "transfer_bottleneck"
            row.likely_cause = "E2E >> kernel; host↔device transfer likely dominates"
    else:
        row.failure_mode = "none"
        row.likely_cause = "No failure signature detected"
    return row


def run_batch_sweep(
    root: Path,
    telemetry: GpuTelemetry,
    batches: List[int],
    n: int,
    dim: int,
    method: str,
    warmup: int,
    iters: int,
    out_csv: Path,
) -> List[ExperimentRow]:
    rows: List[ExperimentRow] = []
    bin_path = root / "build" / "bench_batch"
    if not bin_path.exists():
        return [
            ExperimentRow(
                experiment="batch_sweep",
                status="skipped",
                detail="bench_batch missing (build with CUDA)",
            )
        ]

    for batch in batches:
        sampler = TelemetrySampler(telemetry, interval_s=0.25)
        sampler.start()
        t0 = time.time()
        proc = run_cmd(
            [
                str(bin_path),
                "--num-vectors",
                str(n),
                "--dim",
                str(dim),
                "--batch",
                str(batch),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iters),
                "--method",
                method,
                "--csv",
                str(out_csv),
            ],
            cwd=root,
            timeout=3600,
        )
        summary = sampler.stop()
        parsed = parse_bench_batch_stdout(proc.stdout)
        status = "ok" if proc.returncode == 0 and parsed else "error"
        row = ExperimentRow(
            experiment=f"batch_sweep_Q{batch}",
            status=status,
            num_vectors=n,
            dim=dim,
            batch=batch,
            method=method,
            kernel_mean_ms=parsed.get("kernel_mean_ms"),
            e2e_mean_ms=parsed.get("e2e_mean_ms"),
            query_qps=parsed.get("query_qps"),
            device_mem_bytes=parsed.get("device_mem_bytes"),
            gpu_util_pct_mean=summary.gpu_util_pct_mean,
            gpu_util_pct_max=summary.gpu_util_pct_max,
            mem_used_mib_max=summary.mem_used_mib_max,
            mem_total_mib=summary.mem_total_mib,
            temperature_c_max=summary.temperature_c_max,
            power_w_max=summary.power_w_max,
            detail=proc.stdout[-2000:] if status != "ok" else f"elapsed_s={time.time()-t0:.2f}",
        )
        rows.append(row)
    return rows


def run_transfer_contrast(
    root: Path, telemetry: GpuTelemetry, n: int, dim: int, warmup: int, iters: int
) -> List[ExperimentRow]:
    """Legacy (transfer-heavy) vs resident using bench_harness stage timings."""
    bin_path = root / "build" / "bench_harness"
    if not bin_path.exists():
        return [
            ExperimentRow(
                experiment="stress_transfer",
                status="skipped",
                detail="bench_harness missing",
            )
        ]

    rows: List[ExperimentRow] = []
    for mode in ("legacy", "resident"):
        sampler = TelemetrySampler(telemetry, interval_s=0.25)
        sampler.start()
        proc = run_cmd(
            [
                str(bin_path),
                "--num-vectors",
                str(n),
                "--dim",
                str(dim),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iters),
                "--metric",
                "l2",
                "--mode",
                mode,
                "--method",
                "tiled",
                "--csv",
                str(root / "results" / "diagnostics" / "transfer_contrast.csv"),
                "--json",
                str(root / "results" / "diagnostics" / f"transfer_{mode}.json"),
            ],
            cwd=root,
            timeout=3600,
        )
        summary = sampler.stop()
        # Parse printed e2e/kernel means from harness stdout
        k = re.search(r"kernel-only:\s+mean=([0-9.]+)", proc.stdout)
        e = re.search(r"end-to-end:\s+mean=([0-9.]+)", proc.stdout)
        row = ExperimentRow(
            experiment=f"stress_transfer_{mode}",
            status="ok" if proc.returncode == 0 else "error",
            num_vectors=n,
            dim=dim,
            batch=1,
            method="tiled",
            kernel_mean_ms=float(k.group(1)) if k else None,
            e2e_mean_ms=float(e.group(1)) if e else None,
            gpu_util_pct_mean=summary.gpu_util_pct_mean,
            gpu_util_pct_max=summary.gpu_util_pct_max,
            mem_used_mib_max=summary.mem_used_mib_max,
            mem_total_mib=summary.mem_total_mib,
            temperature_c_max=summary.temperature_c_max,
            power_w_max=summary.power_w_max,
            detail=proc.stdout[-2500:],
        )
        if mode == "legacy" and row.e2e_mean_ms and row.kernel_mean_ms:
            if row.e2e_mean_ms > 5.0 * row.kernel_mean_ms:
                row.status = "expected_fail"  # intentional stress signature
                row.detail = "legacy_path_transfer_heavy; " + row.detail[:500]
        rows.append(row)
    return rows


def run_cpp_stress(root: Path, telemetry: GpuTelemetry) -> List[ExperimentRow]:
    bin_path = root / "build" / "diagnostics_stress"
    if not bin_path.exists():
        return [
            ExperimentRow(
                experiment="diagnostics_stress",
                status="skipped",
                detail="diagnostics_stress missing",
            )
        ]
    sampler = TelemetrySampler(telemetry, interval_s=0.2)
    sampler.start()
    proc = run_cmd([str(bin_path)], cwd=root, timeout=600)
    summary = sampler.stop()
    rows: List[ExperimentRow] = []
    # Each stress case prints a JSON line prefixed with STRESS_JSON=
    for line in proc.stdout.splitlines():
        if not line.startswith("STRESS_JSON="):
            continue
        try:
            obj = json.loads(line[len("STRESS_JSON=") :])
        except json.JSONDecodeError:
            continue
        row = ExperimentRow(
            experiment=obj.get("experiment", "stress"),
            status=obj.get("status", "error"),
            num_vectors=obj.get("num_vectors"),
            dim=obj.get("dim"),
            batch=obj.get("batch"),
            detail=obj.get("detail", ""),
            gpu_util_pct_mean=summary.gpu_util_pct_mean,
            gpu_util_pct_max=summary.gpu_util_pct_max,
            mem_used_mib_max=summary.mem_used_mib_max,
            mem_total_mib=summary.mem_total_mib,
            temperature_c_max=summary.temperature_c_max,
            power_w_max=summary.power_w_max,
        )
        rows.append(row)
    if not rows:
        rows.append(
            ExperimentRow(
                experiment="diagnostics_stress",
                status="error" if proc.returncode != 0 else "ok",
                detail=proc.stdout[-2000:],
                mem_used_mib_max=summary.mem_used_mib_max,
                mem_total_mib=summary.mem_total_mib,
            )
        )
    return rows


def write_rows_csv(path: Path, rows: List[ExperimentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(asdict(ExperimentRow(experiment="", status="")).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_report(path: Path, meta: Dict[str, Any], rows: List[ExperimentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = next((r for r in rows if r.experiment == "batch_sweep_Q1" and r.status == "ok"), None)
    stressed = [r for r in rows if r.failure_mode and r.failure_mode not in ("none", None)]

    lines = [
        "# Colab GPU diagnostics report",
        "",
        "## Experiment setup",
        "",
        f"- Time (UTC): {meta.get('timestamp')}",
        f"- GPU: {meta.get('gpu_name')}",
        f"- Telemetry source: {meta.get('telemetry_source')}",
        f"- Workload defaults: N={meta.get('num_vectors')}, dim={meta.get('dim')}, method={meta.get('method')}",
        f"- Batches: {meta.get('batches')}",
        f"- Note: Does **not** overwrite Phase 4/5 historical CSVs; numbers below are **this run only**.",
        "",
        "## Baseline vs stressed",
        "",
    ]
    if baseline:
        lines += [
            f"- Baseline batch=1: kernel_mean={baseline.kernel_mean_ms} ms, "
            f"e2e_mean={baseline.e2e_mean_ms} ms, QPS={baseline.query_qps}, "
            f"GPU util mean/max={baseline.gpu_util_pct_mean}/{baseline.gpu_util_pct_max}, "
            f"mem_used_max={baseline.mem_used_mib_max} MiB",
            "",
        ]
    else:
        lines += ["- Baseline batch=1: not available (build/GPU missing?)", ""]

    lines += ["| Experiment | Status | Failure mode | Likely cause | Telemetry highlights |",
              "|---|---|---|---|---|"]
    for r in rows:
        tel = (
            f"util_max={r.gpu_util_pct_max}, mem_max={r.mem_used_mib_max}MiB, "
            f"temp_max={r.temperature_c_max}, power_max={r.power_w_max}"
        )
        lines.append(
            f"| {r.experiment} | {r.status} | {r.failure_mode or ''} | "
            f"{(r.likely_cause or '')[:80]} | {tel} |"
        )

    lines += [
        "",
        "## Detected failure modes (this run)",
        "",
    ]
    if not stressed:
        lines.append("- No strong failure signatures (or stress binaries skipped).")
    else:
        for r in stressed:
            lines.append(f"- **{r.experiment}**: `{r.failure_mode}` — {r.likely_cause}")

    lines += [
        "",
        "## Scope limits",
        "",
        "- Single-GPU Colab/laptop telemetry only.",
        "- No NVLink / multi-GPU / PCIe topology / firmware claims.",
        "- Historical T4 Phase 4/5 verified metrics remain in `results/phase1_benchmark.csv` "
        "and `docs/PHASE5_WARP_PER_VECTOR.md` unless remeasured separately.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Colab GPU diagnostics for vector-search-cuda")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--num-vectors", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--method", default="tiled", choices=["tiled", "warp", "naive"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--transfer-n", type=int, default=100000, help="N for legacy vs resident contrast")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "results" / "diagnostics" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry = GpuTelemetry(0)
    idle = telemetry.sample()

    build_info: Dict[str, Any] = {"skipped": True}
    if not args.skip_build:
        print("Building CUDA targets (bench_batch, bench_harness, diagnostics_stress)...")
        build_info = ensure_build(root)
        print(build_info.get("log", "")[-2000:])
        if not build_info.get("built"):
            print("WARNING: build incomplete — sweeps/stress may skip. "
                  "On Colab: Runtime → GPU, install cmake/nvcc, retry.")

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    batch_csv = out_dir / "batch_benchmark_with_telemetry.csv"
    rows: List[ExperimentRow] = []

    print("Running batch sweep with telemetry...")
    sweep = run_batch_sweep(
        root,
        telemetry,
        batches,
        args.num_vectors,
        args.dim,
        args.method,
        args.warmup,
        args.iterations,
        batch_csv,
    )
    rows.extend(sweep)

    if not args.skip_stress:
        print("Running transfer-heavy contrast (legacy vs resident)...")
        rows.extend(
            run_transfer_contrast(
                root,
                telemetry,
                n=min(args.transfer_n, args.num_vectors),
                dim=args.dim,
                warmup=max(1, args.warmup // 2),
                iters=max(3, args.iterations // 2),
            )
        )
        print("Running controlled C++ stress cases...")
        rows.extend(run_cpp_stress(root, telemetry))

    baseline_e2e = next(
        (r.e2e_mean_ms for r in rows if r.experiment == "batch_sweep_Q1" and r.e2e_mean_ms),
        None,
    )
    rows = [diagnose_row(r, baseline_e2e) for r in rows]

    meta = {
        "timestamp": stamp,
        "gpu_name": gpu_name(),
        "telemetry_source": telemetry.source,
        "idle_sample": asdict(idle),
        "num_vectors": args.num_vectors,
        "dim": args.dim,
        "batches": batches,
        "method": args.method,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "build": {k: build_info.get(k) for k in ("built", "skipped", "error")},
        "scope": "single-GPU Colab/laptop diagnostics; not a datacenter study",
    }

    experiments_csv = out_dir / "experiments.csv"
    experiments_json = out_dir / "experiments.json"
    report_md = out_dir / "REPORT.md"
    write_rows_csv(experiments_csv, rows)
    experiments_json.write_text(
        json.dumps({"meta": meta, "experiments": [asdict(r) for r in rows]}, indent=2) + "\n"
    )
    write_report(report_md, meta, rows)

    # Also copy a stable "latest" pointer (overwrite only diagnostics/latest)
    latest = root / "results" / "diagnostics" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("experiments.csv", "experiments.json", "REPORT.md"):
        src = out_dir / name
        dst = latest / name
        dst.write_bytes(src.read_bytes())

    print(f"\nWrote:\n  {experiments_csv}\n  {experiments_json}\n  {report_md}")
    print(f"Latest pointer: {latest}")
    return 0 if any(r.status == "ok" for r in rows) or any(r.status == "expected_fail" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
