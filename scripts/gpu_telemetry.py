#!/usr/bin/env python3
"""
Lightweight GPU telemetry for Colab / single-GPU machines.

Uses pynvml when available, otherwise parses `nvidia-smi`.
Does NOT invent multi-GPU, NVLink, PCIe, or datacenter topology claims.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GpuSample:
    ts: float
    gpu_util_pct: Optional[float] = None
    mem_used_mib: Optional[float] = None
    mem_total_mib: Optional[float] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    source: str = "unavailable"
    note: str = ""


@dataclass
class TelemetrySummary:
    source: str
    samples: int
    gpu_util_pct_mean: Optional[float] = None
    gpu_util_pct_max: Optional[float] = None
    mem_used_mib_mean: Optional[float] = None
    mem_used_mib_max: Optional[float] = None
    mem_total_mib: Optional[float] = None
    temperature_c_mean: Optional[float] = None
    temperature_c_max: Optional[float] = None
    power_w_mean: Optional[float] = None
    power_w_max: Optional[float] = None


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _max(xs: List[float]) -> Optional[float]:
    return max(xs) if xs else None


class GpuTelemetry:
    """Best-effort GPU sampler (single device index 0)."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml = None
        self._handle = None
        self.source = "unavailable"
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.source = "pynvml"
            return
        except Exception:
            self._nvml = None
            self._handle = None

        if shutil.which("nvidia-smi"):
            self.source = "nvidia-smi"
        else:
            self.source = "unavailable"

    def sample(self) -> GpuSample:
        ts = time.time()
        if self.source == "pynvml" and self._nvml and self._handle is not None:
            return self._sample_nvml(ts)
        if self.source == "nvidia-smi":
            return self._sample_smi(ts)
        return GpuSample(ts=ts, source="unavailable", note="no pynvml or nvidia-smi")

    def _sample_nvml(self, ts: float) -> GpuSample:
        pynvml = self._nvml
        h = self._handle
        util = None
        mem_used = mem_total = None
        temp = power = None
        note = ""
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            util = float(u.gpu)
        except Exception as exc:
            note += f"util:{exc};"
        try:
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used = float(m.used) / (1024 * 1024)
            mem_total = float(m.total) / (1024 * 1024)
        except Exception as exc:
            note += f"mem:{exc};"
        try:
            temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except Exception as exc:
            note += f"temp:{exc};"
        try:
            power = float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
        except Exception as exc:
            note += f"power:{exc};"
        return GpuSample(
            ts=ts,
            gpu_util_pct=util,
            mem_used_mib=mem_used,
            mem_total_mib=mem_total,
            temperature_c=temp,
            power_w=power,
            source="pynvml",
            note=note.strip(";"),
        )

    def _sample_smi(self, ts: float) -> GpuSample:
        query = (
            "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        )
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
            ).strip().splitlines()
            if not out:
                return GpuSample(ts=ts, source="nvidia-smi", note="empty nvidia-smi")
            parts = [p.strip() for p in out[0].split(",")]
            while len(parts) < 5:
                parts.append("")

            def fnum(s: str) -> Optional[float]:
                if not s or s.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None

            return GpuSample(
                ts=ts,
                gpu_util_pct=fnum(parts[0]),
                mem_used_mib=fnum(parts[1]),
                mem_total_mib=fnum(parts[2]),
                temperature_c=fnum(parts[3]),
                power_w=fnum(parts[4]),
                source="nvidia-smi",
            )
        except Exception as exc:
            return GpuSample(ts=ts, source="nvidia-smi", note=str(exc))

    def summarize(self, samples: List[GpuSample]) -> TelemetrySummary:
        utils = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
        mems = [s.mem_used_mib for s in samples if s.mem_used_mib is not None]
        temps = [s.temperature_c for s in samples if s.temperature_c is not None]
        powers = [s.power_w for s in samples if s.power_w is not None]
        totals = [s.mem_total_mib for s in samples if s.mem_total_mib is not None]
        return TelemetrySummary(
            source=self.source,
            samples=len(samples),
            gpu_util_pct_mean=_mean(utils),
            gpu_util_pct_max=_max(utils),
            mem_used_mib_mean=_mean(mems),
            mem_used_mib_max=_max(mems),
            mem_total_mib=totals[-1] if totals else None,
            temperature_c_mean=_mean(temps),
            temperature_c_max=_max(temps),
            power_w_mean=_mean(powers),
            power_w_max=_max(powers),
        )


class TelemetrySampler:
    """Background sampler while a workload runs."""

    def __init__(self, telemetry: GpuTelemetry, interval_s: float = 0.2):
        self.telemetry = telemetry
        self.interval_s = interval_s
        self.samples: List[GpuSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.telemetry.sample())
            self._stop.wait(self.interval_s)

    def stop(self) -> TelemetrySummary:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # one trailing sample
        self.samples.append(self.telemetry.sample())
        return self.telemetry.summarize(self.samples)


def write_samples_csv(path: Path, samples: List[GpuSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(GpuSample(ts=0.0)).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))


def gpu_name() -> str:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip().splitlines()
            return out[0] if out else "unknown"
        except Exception:
            pass
    return "unknown"
