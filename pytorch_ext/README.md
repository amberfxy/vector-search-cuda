# PyTorch CUDA Extension

A custom PyTorch C++/CUDA extension exposing the main project's
shared-memory-tiled distance kernel (`src/cuda/distance_tiled.cu`) as a
native `torch.Tensor`-in, `torch.Tensor`-out operator, benchmarked against
PyTorch's own built-in `torch.cdist` and `F.cosine_similarity`.

## Why this exists (separate from the standalone `bench_runner`)

The main project's `bench_runner` intentionally re-uploads the full
dataset to the GPU on every single call (`cudaMalloc` + `cudaMemcpy` each
time) -- see the main README's "Known limitations" section, which flags
this as unrealistic for a real serving workload. This extension fixes
exactly that: it operates on tensors that are already resident on the GPU
(as real PyTorch tensors would be in an actual inference pipeline), with
no host<->device transfer in the timed path.

It also answers a more honest, more interesting benchmarking question
than "is my kernel fast in isolation": **how does a kernel with one
specific hand-written optimization compare to PyTorch's own
production-grade, cuBLAS-backed operators?** (See "Honest expectation" in
`benchmark_torch.py` -- spoiler: probably not faster, and that's fine and
worth stating plainly rather than around.)

## Files

```
csrc/
  tiled_kernels.cu      -- CUDA kernels, adapted to operate on
                            already-device-resident memory (no
                            cudaMalloc/Memcpy -- that's the whole point)
  binding.cpp             -- pybind11 binding: validates torch::Tensor
                             inputs (device/dtype/shape/contiguity),
                             dispatches to the kernel launchers
setup.py                  -- builds the extension via
                             torch.utils.cpp_extension.CUDAExtension
vector_search_torch.py    -- clean Python wrapper (import this, not the
                             raw compiled module)
test_correctness_torch.py -- checks custom kernel output against
                             torch.cdist / F.cosine_similarity
benchmark_torch.py         -- custom kernel vs. native PyTorch ops, timed
                             with torch.cuda.Event
demo_rerank.py              -- end-to-end usage: embed a query, score
                             against a corpus, take top-k -- the actual
                             shape of a retrieval step, not just an
                             isolated distance computation
```

## What's verified vs. not

**Verified on Google Colab Tesla T4 (CUDA 12.8):**
`test_correctness_torch.py` passed against `torch.cdist` /
`F.cosine_similarity` (max abs diff well under 1e-4 across several
`n`/`dim` pairs). `benchmark_torch.py` numbers are in
`../results/benchmark_torch.csv` and summarized in the main README
Results section.

## Building and running (on a machine with an NVIDIA GPU + PyTorch + CUDA toolkit)

```bash
cd pytorch_ext
pip install -e .                      # compiles the extension
python test_correctness_torch.py      # RUN THIS FIRST
python benchmark_torch.py             # then this, for real numbers
python demo_rerank.py                 # optional: see it used end-to-end
```

If `pip install -e .` fails, the most common cause is a mismatch between
your installed PyTorch's CUDA version and your system's CUDA toolkit
version -- check with:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```
These don't need to match exactly, but should be close (e.g. both CUDA
12.x); a PyTorch built for CUDA 11.8 paired with a CUDA 12.4 toolkit is a
common source of build errors.

## Measured numbers (Colab T4)

Custom tiled L2 vs `torch.cdist` at dim=384 (device-resident):

| Dataset size | custom L2 | `torch.cdist` |
|---|---|---|
| 10,000 | 0.24 ms | 6.15 ms |
| 100,000 | 2.47 ms | 4.67 ms |
| 1,000,000 | 20.8 ms | 48.8 ms |

See the main README Results section for cosine numbers and the chart in
`../results/latency_chart_torch.png`.
