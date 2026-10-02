# PyTorch integration

## Goal

Expose device-resident distance ops as `torch.Tensor` → `torch.Tensor`
without host round-trips on the timed path.

## Build

```bash
cd pytorch_ext
pip install -e .
python test_correctness_torch.py
```

Requires NVIDIA GPU, CUDA toolkit, and PyTorch with a compatible CUDA build
(`python -c "import torch; print(torch.version.cuda)"`).

## Python API

```python
import torch
import vector_search_torch as vst

store = torch.randn(100_000, 384, device="cuda")
query = torch.randn(384, device="cuda")
queries = torch.randn(32, 384, device="cuda")

d = vst.l2_distance(query, store, method="tiled")   # or "warp"
c = vst.cosine_similarity(query, store)
B = vst.l2_distance_batch(queries, store, method="warp")  # [32, 100000]
idx, scores = vst.rerank(query, store, k=10, metric="l2", method="warp")
```

## Correctness

`test_correctness_torch.py` compares against `torch.cdist` /
`F.cosine_similarity` with absolute tolerance **1e-3** (slightly looser than
C++ 1e-4 due to PyTorch reduction differences).

## Measured (historical Colab T4, tiled L2 vs `torch.cdist`)

| N | custom L2 | torch.cdist |
|---|---|---|
| 10K | 0.24 ms | 6.15 ms |
| 100K | 2.47 ms | 4.67 ms |
| 1M | 20.8 ms | 48.8 ms |

Re-run `benchmark_torch.py` on your GPU; do not treat these as portable SLOs.

## Relationship to `GpuVectorIndex`

| | C++ `GpuVectorIndex` | PyTorch ext |
|---|---|---|
| Corpus upload | Explicit `build()` H2D | User keeps tensor on CUDA |
| Hot path | query H2D + kernel + score D2H | kernel only (tensor already on device) |
| Batch | `search_batch` | `l2_distance_batch` |
