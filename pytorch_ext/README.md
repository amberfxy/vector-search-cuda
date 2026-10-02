# PyTorch CUDA Extension

Device-resident distance ops for PyTorch (`float32` CUDA tensors).

See [`docs/pytorch-integration.md`](../docs/pytorch-integration.md) for the full API.

```bash
cd pytorch_ext
pip install -e .
python test_correctness_torch.py
python benchmark_torch.py
```

Exports: tiled L2/cosine, warp L2, batched L2 (tiled/warp). Historical T4
numbers vs `torch.cdist` remain in the main README; re-measure on your GPU.
