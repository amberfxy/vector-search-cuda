"""
Correctness check: compares the custom CUDA extension's output against
PyTorch's own built-in operators (torch.cdist for L2, F.cosine_similarity
for cosine), which are the natural "known-correct" reference on the
PyTorch side -- much like tests/test_correctness_gpu.cpp in the main
project uses the CPU implementation as ground truth for the standalone
CUDA kernels.

Run with: python test_correctness_torch.py
Requires a CUDA-capable GPU with the extension built (`pip install -e .`
in this directory first).
"""
import sys
import torch
import torch.nn.functional as F

import vector_search_torch as vst

TOLERANCE = 1e-3  # slightly looser than the C++ tests (1e-4) since
                   # torch.cdist/cosine_similarity may use a different
                   # internal reduction order than our kernel


def check_case(n, dim, seed):
    torch.manual_seed(seed)
    query = torch.randn(dim, device="cuda", dtype=torch.float32)
    store = torch.randn(n, dim, device="cuda", dtype=torch.float32)
    queries = torch.randn(8, dim, device="cuda", dtype=torch.float32)

    custom_l2 = vst.l2_distance(query, store, method="tiled")
    warp_l2 = vst.l2_distance(query, store, method="warp")
    reference_l2 = torch.cdist(query.unsqueeze(0), store).squeeze(0)
    max_diff_l2 = (custom_l2 - reference_l2).abs().max().item()
    max_diff_warp = (warp_l2 - reference_l2).abs().max().item()

    custom_cos = vst.cosine_similarity(query, store)
    reference_cos = F.cosine_similarity(query.unsqueeze(0), store)
    max_diff_cos = (custom_cos - reference_cos).abs().max().item()

    batch = vst.l2_distance_batch(queries, store, method="tiled")
    ref_batch = torch.cdist(queries, store)
    max_diff_batch = (batch - ref_batch).abs().max().item()

    ok = all(d < TOLERANCE for d in (max_diff_l2, max_diff_warp, max_diff_cos, max_diff_batch))
    print(
        f"n={n:>8} dim={dim:<4}  tiledL2={max_diff_l2:.6f} warpL2={max_diff_warp:.6f} "
        f"cos={max_diff_cos:.6f} batch={max_diff_batch:.6f} [{'OK' if ok else 'FAIL'}]"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available -- this test requires a GPU. Skipping.")
        sys.exit(0)

    cases = [(100, 384), (257, 384), (383, 385), (10_000, 384), (100_000, 768)]

    all_ok = True
    for n, dim in cases:
        all_ok &= check_case(n, dim, seed=42)

    if all_ok:
        print("\nAll PyTorch extension correctness checks passed.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED -- do not trust benchmark_torch.py numbers until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
