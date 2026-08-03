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

    # --- L2 ---
    custom_l2 = vst.l2_distance(query, store)
    reference_l2 = torch.cdist(query.unsqueeze(0), store).squeeze(0)
    max_diff_l2 = (custom_l2 - reference_l2).abs().max().item()

    # --- Cosine ---
    custom_cos = vst.cosine_similarity(query, store)
    reference_cos = F.cosine_similarity(query.unsqueeze(0), store)
    max_diff_cos = (custom_cos - reference_cos).abs().max().item()

    ok_l2 = max_diff_l2 < TOLERANCE
    ok_cos = max_diff_cos < TOLERANCE

    print(f"n={n:>8} dim={dim:<4}  L2 max diff={max_diff_l2:.6f} [{'OK' if ok_l2 else 'FAIL'}]"
          f"   cosine max diff={max_diff_cos:.6f} [{'OK' if ok_cos else 'FAIL'}]")
    return ok_l2 and ok_cos


def main():
    if not torch.cuda.is_available():
        print("No CUDA device available -- this test requires a GPU. Skipping.")
        sys.exit(0)

    # Same boundary-case reasoning as the C++ GPU test: sizes smaller than
    # one thread block, sizes that aren't clean multiples of the block size.
    cases = [(100, 384), (257, 384), (10_000, 384), (100_000, 768)]

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
