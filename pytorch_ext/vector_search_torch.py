"""
Thin, user-friendly wrapper around the compiled `vector_search_torch_cpp`
extension. Import this module rather than the raw compiled extension
directly -- it adds a couple of convenience behaviors (accepting either a
1D or [1, dim] query tensor) without touching the C++/CUDA side.
"""
import torch

try:
    import vector_search_torch_cpp as _cpp
except ImportError as e:
    raise ImportError(
        "vector_search_torch_cpp is not built yet. Run 'pip install -e .' "
        "in the pytorch_ext/ directory on a machine with a CUDA-capable "
        "GPU and PyTorch installed."
    ) from e


def _flatten_query(query: torch.Tensor) -> torch.Tensor:
    if query.dim() == 2 and query.size(0) == 1:
        return query.squeeze(0)
    return query


def l2_distance(query: torch.Tensor, store: torch.Tensor) -> torch.Tensor:
    """L2 distance from `query` ([dim] or [1, dim]) to every row of `store`
    ([N, dim]). Returns a [N] tensor; smaller = closer. Both tensors must
    already be float32 CUDA tensors."""
    return _cpp.tiled_l2_distance(_flatten_query(query).contiguous(), store.contiguous())


def cosine_similarity(query: torch.Tensor, store: torch.Tensor) -> torch.Tensor:
    """Cosine similarity from `query` ([dim] or [1, dim]) to every row of
    `store` ([N, dim]). Returns a [N] tensor; larger = more similar."""
    return _cpp.tiled_cosine_similarity(_flatten_query(query).contiguous(), store.contiguous())


def rerank(query: torch.Tensor, store: torch.Tensor, k: int, metric: str = "cosine"):
    """Convenience function tying the custom kernel to an actual retrieval
    step: returns the top-k (indices, scores) from `store` for `query`.

    This is the piece that demonstrates the kernel plugged into something
    resembling a real usage pattern (query a corpus, get back ranked
    candidates) rather than just a microbenchmark of the distance
    computation in isolation.
    """
    if metric == "cosine":
        scores = cosine_similarity(query, store)
        top_scores, top_indices = torch.topk(scores, k, largest=True)
    elif metric == "l2":
        scores = l2_distance(query, store)
        top_scores, top_indices = torch.topk(scores, k, largest=False)
    else:
        raise ValueError(f"Unknown metric '{metric}', expected 'cosine' or 'l2'")
    return top_indices, top_scores
