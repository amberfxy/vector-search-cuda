"""
Thin Python API over the compiled CUDA extension.

Tensors must already be float32 CUDA and contiguous (wrappers call
.contiguous()). Avoids host round-trips: store/query stay on device.
"""
import torch

try:
    import vector_search_torch_cpp as _cpp
except ImportError as e:
    raise ImportError(
        "vector_search_torch_cpp is not built yet. Run 'pip install -e .' "
        "in pytorch_ext/ on a CUDA + matching PyTorch machine."
    ) from e


def _as_query_1d(query: torch.Tensor) -> torch.Tensor:
    if query.dim() == 2 and query.size(0) == 1:
        return query.squeeze(0)
    return query


def l2_distance(query: torch.Tensor, store: torch.Tensor, method: str = "tiled") -> torch.Tensor:
    """L2 distance query[dim] vs store[N,dim] -> [N]. method: tiled|warp."""
    q = _as_query_1d(query).contiguous()
    s = store.contiguous()
    if method == "tiled":
        return _cpp.tiled_l2_distance(q, s)
    if method == "warp":
        return _cpp.warp_l2_distance(q, s)
    raise ValueError("method must be 'tiled' or 'warp'")


def cosine_similarity(query: torch.Tensor, store: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (tiled kernel). Larger = more similar."""
    return _cpp.tiled_cosine_similarity(_as_query_1d(query).contiguous(), store.contiguous())


def l2_distance_batch(queries: torch.Tensor, store: torch.Tensor,
                      method: str = "tiled") -> torch.Tensor:
    """Batched L2: queries[Q,dim] vs store[N,dim] -> [Q,N]."""
    if queries.dim() != 2:
        raise ValueError("queries must be [Q, dim]")
    q = queries.contiguous()
    s = store.contiguous()
    if method == "tiled":
        return _cpp.tiled_l2_distance_batch(q, s)
    if method == "warp":
        return _cpp.warp_l2_distance_batch(q, s)
    raise ValueError("method must be 'tiled' or 'warp'")


def rerank(query: torch.Tensor, store: torch.Tensor, k: int, metric: str = "cosine",
           method: str = "tiled"):
    """Top-k indices/scores for a single query (device-resident path)."""
    if metric == "cosine":
        scores = cosine_similarity(query, store)
        return torch.topk(scores, k, largest=True)
    if metric == "l2":
        scores = l2_distance(query, store, method=method)
        return torch.topk(scores, k, largest=False)
    raise ValueError("metric must be 'cosine' or 'l2'")
