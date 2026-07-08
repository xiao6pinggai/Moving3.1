import torch

try:
    import feature_topk_v25_cuda_ext
except ImportError as exc:  # pragma: no cover
    feature_topk_v25_cuda_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _as_i32_cuda_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        x = x.to(dtype=torch.int32)
    return x.contiguous()


def feature_topk_v25_exact(
    indices: torch.Tensor,
    index_map: torch.Tensor,
    features: torch.Tensor,
    window_offsets: torch.Tensor,
    temporal_dilation: int,
    topk: int,
):
    """Feature-distance top-k CUDA wrapper for cosv25.

    Selects top-k candidates independently from previous, current, and next
    temporal windows, returning 3 * topk candidate slots per query. The current
    query point itself is excluded from the current-frame candidates.

    Returns:
        idx_prev, idx_cur, idx_next: [N, topk], torch.long
        mask_prev, mask_cur, mask_next: [N, topk], torch.bool
    """
    if feature_topk_v25_cuda_ext is None:
        raise ImportError(
            "feature_topk_v25_cuda_ext is not compiled/importable. "
            "Run: cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
        ) from _IMPORT_ERROR

    if topk <= 0:
        raise ValueError("topk must be > 0")
    if temporal_dilation <= 0:
        raise ValueError("temporal_dilation must be > 0")
    if not features.is_cuda:
        raise ValueError("features must be a CUDA tensor")
    if not features.is_floating_point():
        raise ValueError("features must be a floating point tensor")

    return feature_topk_v25_cuda_ext.forward(
        _as_i32_cuda_contiguous(indices, "indices"),
        _as_i32_cuda_contiguous(index_map, "index_map"),
        features.contiguous(),
        _as_i32_cuda_contiguous(window_offsets, "window_offsets"),
        int(temporal_dilation),
        int(topk),
    )
