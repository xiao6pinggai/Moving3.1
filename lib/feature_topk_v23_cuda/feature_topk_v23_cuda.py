import torch

try:
    import feature_topk_v23_cuda_ext
except ImportError as exc:  # pragma: no cover
    feature_topk_v23_cuda_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _as_i32_cuda_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        x = x.to(dtype=torch.int32)
    return x.contiguous()


def feature_topk_v23_exact(
    indices: torch.Tensor,
    index_map: torch.Tensor,
    features: torch.Tensor,
    window_offsets: torch.Tensor,
    temporal_dilation: int,
    topk: int,
):
    """Feature-distance top-k CUDA wrapper for cosv23.

    Args:
        indices: [N, 4], CUDA, int32-compatible b/t/y/x.
        index_map: [B, T, H, W], CUDA, int32 preferred, invalid = -1.
        features: [N, C], CUDA floating tensor used for distance ranking.
        window_offsets: [O, 2], CUDA, int32-compatible.
        temporal_dilation: positive int.
        topk: positive int.

    Returns:
        idx_prev, idx_next: [N, topk], torch.long
        mask_prev, mask_next: [N, topk], torch.bool
    """
    if feature_topk_v23_cuda_ext is None:
        raise ImportError(
            "feature_topk_v23_cuda_ext is not compiled/importable. "
            "Run: cd lib/feature_topk_v23_cuda && python setup.py build_ext --inplace"
        ) from _IMPORT_ERROR

    if topk <= 0:
        raise ValueError("topk must be > 0")
    if temporal_dilation <= 0:
        raise ValueError("temporal_dilation must be > 0")
    if not features.is_cuda:
        raise ValueError("features must be a CUDA tensor")
    if not features.is_floating_point():
        raise ValueError("features must be a floating point tensor")

    return feature_topk_v23_cuda_ext.forward(
        _as_i32_cuda_contiguous(indices, "indices"),
        _as_i32_cuda_contiguous(index_map, "index_map"),
        features.contiguous(),
        _as_i32_cuda_contiguous(window_offsets, "window_offsets"),
        int(temporal_dilation),
        int(topk),
    )
