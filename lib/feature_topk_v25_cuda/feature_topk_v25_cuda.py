import torch

_REQUIRED_API_VERSION = 2

try:
    import feature_topk_v25_cuda_ext
except ImportError as exc:  # pragma: no cover
    feature_topk_v25_cuda_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _as_i32_cuda_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        x = x.to(dtype=torch.int32)
    return x.contiguous()


def _check_extension():
    if feature_topk_v25_cuda_ext is None:
        raise ImportError(
            "feature_topk_v25_cuda_ext is required but is not compiled/importable. "
            "Compile it with: cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
        ) from _IMPORT_ERROR
    version = getattr(feature_topk_v25_cuda_ext, "api_version", None)
    if version is None or int(version()) != _REQUIRED_API_VERSION:
        got = "missing" if version is None else int(version())
        raise RuntimeError(
            f"feature_topk_v25_cuda_ext API version mismatch: expected {_REQUIRED_API_VERSION}, got {got}. "
            "Recompile with: cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
        )


def feature_topk_v25_exact(
    indices: torch.Tensor,
    index_map: torch.Tensor,
    features: torch.Tensor,
    window_offsets: torch.Tensor,
    temporal_dilation: int,
    topk: int,
):
    """Feature-distance top-k CUDA wrapper for cosv25 API v2.

    Returns:
        idx_prev, idx_cur, idx_next: [N, K], torch.long
        mask_prev, mask_cur, mask_next: [N, K], torch.bool
        dist_prev, dist_cur, dist_next: [N, K], torch.float32
        pos_prev, pos_cur, pos_next: [N, K], torch.long offset positions
    """
    _check_extension()
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if int(temporal_dilation) != 1:
        raise ValueError("feature_topk_v25_exact is specialized for temporal_dilation=1")
    if not indices.is_cuda or not features.is_cuda:
        raise RuntimeError("feature_topk_v25_exact requires CUDA indices and features")
    if not features.is_floating_point():
        raise ValueError("features must be a floating point tensor")

    out = feature_topk_v25_cuda_ext.forward(
        _as_i32_cuda_contiguous(indices, "indices"),
        _as_i32_cuda_contiguous(index_map, "index_map"),
        features.contiguous(),
        _as_i32_cuda_contiguous(window_offsets, "window_offsets"),
        1,
        int(topk),
    )
    if len(out) != 12:
        raise RuntimeError(
            f"feature_topk_v25_cuda_ext returned {len(out)} tensors, expected 12. "
            "Recompile with: cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
        )
    return out
