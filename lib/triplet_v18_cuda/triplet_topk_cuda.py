import torch

try:
    import triplet_topk_cuda_ext
except ImportError as exc:  # pragma: no cover
    triplet_topk_cuda_ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _as_i32_cuda_contiguous(x: torch.Tensor, name: str) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.int32:
        x = x.to(dtype=torch.int32)
    return x.contiguous()


def triplet_topk_exact(
    indices: torch.Tensor,
    index_map: torch.Tensor,
    pair_prev_offset_id: torch.Tensor,
    pair_next_offset_id: torch.Tensor,
    window_offsets: torch.Tensor,
    temporal_dilation: int,
    topk: int,
):
    """Exact motion-consistency pair top-k CUDA wrapper.

    Args:
        indices: [N, 4], CUDA, b/t/y/x. Converted to int32 internally.
        index_map: [B, T, H, W], CUDA, int32 preferred, invalid = -1.
        pair_prev_offset_id: [P], CUDA/CPU tensor accepted if caller moves to CUDA first.
        pair_next_offset_id: [P].
        window_offsets: [O, 2].
        temporal_dilation: positive int.
        topk: positive int, arbitrary value supported. Large topk may use slower serial fallback.

    Returns:
        idx_prev:  [N, topk], torch.long
        idx_next:  [N, topk], torch.long
        mask_pair: [N, topk], torch.bool
    """
    if triplet_topk_cuda_ext is None:
        raise ImportError(
            "triplet_topk_cuda_ext is not compiled/importable. "
            "Run: python setup.py build_ext --inplace"
        ) from _IMPORT_ERROR

    if topk <= 0:
        raise ValueError("topk must be > 0")
    if temporal_dilation <= 0:
        raise ValueError("temporal_dilation must be > 0")

    indices_i32 = _as_i32_cuda_contiguous(indices, "indices")
    index_map_i32 = _as_i32_cuda_contiguous(index_map, "index_map")
    pair_prev_i32 = _as_i32_cuda_contiguous(pair_prev_offset_id, "pair_prev_offset_id")
    pair_next_i32 = _as_i32_cuda_contiguous(pair_next_offset_id, "pair_next_offset_id")
    offsets_i32 = _as_i32_cuda_contiguous(window_offsets, "window_offsets")

    return triplet_topk_cuda_ext.forward(
        indices_i32,
        index_map_i32,
        pair_prev_i32,
        pair_next_i32,
        offsets_i32,
        int(temporal_dilation),
        int(topk),
    )
