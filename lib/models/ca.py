import torch
import torch.nn as nn

from lib.models.cos_update_v23 import TripletMotionConsistencyMotionPairSparseConv as _CosV23Base


class CrossFrameAttentionSparseConv(_CosV23Base):
    """CA: cosv23-compatible sparse cross-frame attention ablation.

    The sparse input, local search window, and feature-distance Top-K candidates
    are identical to cosv23. The attention body is replaced by standard
    cross-attention from each current point to its prev/next frame candidates.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        alpha=0.5,
        temporal_dilation=1,
        conv=None,
        topk=3,
        window_size=11,
        hidden_ratio=4,
        pos_hidden=16,
        pos_scale=16.0,
        chunk_size=1024,
        valid_norm=True,
        use_cuda_kernel=True,
        allow_python_fallback=False,
        topk_relu="fanghui",
        **kwargs,
    ):
        nn.Module.__init__(self)
        del kernel_size, alpha, pos_hidden, pos_scale, valid_norm, topk_relu

        kwargs.pop("ffn_position", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected CA kwargs: {sorted(kwargs)}")

        self.in_channels = int(in_channels)
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("CA requires tmc_topk > 0")

        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        self.chunk_size = int(chunk_size)
        self.conv = conv if conv is not None else nn.Identity()
        self.use_cuda_kernel = bool(use_cuda_kernel)
        self.allow_python_fallback = bool(allow_python_fallback)

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))

        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)

        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )

        self.scale = in_channels ** -0.5

        coords = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            dtype=torch.int32,
        )
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1).contiguous()
        offset_cost = offsets.to(torch.long).square().sum(dim=-1)
        offset_order = torch.argsort(offset_cost, stable=True).to(torch.long)
        offsets = offsets[offset_order].contiguous()
        self.register_buffer("window_offsets", offsets, persistent=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        nn.Module._load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, index_map=None):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features
        n_points, channels = features.shape
        k = self.topk

        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        features_norm = self.norm_attn(features)

        idx_prev, idx_next, mask_prev, mask_next = self._select_topk_by_feature_distance(
            indices,
            features_norm,
            dense_index_map,
        )

        has_neighbor = mask_prev.any(dim=1) | mask_next.any(dim=1)
        valid_idx = has_neighbor.nonzero(as_tuple=False).squeeze(1)

        if valid_idx.numel() == 0:
            return input_features, features.new_zeros((n_points, 1))

        update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)

        attn_chunk = self._adaptive_chunk_size(valid_idx.numel(), 2 * k, channels, pair_factor=1)
        for start in range(0, valid_idx.numel(), attn_chunk):
            end = min(start + attn_chunk, valid_idx.numel())
            rows = valid_idx[start:end]
            n_valid = rows.shape[0]

            idx_prev_v = idx_prev[rows]
            idx_next_v = idx_next[rows]
            mask_prev_v = mask_prev[rows]
            mask_next_v = mask_next[rows]

            idx_all = torch.cat([idx_prev_v, idx_next_v], dim=1)
            mask_all = torch.cat([mask_prev_v, mask_next_v], dim=1)

            f_cur_res = input_features[rows]
            f_cur = features_norm[rows]
            f_nei = features_norm[idx_all.reshape(-1)].view(n_valid, 2 * k, channels)

            q = self.q_proj(f_cur).view(n_valid, 1, channels)
            key = self.k_proj(f_nei)
            value = self.v_proj(f_nei)

            attn = (q * key).sum(dim=-1) * self.scale
            attn = attn.masked_fill(~mask_all, -1e4)
            attn = torch.softmax(attn, dim=1)
            attn = attn * mask_all.to(attn.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

            attn_update = (attn.unsqueeze(-1) * value).sum(dim=1)
            attn_update = self.out_proj(attn_update)
            x_attn = f_cur_res + attn_update
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn))

            update[rows] = out_v - f_cur_res
            score[rows] = attn.amax(dim=1, keepdim=True)

        return input_features + update, score


TripletMotionConsistencyMotionPairSparseConvCA = CrossFrameAttentionSparseConv
TripletMotionConsistencyMotionPairSparseConv = CrossFrameAttentionSparseConv
