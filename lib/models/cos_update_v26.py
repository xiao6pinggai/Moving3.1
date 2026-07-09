import torch
import torch.nn as nn

from lib.models.feature_topk_v25_loader import load_feature_topk_v25_exact

feature_topk_v25_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V26 neighbor self-attention followed by center cross-attention.

    V26 reuses the V25 CUDA top-k kernel to select K previous-frame neighbors,
    one current self slot plus K-1 current-frame neighbors, and K next-frame
    neighbors. The 3K neighbor tokens first run pre-norm self-attention with a
    discrete relative-position bias table. The center point then cross-attends
    to the self-attended neighbor tokens and updates its feature with a standard
    residual path.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        alpha=0.5,
        temporal_dilation=1,
        tpdilation=None,
        conv=None,
        topk=3,
        window_size=11,
        hidden_ratio=4,
        pos_hidden=16,
        pos_scale=16.0,
        **kwargs,
    ):
        super().__init__()
        self.useFFN = bool(kwargs.pop("useFFN", True))
        kwargs.pop("ffn_position", None)
        kwargs.pop("topk_relu", None)
        kwargs.pop("k_laiyuan", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv26 kwargs: {sorted(kwargs)}")

        del kernel_size, alpha, pos_hidden, pos_scale
        self.in_channels = int(in_channels)
        self.temporal_dilation = self._validate_single_dilation(tpdilation, temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv26 requires tmc_topk > 0")

        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        self.window_extent = 2 * self.window_radius + 1
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = max(4, int(round(self.in_channels * float(hidden_ratio))))
        self.norm_topk = nn.LayerNorm(self.in_channels)
        self.norm_neighbor_sa = nn.LayerNorm(self.in_channels)
        self.norm_cross_q = nn.LayerNorm(self.in_channels)
        self.norm_cross_kv = nn.LayerNorm(self.in_channels)
        self.norm_ffn = nn.LayerNorm(self.in_channels) if self.useFFN else None

        self.neighbor_proj = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.sa_qkv_proj = nn.Linear(self.in_channels, 3 * self.in_channels, bias=False)
        self.sa_out_proj = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.cross_q_proj = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.cross_kv_proj = nn.Linear(self.in_channels, 2 * self.in_channels, bias=False)
        self.cross_out_proj = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.ffn = (
            nn.Sequential(
                nn.Linear(self.in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, self.in_channels),
            )
            if self.useFFN
            else None
        )
        self.scale = self.in_channels ** -0.5

        coords = torch.arange(-self.window_radius, self.window_radius + 1, dtype=torch.int32)
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1).contiguous()
        self.register_buffer("window_offsets", offsets, persistent=False)

        self.rel_t_radius = 2 * self.temporal_dilation
        self.rel_hw_radius = 2 * self.window_radius
        self.rel_t_size = 2 * self.rel_t_radius + 1
        self.rel_hw_size = 2 * self.rel_hw_radius + 1
        rel_state_count = self.rel_t_size * self.rel_hw_size * self.rel_hw_size
        self.relative_position_bias = nn.Embedding(rel_state_count, 1)
        nn.init.zeros_(self.relative_position_bias.weight)

    @staticmethod
    def _validate_single_dilation(tpdilation, temporal_dilation):
        if tpdilation is None:
            values = [int(temporal_dilation)]
        elif isinstance(tpdilation, str):
            text = tpdilation.strip()
            if text.startswith("[") or text.startswith("("):
                values = [int(v) for v in text.strip("[]()").replace(",", " ").split()]
            elif "," in text:
                values = [int(v.strip()) for v in text.split(",") if v.strip()]
            else:
                values = [int(v.strip()) for v in text.split() if v.strip()]
        elif isinstance(tpdilation, (list, tuple)):
            values = [int(v) for v in tpdilation]
        else:
            values = [int(tpdilation)]
        if values != [1]:
            raise ValueError(f"cosv26 is specialized for tpdilation=[1], got {values}")
        return 1

    @staticmethod
    def _infer_dense_shape(x_conv, indices):
        batch_size = getattr(x_conv, "batch_size", None)
        spatial_shape = getattr(x_conv, "spatial_shape", None)
        if batch_size is not None and spatial_shape is not None and len(spatial_shape) >= 3:
            return int(batch_size), int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2])
        coords = indices.long()
        return (
            int(coords[:, 0].max().item()) + 1,
            int(coords[:, 1].max().item()) + 1,
            int(coords[:, 2].max().item()) + 1,
            int(coords[:, 3].max().item()) + 1,
        )

    @staticmethod
    def _build_index_map(indices, B, T, H, W):
        index_map = torch.full((B, T, H, W), -1, device=indices.device, dtype=torch.int32)
        if indices.numel() == 0:
            return index_map
        coords = indices.long()
        row_ids = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32)
        index_map[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = row_ids
        return index_map.contiguous()

    @staticmethod
    def _extract_external_index_map(x_conv):
        for name in ("index_map", "sparse_index_map", "dense_index_map"):
            value = getattr(x_conv, name, None)
            if value is not None:
                return value
        return None

    def _get_index_map(self, x_conv, indices, index_map=None):
        if index_map is None:
            index_map = self._extract_external_index_map(x_conv)
        if index_map is not None:
            if not index_map.is_cuda:
                raise RuntimeError("cosv26 requires CUDA index_map; no CPU fallback is available")
            if index_map.dtype != torch.int32:
                index_map = index_map.to(dtype=torch.int32)
            return index_map.contiguous()
        B, T, H, W = self._infer_dense_shape(x_conv, indices)
        return self._build_index_map(indices, B, T, H, W)

    @staticmethod
    def _check_conv_contract(input_features, x_conv):
        if x_conv.features.shape != input_features.shape:
            raise RuntimeError(
                "cosv26 requires conv to preserve sparse point count, order, and channels; "
                f"got input features {tuple(input_features.shape)} and conv features {tuple(x_conv.features.shape)}"
            )

    def _load_cuda_topk(self):
        global feature_topk_v25_exact
        if feature_topk_v25_exact is None:
            feature_topk_v25_exact = load_feature_topk_v25_exact()
        return feature_topk_v25_exact

    def _select_topk_neighbors(self, indices, features_norm, index_map):
        if not indices.is_cuda or not features_norm.is_cuda:
            raise RuntimeError("cosv26 Top-K requires CUDA indices and features; no Python fallback is available")
        return self._load_cuda_topk()(
            indices=indices,
            index_map=index_map,
            features=features_norm,
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=1,
            topk=self.topk,
        )

    def _build_neighbor_coords(self, n_points, top_pos, mask_nei, device):
        k = self.topk
        slots = 3 * k
        dtype = torch.long
        rel_t = torch.cat(
            [
                torch.full((n_points, k, 1), -self.temporal_dilation, device=device, dtype=dtype),
                torch.zeros((n_points, k, 1), device=device, dtype=dtype),
                torch.full((n_points, k, 1), self.temporal_dilation, device=device, dtype=dtype),
            ],
            dim=1,
        )
        rel_yx = self.window_offsets.to(device=device, dtype=dtype)[top_pos.reshape(-1)].view(n_points, slots, 2)
        coords = torch.cat([rel_t, rel_yx], dim=-1)
        return coords.masked_fill(~mask_nei.unsqueeze(-1), 0)

    def _relative_bias(self, query_coords, key_coords):
        delta = key_coords.unsqueeze(-3) - query_coords.unsqueeze(-2)
        rel_t = delta[..., 0] + self.rel_t_radius
        rel_y = delta[..., 1] + self.rel_hw_radius
        rel_x = delta[..., 2] + self.rel_hw_radius
        rel_index = (rel_t * self.rel_hw_size + rel_y) * self.rel_hw_size + rel_x
        return self.relative_position_bias(rel_index).squeeze(-1)

    @staticmethod
    def _masked_softmax(scores, key_mask):
        attn = scores.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(attn, dim=-1)
        attn = attn * key_mask.to(attn.dtype)
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _neighbor_self_attention(self, neighbor_tokens, neighbor_coords, mask_nei):
        token_mask = mask_nei.unsqueeze(-1)
        neighbor_tokens = neighbor_tokens.masked_fill(~token_mask, 0)

        sa_input = self.norm_neighbor_sa(neighbor_tokens)
        q, k, v = self.sa_qkv_proj(sa_input).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        scores = scores + self._relative_bias(neighbor_coords, neighbor_coords)
        attn = self._masked_softmax(scores, mask_nei.unsqueeze(1))
        sa_update = torch.matmul(attn, v)
        sa_update = self.sa_out_proj(sa_update).masked_fill(~token_mask, 0)
        return (neighbor_tokens + sa_update).masked_fill(~token_mask, 0)

    def _center_cross_attention(self, center_features, neighbor_tokens, center_coords, neighbor_coords, mask_nei):
        q = self.cross_q_proj(self.norm_cross_q(center_features)).unsqueeze(1)
        k, v = self.cross_kv_proj(self.norm_cross_kv(neighbor_tokens)).chunk(2, dim=-1)
        scores = (q * k).sum(dim=-1) * self.scale
        scores = scores + self._relative_bias(center_coords, neighbor_coords).squeeze(1)
        attn = self._masked_softmax(scores, mask_nei)
        context = (attn.unsqueeze(-1) * v).sum(dim=1)
        return self.cross_out_proj(context), attn

    def forward(self, x, index_map=None):
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))
        if not input_features.is_cuda:
            raise RuntimeError("cosv26 requires CUDA sparse features; no CPU fallback is available")

        x_conv = self.conv(x)
        self._check_conv_contract(input_features, x_conv)
        indices = x_conv.indices
        features = x_conv.features
        if not indices.is_cuda or not features.is_cuda:
            raise RuntimeError("cosv26 requires CUDA indices and features; no Python fallback is available")
        if indices.shape[0] != features.shape[0] or features.shape[1] != self.in_channels:
            raise RuntimeError("cosv26 received inconsistent sparse indices/features shape")

        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        features_norm = self.norm_topk(features)

        (
            idx_prev,
            idx_cur,
            idx_next,
            mask_prev,
            mask_cur,
            mask_next,
            _dist_prev,
            _dist_cur,
            _dist_next,
            pos_prev,
            pos_cur,
            pos_next,
        ) = self._select_topk_neighbors(indices, features_norm, dense_index_map)

        idx_nei = torch.cat([idx_prev, idx_cur, idx_next], dim=1)
        mask_nei = torch.cat([mask_prev, mask_cur, mask_next], dim=1)
        top_pos = torch.cat([pos_prev, pos_cur, pos_next], dim=1).clamp_min(0)

        neighbor_base = self.neighbor_proj(features_norm)[idx_nei]
        neighbor_coords = self._build_neighbor_coords(features.shape[0], top_pos, mask_nei, features.device)
        neighbor_tokens = self._neighbor_self_attention(neighbor_base, neighbor_coords, mask_nei)

        center_coords = neighbor_coords.new_zeros((features.shape[0], 1, 3))
        cross_update, cross_attn = self._center_cross_attention(
            features,
            neighbor_tokens,
            center_coords,
            neighbor_coords,
            mask_nei,
        )
        x_attn = features + cross_update
        output_features = x_attn + self.ffn(self.norm_ffn(x_attn)) if self.useFFN else x_attn
        score = cross_attn.amax(dim=1, keepdim=True)
        return output_features, score


TripletMotionConsistencyMotionPairSparseConvV26 = TripletMotionConsistencyMotionPairSparseConv
