import os
import sys

import torch
import torch.nn as nn

from lib.models.triplet_topk_loader import load_triplet_topk_exact

triplet_topk_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V22: repeated V18 blocks with one shared CUDA motion-pair top-k lookup."""

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
        repeat=1,
        **kwargs,
    ):
        super().__init__()
        global triplet_topk_exact
        triplet_topk_exact = load_triplet_topk_exact(topk_relu)
        del kernel_size, pos_scale, chunk_size, use_cuda_kernel, allow_python_fallback

        self.ffn_position = str(kwargs.pop("ffn_position", "before_mean")).lower()
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)

        self.in_channels = in_channels
        self.alpha = alpha
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        self.pos_scale = window_size // 2
        self.valid_norm = bool(valid_norm)
        self.use_pos_gate = True
        self.repeat = int(repeat)
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))

        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.pos_mlp = self._make_pos_mlp(pos_hidden)
        self.ffn = self._make_ffn(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU(inplace=True)

        self.repeat_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "prev_proj": nn.Linear(in_channels, in_channels, bias=False),
                        "cur_proj": nn.Linear(in_channels, in_channels, bias=False),
                        "next_proj": nn.Linear(in_channels, in_channels, bias=False),
                        "pos_mlp": self._make_pos_mlp(pos_hidden),
                        "ffn": self._make_ffn(in_channels, hidden_channels),
                        "norm": nn.LayerNorm(in_channels),
                    }
                )
                for _ in range(max(0, self.repeat - 1))
            ]
        )

        self._init_pos_mlp(self.pos_mlp)
        for block in self.repeat_blocks:
            self._init_pos_mlp(block["pos_mlp"])

        coords = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            dtype=torch.int32,
        )
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1).contiguous()
        self.register_buffer("window_offsets", offsets, persistent=False)

        n_offsets = offsets.shape[0]
        op = offsets.to(torch.long).view(n_offsets, 1, 2)
        on = offsets.to(torch.long).view(1, n_offsets, 2)
        pair_cost = (op + on).square().sum(dim=-1).reshape(-1)

        pair_order = torch.argsort(pair_cost, stable=True).to(torch.int32)
        pair_prev_offset_id = torch.div(pair_order, n_offsets, rounding_mode="floor").to(torch.int32)
        pair_next_offset_id = (pair_order - pair_prev_offset_id * n_offsets).to(torch.int32)

        self.register_buffer("pair_prev_offset_id", pair_prev_offset_id, persistent=False)
        self.register_buffer("pair_next_offset_id", pair_next_offset_id, persistent=False)

    @staticmethod
    def _make_pos_mlp(pos_hidden):
        return nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, 1),
        )

    @staticmethod
    def _make_ffn(in_channels, hidden_channels):
        return nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )

    @staticmethod
    def _init_pos_mlp(pos_mlp):
        nn.init.zeros_(pos_mlp[-1].weight)
        nn.init.zeros_(pos_mlp[-1].bias)

    @staticmethod
    def _infer_dense_shape(x_conv, indices):
        batch_size = getattr(x_conv, "batch_size", None)
        spatial_shape = getattr(x_conv, "spatial_shape", None)

        if batch_size is not None and spatial_shape is not None and len(spatial_shape) >= 3:
            B = int(batch_size)
            T = int(spatial_shape[0])
            H = int(spatial_shape[1])
            W = int(spatial_shape[2])
            return B, T, H, W

        coords = indices.long()
        B = int(coords[:, 0].max().item()) + 1
        T = int(coords[:, 1].max().item()) + 1
        H = int(coords[:, 2].max().item()) + 1
        W = int(coords[:, 3].max().item()) + 1
        return B, T, H, W

    @staticmethod
    def _build_index_map(indices, B, T, H, W):
        device = indices.device
        n_points = indices.shape[0]
        index_map = torch.full((B, T, H, W), -1, device=device, dtype=torch.int32)
        if n_points == 0:
            return index_map

        coords = indices.long()
        row_ids = torch.arange(n_points, device=device, dtype=torch.int32)
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
                index_map = index_map.to(device=indices.device)
            if index_map.dtype != torch.int32:
                index_map = index_map.to(dtype=torch.int32)
            return index_map.contiguous()

        B, T, H, W = self._infer_dense_shape(x_conv, indices)
        return self._build_index_map(indices, B, T, H, W)

    def _find_triplet_neighbors(self, indices, x_conv=None, index_map=None):
        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        return triplet_topk_exact(
            indices=indices,
            index_map=dense_index_map,
            pair_prev_offset_id=self.pair_prev_offset_id.to(device=indices.device),
            pair_next_offset_id=self.pair_next_offset_id.to(device=indices.device),
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=self.temporal_dilation,
            topk=self.topk,
        )

    def _block_layers(self, repeat_idx):
        if repeat_idx == 0:
            return {
                "prev_proj": self.prev_proj,
                "cur_proj": self.cur_proj,
                "next_proj": self.next_proj,
                "pos_mlp": self.pos_mlp,
                "ffn": self.ffn,
                "norm": self.norm,
            }
        return self.repeat_blocks[repeat_idx - 1]

    def _apply_v18_block(
        self,
        features,
        residual_features,
        indices,
        valid_idx,
        idx_prev_v,
        idx_next_v,
        mask_pair_v,
        motion_input,
        layers,
    ):
        n_valid = valid_idx.shape[0]
        channels = features.shape[1]
        k = self.topk

        prev_feat = layers["prev_proj"](features[idx_prev_v.reshape(-1)]).view(n_valid, k, channels)
        cur_feat = layers["cur_proj"](features[valid_idx]).view(n_valid, 1, channels)
        next_feat = layers["next_proj"](features[idx_next_v.reshape(-1)]).view(n_valid, k, channels)

        triplet_feat = prev_feat + cur_feat + next_feat
        mask_pair_f = mask_pair_v.unsqueeze(-1).to(triplet_feat.dtype)

        pos_score = layers["pos_mlp"](motion_input.reshape(-1, 6))
        pos_score = pos_score.view(n_valid, k, 1)
        pos_gate = 2.0 * torch.sigmoid(pos_score)
        pos_gate = pos_gate * mask_pair_f
        triplet_input = triplet_feat * pos_gate
        score_sum = pos_gate.sum(dim=1)

        if self.valid_norm:
            denom = mask_pair_v.sum(dim=1)
            denom = denom.to(features.dtype).clamp_min(1.0).unsqueeze(1)
        else:
            denom = features.new_full((n_valid, 1), float(k))

        if self.ffn_position == "before_mean":
            triplet_msg = layers["ffn"](triplet_input.reshape(-1, channels))
            triplet_msg = triplet_msg.view(n_valid, k, channels)
            triplet_msg = triplet_msg * mask_pair_f.to(triplet_msg.dtype)
            update_v = triplet_msg.sum(dim=1) / denom
        else:
            triplet_input = triplet_input * mask_pair_f.to(triplet_input.dtype)
            update_v = layers["ffn"](triplet_input.sum(dim=1) / denom)
        update_v = self.act(layers["norm"](update_v))

        update = features.new_zeros(features.shape[0], channels)
        update[valid_idx] = update_v

        score_v = score_sum / denom.to(score_sum.dtype)
        score = features.new_zeros(features.shape[0], 1)
        score[valid_idx] = score_v

        return residual_features + update, score

    def forward(self, x, index_map=None):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features

        idx_prev, idx_next, mask_pair = self._find_triplet_neighbors(
            indices,
            x_conv=x_conv,
            index_map=index_map,
        )

        has_pair = mask_pair.any(dim=1)
        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)

        if valid_idx.numel() == 0:
            return input_features, features.new_zeros((features.shape[0], 1))

        idx_prev_v = idx_prev[valid_idx]
        idx_next_v = idx_next[valid_idx]
        mask_pair_v = mask_pair[valid_idx]

        spatial = indices[:, 2:4].to(features.dtype)
        s_cur = spatial[valid_idx].view(valid_idx.shape[0], 1, 2)
        s_prev = spatial[idx_prev_v]
        s_next = spatial[idx_next_v]
        s_minus = s_cur - s_prev
        s_plus = s_next - s_cur
        accel = s_plus - s_minus
        motion_input = torch.cat([s_minus, s_plus, accel], dim=-1) / max(self.pos_scale, 1e-6)

        residual_features = input_features
        score = features.new_zeros((features.shape[0], 1))
        for repeat_idx in range(self.repeat):
            features, score = self._apply_v18_block(
                features,
                residual_features,
                indices,
                valid_idx,
                idx_prev_v,
                idx_next_v,
                mask_pair_v,
                motion_input,
                self._block_layers(repeat_idx),
            )
            residual_features = features

        return features, score


TripletMotionConsistencyMotionPairSparseConvV22 = TripletMotionConsistencyMotionPairSparseConv
