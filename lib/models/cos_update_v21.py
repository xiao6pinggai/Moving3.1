import os
import sys

import torch
import torch.nn as nn

from lib.models.triplet_topk_loader import load_triplet_topk_exact

triplet_topk_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V21: CUDA exact pair top-k with per-branch position gates.

    CUDA extension only replaces neighbor pair selection:
        indices/index_map/window_offsets -> idx_prev/idx_next/mask_pair

    Trainable layer names are mostly kept compatible with the previous module:
        prev_proj, cur_proj, next_proj, pos_mlp, ffn, norm

    Compared with V18, the position MLP outputs 3 gates for prev/cur/next.
    The gated triplet feature is:
        gate_prev * prev_proj + gate_cur * cur_proj + gate_next * next_proj

    The selected pair semantics remain the original replacement-allowed pair top-k:
        the same previous point or next point may appear in multiple selected pairs.
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
        super().__init__()
        global triplet_topk_exact
        triplet_topk_exact = load_triplet_topk_exact(topk_relu)
        del kernel_size, chunk_size

        self.ffn_position = str(kwargs.pop("ffn_position", "before_mean")).lower()
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv21 kwargs: {sorted(kwargs)}")
        if self.ffn_position not in ("before_mean", "after_mean"):
            raise ValueError(
                f"ffn_position should be \"before_mean\" or \"after_mean\", got {self.ffn_position}"
            )

        self.in_channels = in_channels
        self.alpha = alpha
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("CUDA v21 currently requires tmc_topk > 0; use the Python baseline for topk<=0 all-pair mode")
        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        # self.pos_scale = float(pos_scale)
        self.pos_scale = window_size // 2
        self.valid_norm = bool(valid_norm)
        self.use_pos_gate = True
        self.conv = conv if conv is not None else nn.Identity()
        self.use_cuda_kernel = bool(use_cuda_kernel)
        self.allow_python_fallback = bool(allow_python_fallback)

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))

        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, 3),
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )

        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

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

        # stable=True keeps the original flattened offset-pair order for equal motion cost.
        pair_order = torch.argsort(pair_cost, stable=True).to(torch.int32)
        pair_prev_offset_id = torch.div(pair_order, n_offsets, rounding_mode="floor").to(torch.int32)
        pair_next_offset_id = (pair_order - pair_prev_offset_id * n_offsets).to(torch.int32)

        self.register_buffer("pair_prev_offset_id", pair_prev_offset_id, persistent=False)
        self.register_buffer("pair_next_offset_id", pair_next_offset_id, persistent=False)

    @staticmethod
    def _infer_dense_shape(x_conv, indices):
        """Infer [B, T, H, W] for dense index_map.

        Prefer SparseConvTensor metadata when available. Fallback to max coordinate + 1.
        """
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
        """Build dense map: index_map[b,t,y,x] = sparse row id, invalid = -1."""
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
        """Optionally use a precomputed index_map attached by the data/filtering pipeline."""
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

    def _find_triplet_neighbors_python_exact(self, indices):
        """Debug-only exact fallback. Slow, but useful when CUDA extension is unavailable."""
        device = indices.device
        n_points = indices.shape[0]
        k = self.topk

        out_prev = torch.zeros(n_points, k, dtype=torch.long, device=device)
        out_next = torch.zeros(n_points, k, dtype=torch.long, device=device)
        mask_pair = torch.zeros(n_points, k, dtype=torch.bool, device=device)

        if n_points == 0:
            return out_prev, out_next, mask_pair

        coords_cpu = indices.detach().long().cpu()
        coord_to_idx = {
            (int(row[0]), int(row[1]), int(row[2]), int(row[3])): i
            for i, row in enumerate(coords_cpu.tolist())
        }

        offsets_cpu = self.window_offsets.detach().cpu().long()
        prev_ids_cpu = self.pair_prev_offset_id.detach().cpu().long()
        next_ids_cpu = self.pair_next_offset_id.detach().cpu().long()
        dt = self.temporal_dilation

        prev_out = []
        next_out = []
        mask_out = []

        for row in coords_cpu.tolist():
            b, t, y, x = map(int, row)
            p_list = []
            n_list = []
            m_list = []
            for po, no in zip(prev_ids_cpu.tolist(), next_ids_cpu.tolist()):
                op_y, op_x = offsets_cpu[po].tolist()
                on_y, on_x = offsets_cpu[no].tolist()
                prev_key = (b, t - dt, y + int(op_y), x + int(op_x))
                next_key = (b, t + dt, y + int(on_y), x + int(on_x))
                prev_idx = coord_to_idx.get(prev_key, None)
                if prev_idx is None:
                    continue
                next_idx = coord_to_idx.get(next_key, None)
                if next_idx is None:
                    continue
                p_list.append(prev_idx)
                n_list.append(next_idx)
                m_list.append(True)
                if len(p_list) >= k:
                    break
            while len(p_list) < k:
                p_list.append(0)
                n_list.append(0)
                m_list.append(False)
            prev_out.append(p_list)
            next_out.append(n_list)
            mask_out.append(m_list)

        return (
            torch.tensor(prev_out, dtype=torch.long, device=device),
            torch.tensor(next_out, dtype=torch.long, device=device),
            torch.tensor(mask_out, dtype=torch.bool, device=device),
        )

    def _find_triplet_neighbors(self, indices, features, x_conv=None, index_map=None):
        """Exact window pair top-k by motion consistency cost.

        The CUDA kernel uses pair_prev_offset_id/pair_next_offset_id, which are already sorted by:
            ||op + on||^2 = ||previous + next - 2 * current||^2
        Therefore the first K valid offset-pairs are the exact top-k valid motion-consistent pairs.
        """
        del features

        if indices.numel() == 0:
            n_points = indices.shape[0]
            k = self.topk
            return (
                torch.zeros(n_points, k, dtype=torch.long, device=indices.device),
                torch.zeros(n_points, k, dtype=torch.long, device=indices.device),
                torch.zeros(n_points, k, dtype=torch.bool, device=indices.device),
            )

        if indices.is_cuda and self.use_cuda_kernel:
            if triplet_topk_exact is None:
                if not self.allow_python_fallback:
                    raise ImportError(
                        "triplet_topk_cuda_ext is not available. Compile it with: "
                        "python setup.py build_ext --inplace"
                    )
                return self._find_triplet_neighbors_python_exact(indices)

            if x_conv is None and index_map is None:
                raise RuntimeError("x_conv or index_map is required to build/use dense index_map")

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

        if not self.allow_python_fallback:
            raise RuntimeError(
                "CUDA exact top-k kernel is disabled/unavailable and Python fallback is disabled. "
                "Set allow_python_fallback=True for debugging only."
            )
        return self._find_triplet_neighbors_python_exact(indices)

    def forward(self, x, index_map=None):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features

        n_points, channels = features.shape
        k = self.topk

        idx_prev, idx_next, mask_pair = self._find_triplet_neighbors(
            indices,
            features,
            x_conv=x_conv,
            index_map=index_map,
        )

        has_pair = mask_pair.any(dim=1)
        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)

        if valid_idx.numel() == 0:
            return input_features, features.new_zeros((n_points, 1))

        idx_prev_v = idx_prev[valid_idx]
        idx_next_v = idx_next[valid_idx]
        mask_pair_v = mask_pair[valid_idx]

        n_valid = valid_idx.shape[0]

        prev_feat = self.prev_proj(features[idx_prev_v.reshape(-1)]).view(n_valid, k, channels)
        cur_feat = self.cur_proj(features[valid_idx]).view(n_valid, 1, channels)
        next_feat = self.next_proj(features[idx_next_v.reshape(-1)]).view(n_valid, k, channels)

        mask_pair_f = mask_pair_v.unsqueeze(-1).to(features.dtype)
        if self.use_pos_gate:
            spatial = indices[:, 2:4].to(features.dtype)

            s_cur = spatial[valid_idx].view(n_valid, 1, 2)
            s_prev = spatial[idx_prev_v]
            s_next = spatial[idx_next_v]

            s_minus = s_cur - s_prev
            s_plus = s_next - s_cur
            accel = s_plus - s_minus

            motion_input = torch.cat([s_minus, s_plus, accel], dim=-1) / max(self.pos_scale, 1e-6)

            pos_score = self.pos_mlp(motion_input.reshape(-1, 6))
            pos_score = pos_score.view(n_valid, k, 3)

            pos_gate = 2.0 * torch.sigmoid(pos_score)
            pos_gate = pos_gate * mask_pair_f
            triplet_input = (
                prev_feat * pos_gate[..., 0:1]
                + cur_feat * pos_gate[..., 1:2]
                + next_feat * pos_gate[..., 2:3]
            )
            score_sum = pos_gate.mean(dim=-1, keepdim=True).sum(dim=1)
        else:
            triplet_feat = prev_feat + cur_feat + next_feat
            triplet_input = triplet_feat
            score_sum = mask_pair_f.sum(dim=1)

        if self.valid_norm:
            denom = mask_pair_v.sum(dim=1)
            denom = denom.to(features.dtype).clamp_min(1.0).unsqueeze(1)
        else:
            denom = features.new_full((n_valid, 1), float(k))

        if self.ffn_position == "before_mean":
            triplet_msg = self.ffn(triplet_input.reshape(-1, channels))
            triplet_msg = triplet_msg.view(n_valid, k, channels)
            triplet_msg = triplet_msg * mask_pair_f.to(triplet_msg.dtype)
            update_v = triplet_msg.sum(dim=1) / denom
        else:
            triplet_input = triplet_input * mask_pair_f.to(triplet_input.dtype)
            update_v = self.ffn(triplet_input.sum(dim=1) / denom)
        update_v = self.act(self.norm(update_v))

        update = features.new_zeros(n_points, channels)
        update[valid_idx] = update_v

        score_v = score_sum / denom.to(score_sum.dtype)

        score = features.new_zeros(n_points, 1)
        score[valid_idx] = score_v

        return input_features + update, score


# Optional explicit alias for versioned imports.
TripletMotionConsistencyMotionPairSparseConvV21 = TripletMotionConsistencyMotionPairSparseConv
