import ast

import torch
import torch.nn as nn

from lib.models.feature_topk_v25_loader import load_feature_topk_v25_exact

feature_topk_v25_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V25: neighbor attention without explicit prev-current-next triplets.

    This keeps the V24 multi-temporal-dilation channel split, but each branch
    selects top-k neighbors independently from previous/current/next temporal
    windows. Attention is normalized over the valid 3 * K neighbor slots.
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
        chunk_size=1024,
        valid_norm=True,
        use_cuda_kernel=True,
        allow_python_fallback=False,
        topk_relu="fanghui",
        **kwargs,
    ):
        super().__init__()
        self.k_laiyuan = kwargs.pop("k_laiyuan", "coords")
        self.useFFN = bool(kwargs.pop("useFFN", True))
        del kernel_size, alpha, valid_norm, topk_relu

        kwargs.pop("ffn_position", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv25 kwargs: {sorted(kwargs)}")
        if self.k_laiyuan not in ("coords", "features"):
            raise ValueError(f"Unsupported k_laiyuan: {self.k_laiyuan}")

        self.in_channels = int(in_channels)
        self.tpdilation = self._parse_tpdilation(tpdilation, temporal_dilation)
        self.temporal_dilation = int(self.tpdilation[0])
        self.channel_splits = self._split_channels(self.in_channels, len(self.tpdilation))
        self.channel_offsets = []
        start = 0
        for split_channels in self.channel_splits:
            end = start + split_channels
            self.channel_offsets.append((start, end))
            start = end

        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv25 requires tmc_topk > 0")

        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        del pos_scale
        self.pos_scale = float(max(self.window_radius, 1))
        self.chunk_size = int(chunk_size)
        self.conv = conv if conv is not None else nn.Identity()
        self.use_cuda_kernel = bool(use_cuda_kernel)
        self.allow_python_fallback = bool(allow_python_fallback)

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))
        use_feature_keys = self.k_laiyuan == "features"
        use_coord_keys = self.k_laiyuan == "coords"

        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels) if self.useFFN else None

        self.q_proj = nn.ModuleList()
        self.cur_proj = nn.ModuleList()
        self.prev_proj = nn.ModuleList()
        self.next_proj = nn.ModuleList()
        self.k_cur_proj = nn.ModuleList() if use_feature_keys else None
        self.k_prev_proj = nn.ModuleList() if use_feature_keys else None
        self.k_next_proj = nn.ModuleList() if use_feature_keys else None
        self.out_proj = nn.ModuleList()
        self.traj_mlp = nn.ModuleList() if use_coord_keys else None
        self.scales = []
        for split_channels in self.channel_splits:
            self.q_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            self.cur_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            self.prev_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            self.next_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            if use_feature_keys:
                self.k_cur_proj.append(nn.Linear(split_channels, split_channels, bias=False))
                self.k_prev_proj.append(nn.Linear(split_channels, split_channels, bias=False))
                self.k_next_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            self.out_proj.append(nn.Linear(split_channels, split_channels, bias=False))
            if use_coord_keys:
                self.traj_mlp.append(
                    nn.Sequential(
                        nn.Linear(4, pos_hidden),
                        nn.ReLU(inplace=True),
                        nn.Linear(pos_hidden, split_channels),
                    )
                )
            self.scales.append(split_channels ** -0.5)

        self.ffn = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, in_channels),
            )
            if self.useFFN
            else None
        )

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

    @staticmethod
    def _parse_tpdilation(tpdilation, temporal_dilation):
        if tpdilation is None:
            values = [int(temporal_dilation)]
        elif isinstance(tpdilation, str):
            text = tpdilation.strip()
            if text.startswith("[") or text.startswith("("):
                values = ast.literal_eval(text)
            elif "," in text:
                values = [token.strip() for token in text.split(",") if token.strip()]
            else:
                values = [token.strip() for token in text.split() if token.strip()]
        elif isinstance(tpdilation, (list, tuple)):
            values = list(tpdilation)
        else:
            values = [tpdilation]
        values = [int(value) for value in values]
        if not values:
            raise ValueError("cosv25 requires tpdilation to contain at least one value")
        if any(value <= 0 for value in values):
            raise ValueError(f"cosv25 tpdilation values must be positive, got {values}")
        return values

    @staticmethod
    def _split_channels(in_channels, n_branches):
        if n_branches <= 0:
            raise ValueError("n_branches must be positive")
        if in_channels < n_branches:
            raise ValueError(
                f"cosv25 requires in_channels >= len(tpdilation), got {in_channels} and {n_branches}"
            )
        base = int(in_channels) // int(n_branches)
        remainder = int(in_channels) - base * int(n_branches)
        return [base + remainder] + [base] * (n_branches - 1)

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
        module_items = (
            ("k_cur_proj.", self.k_cur_proj),
            ("k_prev_proj.", self.k_prev_proj),
            ("k_next_proj.", self.k_next_proj),
            ("norm_ffn.", self.norm_ffn),
            ("ffn.", self.ffn),
            ("traj_mlp.", self.traj_mlp),
        )
        v23_v24_removed = (
            prefix + "global_jitter_mlp.",
            prefix + "norm_sa_attn.",
            prefix + "norm_sa_ffn.",
            prefix + "sa_q_proj.",
            prefix + "sa_k_proj.",
            prefix + "sa_v_proj.",
            prefix + "sa_out_proj.",
            prefix + "sa_ffn.",
        )
        disabled_prefixes = tuple(prefix + module_prefix for module_prefix, module in module_items if module is None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        compatible_prefixes = disabled_prefixes + v23_v24_removed
        missing_keys[:] = [key for key in missing_keys if not key.startswith(disabled_prefixes)]
        unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(compatible_prefixes)]

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

    def _adaptive_chunk_size(self, n_points, n_offsets, channels, pair_factor=1):
        if self.chunk_size > 0:
            return max(1, self.chunk_size)
        denom = max(1, n_offsets * channels * pair_factor)
        return max(1, min(int(n_points), 4_000_000 // denom))

    def _lookup_window(self, query_indices, index_map, temporal_offset):
        device = query_indices.device
        offsets = self.window_offsets.to(device=device).long()
        B, T, H, W = index_map.shape

        b = query_indices[:, 0:1].long()
        t = query_indices[:, 1:2].long() + int(temporal_offset)
        y = query_indices[:, 2:3].long() + offsets[:, 0].view(1, -1)
        x = query_indices[:, 3:4].long() + offsets[:, 1].view(1, -1)

        valid = (b >= 0) & (b < B) & (t >= 0) & (t < T) & (y >= 0) & (y < H) & (x >= 0) & (x < W)

        safe_b = b.expand_as(y).clamp(0, max(B - 1, 0))
        safe_t = t.expand_as(y).clamp(0, max(T - 1, 0))
        safe_y = y.clamp(0, max(H - 1, 0))
        safe_x = x.clamp(0, max(W - 1, 0))

        cand = index_map[safe_b, safe_t, safe_y, safe_x].long()
        return torch.where(valid, cand, cand.new_full(cand.shape, -1))

    def _select_one_side_topk(self, query_indices, query_features, query_rows, features_norm, index_map, temporal_offset):
        cand = self._lookup_window(query_indices, index_map, temporal_offset)
        safe_cand = cand.clamp_min(0)
        cand_features = features_norm[safe_cand.reshape(-1)]
        cand_features = cand_features.view(query_indices.shape[0], cand.shape[1], features_norm.shape[1])

        dist = (cand_features - query_features.unsqueeze(1)).square().sum(dim=-1)
        dist = dist.masked_fill(cand < 0, float("inf"))

        k_eff = min(self.topk, cand.shape[1])
        top_dist, top_pos = torch.topk(dist, k=k_eff, dim=1, largest=False, sorted=True)
        top_idx = cand.gather(1, top_pos)
        top_mask = torch.isfinite(top_dist)
        top_idx = torch.where(top_mask, top_idx, top_idx.new_zeros(top_idx.shape))
        if k_eff == self.topk:
            return top_idx, top_mask

        pad_cols = self.topk - k_eff
        idx_pad = top_idx.new_zeros(top_idx.shape[0], pad_cols)
        mask_pad = top_mask.new_zeros(top_mask.shape[0], pad_cols)
        return torch.cat([top_idx, idx_pad], dim=1), torch.cat([top_mask, mask_pad], dim=1)

    def _select_topk_neighbors(self, indices, features_norm, index_map, temporal_dilation):
        n_points, channels = features_norm.shape
        k = self.topk

        if indices.is_cuda and features_norm.is_cuda and self.use_cuda_kernel:
            global feature_topk_v25_exact
            if feature_topk_v25_exact is None:
                try:
                    feature_topk_v25_exact = load_feature_topk_v25_exact()
                except ImportError:
                    if not self.allow_python_fallback:
                        raise ImportError(
                            "feature_topk_v25_cuda_ext is not available. Compile it with: "
                            "cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
                        )
                    feature_topk_v25_exact = None
            if feature_topk_v25_exact is not None:
                return feature_topk_v25_exact(
                    indices=indices,
                    index_map=index_map,
                    features=features_norm,
                    window_offsets=self.window_offsets.to(device=indices.device),
                    temporal_dilation=int(temporal_dilation),
                    topk=k,
                )

        n_offsets = int(self.window_offsets.shape[0])
        chunk = self._adaptive_chunk_size(n_points, n_offsets * 3, channels)

        out_prev = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        out_cur = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        out_next = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        mask_prev = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)
        mask_cur = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)
        mask_next = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)

        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            query_indices = indices[start:end]
            query_features = features_norm[start:end]
            query_rows = torch.arange(start, end, device=indices.device, dtype=torch.long).view(-1, 1)

            prev_idx, prev_mask = self._select_one_side_topk(
                query_indices, query_features, query_rows, features_norm, index_map, -int(temporal_dilation)
            )
            cur_idx, cur_mask = self._select_one_side_topk(
                query_indices, query_features, query_rows, features_norm, index_map, 0
            )
            next_idx, next_mask = self._select_one_side_topk(
                query_indices, query_features, query_rows, features_norm, index_map, int(temporal_dilation)
            )
            out_prev[start:end] = prev_idx
            out_cur[start:end] = cur_idx
            out_next[start:end] = next_idx
            mask_prev[start:end] = prev_mask
            mask_cur[start:end] = cur_mask
            mask_next[start:end] = next_mask

        return out_prev, out_cur, out_next, mask_prev, mask_cur, mask_next

    def _project_neighbor_values(self, branch_id, f_prev, f_cur_nei, f_next):
        return torch.cat(
            [
                self.prev_proj[branch_id](f_prev),
                self.cur_proj[branch_id](f_cur_nei),
                self.next_proj[branch_id](f_next),
            ],
            dim=1,
        )

    def _build_feature_keys(self, branch_id, f_cur, f_prev, f_cur_nei, f_next):
        cur_key = self.k_cur_proj[branch_id](f_cur).unsqueeze(1)
        nei_key = torch.cat(
            [
                self.k_prev_proj[branch_id](f_prev),
                self.k_cur_proj[branch_id](f_cur_nei),
                self.k_next_proj[branch_id](f_next),
            ],
            dim=1,
        )
        return cur_key + nei_key

    def forward(self, x, index_map=None):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features
        n_points, channels = features.shape
        k = self.topk
        slots = 3 * k

        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        features_norm = self.norm_attn(features)

        attn_update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)
        valid_any = torch.zeros(n_points, dtype=torch.bool, device=features.device)
        coords = indices[:, 1:4].to(features.dtype)

        for branch_id, dilation in enumerate(self.tpdilation):
            ch_start, ch_end = self.channel_offsets[branch_id]
            split_channels = ch_end - ch_start
            features_part = features_norm[:, ch_start:ch_end].contiguous()

            idx_prev, idx_cur, idx_next, mask_prev, mask_cur, mask_next = self._select_topk_neighbors(
                indices,
                features_part,
                dense_index_map,
                dilation,
            )
            mask_nei = torch.cat([mask_prev, mask_cur, mask_next], dim=1)

            valid_idx = mask_nei.any(dim=1).nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            valid_any[valid_idx] = True
            pair_chunk = self._adaptive_chunk_size(valid_idx.numel(), slots, split_channels, pair_factor=1)
            for start in range(0, valid_idx.numel(), pair_chunk):
                end = min(start + pair_chunk, valid_idx.numel())
                rows = valid_idx[start:end]
                n_valid = rows.shape[0]

                idx_prev_v = idx_prev[rows]
                idx_cur_v = idx_cur[rows]
                idx_next_v = idx_next[rows]
                mask_v = mask_nei[rows]
                idx_v = torch.cat([idx_prev_v, idx_cur_v, idx_next_v], dim=1)

                f_cur = features_part[rows]
                f_prev = features_part[idx_prev_v.reshape(-1)].view(n_valid, k, split_channels)
                f_cur_nei = features_part[idx_cur_v.reshape(-1)].view(n_valid, k, split_channels)
                f_next = features_part[idx_next_v.reshape(-1)].view(n_valid, k, split_channels)
                f_nei = torch.cat([f_prev, f_cur_nei, f_next], dim=1)

                q = self.q_proj[branch_id](f_cur).view(n_valid, 1, split_channels)
                cur_value = self.cur_proj[branch_id](f_cur)
                nei_value = self._project_neighbor_values(branch_id, f_prev, f_cur_nei, f_next)

                if self.k_laiyuan == "coords":
                    cur_coords = coords[rows].view(n_valid, 1, 3)
                    nei_coords = coords[idx_v.reshape(-1)].view(n_valid, slots, 3)
                    rel = nei_coords - cur_coords
                    temporal_scale = float(max(abs(int(dilation)), 1))
                    rel_t = rel[..., 0:1] / temporal_scale
                    rel_yx = rel[..., 1:3] / max(self.pos_scale, 1e-6)
                    dist_norm = torch.sqrt((f_nei - f_cur.unsqueeze(1)).square().sum(dim=-1).clamp_min(0) / max(float(split_channels), 1.0)).unsqueeze(-1)
                    key_input = torch.cat([rel_t, rel_yx, dist_norm], dim=-1)
                    traj_key = self.traj_mlp[branch_id](key_input.reshape(-1, 4)).view(
                        n_valid, slots, split_channels
                    )
                elif self.k_laiyuan == "features":
                    traj_key = self._build_feature_keys(branch_id, f_cur, f_prev, f_cur_nei, f_next)
                else:
                    raise ValueError(f"Unsupported k_laiyuan: {self.k_laiyuan}")

                attn = (q * traj_key).sum(dim=-1) * self.scales[branch_id]
                attn = attn.masked_fill(~mask_v, -1e4)
                attn = torch.softmax(attn, dim=1)
                attn = attn * mask_v.to(attn.dtype)
                attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

                neighbor_context = (attn.unsqueeze(-1) * nei_value).sum(dim=1)
                value = cur_value + neighbor_context
                branch_update = self.out_proj[branch_id](value)
                attn_update[rows, ch_start:ch_end] = branch_update
                score[rows] = torch.maximum(score[rows], attn.amax(dim=1, keepdim=True))

        valid_idx = valid_any.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            output_features = input_features
        else:
            update = input_features.new_zeros(n_points, channels)
            x_attn = input_features[valid_idx] + attn_update[valid_idx]
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn)) if self.useFFN else x_attn
            update[valid_idx] = out_v - input_features[valid_idx]
            output_features = input_features + update

        return output_features, score


TripletMotionConsistencyMotionPairSparseConvV25 = TripletMotionConsistencyMotionPairSparseConv
