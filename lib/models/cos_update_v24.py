import ast

import torch
import torch.nn as nn

from lib.models.feature_topk_v23_loader import load_feature_topk_v23_exact

feature_topk_v23_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V24: multi-temporal-dilation channel-split variant of V23.

    ``tpdilation`` controls the temporal offsets. Channels are split across the
    dilation branches; when the channel count is not divisible, the first branch
    receives the remainder. Each branch performs the V23 prev/next trajectory
    attention on its own channel slice and temporal dilation, then all slices are
    merged before the shared FFN.
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
        # useFFN=True,
        **kwargs,
    ):
        super().__init__()
        self.quanjuduiji = False
        self.wbianyuanhua = True # 与False等价，True计算量更小
        self.k_laiyuan = 'coords'
        self.add_sa = False
        self.useFFN = True
        del kernel_size, alpha, valid_norm, topk_relu

        kwargs.pop("ffn_position", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv24 kwargs: {sorted(kwargs)}")

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
            raise ValueError("cosv24 requires tmc_topk > 0")

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

        use_feature_keys = self.k_laiyuan == 'features'
        use_coord_keys = self.k_laiyuan == 'coords'

        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels) if self.useFFN else None
        self.norm_sa_attn = nn.LayerNorm(in_channels) if self.add_sa else None
        self.norm_sa_ffn = nn.LayerNorm(in_channels) if self.add_sa and self.useFFN else None

        self.q_proj = nn.ModuleList()
        self.cur_proj = nn.ModuleList()
        self.prev_proj = nn.ModuleList()
        self.next_proj = nn.ModuleList()
        self.k_cur_proj = nn.ModuleList() if use_feature_keys else None
        self.k_prev_proj = nn.ModuleList() if use_feature_keys else None
        self.k_next_proj = nn.ModuleList() if use_feature_keys else None
        self.out_proj = nn.ModuleList()
        self.traj_mlp = nn.ModuleList() if use_coord_keys else None
        self.sa_q_proj = nn.Linear(in_channels, in_channels, bias=False) if self.add_sa else None
        self.sa_k_proj = nn.Linear(in_channels, in_channels, bias=False) if self.add_sa else None
        self.sa_v_proj = nn.Linear(in_channels, in_channels, bias=False) if self.add_sa else None
        self.sa_out_proj = nn.Linear(in_channels, in_channels, bias=False) if self.add_sa else None
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
                        nn.Linear(6, pos_hidden),
                        nn.ReLU(inplace=True),
                        nn.Linear(pos_hidden, split_channels),
                    )
                )
            self.scales.append(split_channels ** -0.5)

        jitter_hidden = max(4, int(round(in_channels * 0.5)))
        self.global_jitter_mlp = (
            nn.Sequential(
                nn.Linear(in_channels * 3, jitter_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(jitter_hidden, 4),
            )
            if self.quanjuduiji
            else None
        )

        self.ffn = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, in_channels),
            )
            if self.useFFN
            else None
        )
        self.sa_ffn = (
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, in_channels),
            )
            if self.add_sa and self.useFFN
            else None
        )
        self.sa_scale = in_channels ** -0.5

        if self.global_jitter_mlp is not None:
            nn.init.zeros_(self.global_jitter_mlp[-1].weight)
            nn.init.zeros_(self.global_jitter_mlp[-1].bias)

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
            raise ValueError("cosv24 requires tpdilation to contain at least one value")
        if any(value <= 0 for value in values):
            raise ValueError(f"cosv24 tpdilation values must be positive, got {values}")
        return values

    @staticmethod
    def _split_channels(in_channels, n_branches):
        if n_branches <= 0:
            raise ValueError("n_branches must be positive")
        if in_channels < n_branches:
            raise ValueError(
                f"cosv24 requires in_channels >= len(tpdilation), got {in_channels} and {n_branches}"
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
            ("global_jitter_mlp.", self.global_jitter_mlp),
            ("k_cur_proj.", self.k_cur_proj),
            ("k_prev_proj.", self.k_prev_proj),
            ("k_next_proj.", self.k_next_proj),
            ("norm_sa_attn.", self.norm_sa_attn),
            ("norm_sa_ffn.", self.norm_sa_ffn),
            ("sa_q_proj.", self.sa_q_proj),
            ("sa_k_proj.", self.sa_k_proj),
            ("sa_v_proj.", self.sa_v_proj),
            ("sa_out_proj.", self.sa_out_proj),
            ("sa_ffn.", self.sa_ffn),
            ("norm_ffn.", self.norm_ffn),
            ("ffn.", self.ffn),
            ("traj_mlp.", self.traj_mlp),
        )
        legacy_optional = {
            "global_jitter_mlp.",
            "k_cur_proj.",
            "k_prev_proj.",
            "k_next_proj.",
            "norm_sa_attn.",
            "norm_sa_ffn.",
            "sa_q_proj.",
            "sa_k_proj.",
            "sa_v_proj.",
            "sa_out_proj.",
            "sa_ffn.",
        }
        disabled_prefixes = tuple(prefix + module_prefix for module_prefix, module in module_items if module is None)
        filled_prefixes = []
        for module_prefix, module in module_items:
            if module is None or module_prefix not in legacy_optional:
                continue
            filled_prefixes.append(prefix + module_prefix)
            for name, value in module.state_dict().items():
                key = prefix + module_prefix + name
                if key not in state_dict:
                    state_dict[key] = value.detach()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        compatible_missing_prefixes = tuple(filled_prefixes) + disabled_prefixes
        missing_keys[:] = [key for key in missing_keys if not key.startswith(compatible_missing_prefixes)]
        unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(disabled_prefixes)]

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

    def _select_topk_by_feature_distance(self, indices, features_norm, index_map, temporal_dilation):
        n_points, channels = features_norm.shape
        k = self.topk

        if indices.is_cuda and features_norm.is_cuda and self.use_cuda_kernel:
            global feature_topk_v23_exact
            if feature_topk_v23_exact is None:
                try:
                    feature_topk_v23_exact = load_feature_topk_v23_exact()
                except ImportError:
                    if not self.allow_python_fallback:
                        raise ImportError(
                            "feature_topk_v23_cuda_ext is not available. Compile it with: "
                            "cd lib/feature_topk_v23_cuda && python setup.py build_ext --inplace"
                        )
                    feature_topk_v23_exact = None
            if feature_topk_v23_exact is not None:
                return feature_topk_v23_exact(
                    indices=indices,
                    index_map=index_map,
                    features=features_norm,
                    window_offsets=self.window_offsets.to(device=indices.device),
                    temporal_dilation=int(temporal_dilation),
                    topk=self.topk,
                )

        n_offsets = int(self.window_offsets.shape[0])
        chunk = self._adaptive_chunk_size(n_points, n_offsets, channels)

        out_prev = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        out_next = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        mask_prev = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)
        mask_next = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)

        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            query_indices = indices[start:end]
            query_features = features_norm[start:end]

            prev_idx, prev_mask = self._select_one_side(
                query_indices,
                query_features,
                features_norm,
                index_map,
                temporal_offset=-int(temporal_dilation),
            )
            next_idx, next_mask = self._select_one_side(
                query_indices,
                query_features,
                features_norm,
                index_map,
                temporal_offset=int(temporal_dilation),
            )

            out_prev[start:end] = prev_idx
            out_next[start:end] = next_idx
            mask_prev[start:end] = prev_mask
            mask_next[start:end] = next_mask

        return out_prev, out_next, mask_prev, mask_next

    def _select_one_side(self, query_indices, query_features, features_norm, index_map, temporal_offset):
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

    @staticmethod
    def _global_average_pool_frames(features, indices, B, T):
        channels = features.shape[1]
        pooled = features.new_zeros(B * T, channels)
        counts = features.new_zeros(B * T, 1)

        frame_ids = indices[:, 0].long() * int(T) + indices[:, 1].long()
        pooled.index_add_(0, frame_ids, features)
        counts.index_add_(0, frame_ids, features.new_ones(features.shape[0], 1))

        pooled = pooled / counts.clamp_min(1.0)
        return pooled.view(B, T, channels), counts.view(B, T, 1) > 0

    def _predict_global_jitter_by_frame(self, features, indices, B, T, temporal_dilation):
        frame_pool, frame_has = self._global_average_pool_frames(features, indices, B, T)
        device = features.device
        batch_grid = torch.arange(B, device=device).view(B, 1).expand(B, T)
        time_grid = torch.arange(T, device=device).view(1, T).expand(B, T)

        prev_time = time_grid - int(temporal_dilation)
        next_time = time_grid + int(temporal_dilation)
        valid_prev = (prev_time >= 0) & (prev_time < T)
        valid_next = (next_time >= 0) & (next_time < T)
        safe_prev_time = prev_time.clamp(0, max(T - 1, 0))
        safe_next_time = next_time.clamp(0, max(T - 1, 0))

        prev_pool = frame_pool[batch_grid, safe_prev_time]
        cur_pool = frame_pool
        next_pool = frame_pool[batch_grid, safe_next_time]
        jitter_input = torch.cat([prev_pool, cur_pool, next_pool], dim=-1)

        jitter_xy = self.global_jitter_mlp(jitter_input.reshape(B * T, -1)).view(B, T, 4)
        scale = float(max(self.pos_scale, 1e-6))
        jitter_xy = torch.tanh(jitter_xy) * scale

        prev_has = valid_prev & frame_has[batch_grid, safe_prev_time, 0]
        next_has = valid_next & frame_has[batch_grid, safe_next_time, 0]
        prev_delta_xy = jitter_xy[..., 0:2] * prev_has.unsqueeze(-1).to(jitter_xy.dtype)
        next_delta_xy = jitter_xy[..., 2:4] * next_has.unsqueeze(-1).to(jitter_xy.dtype)

        prev_delta_yx = torch.stack([prev_delta_xy[..., 1], prev_delta_xy[..., 0]], dim=-1)
        next_delta_yx = torch.stack([next_delta_xy[..., 1], next_delta_xy[..., 0]], dim=-1)
        return prev_delta_yx, next_delta_yx


    def _select_self_topk_by_feature_distance(self, indices, features_norm, index_map):
        n_points, channels = features_norm.shape
        k = self.topk
        n_offsets = int(self.window_offsets.shape[0])
        chunk = self._adaptive_chunk_size(n_points, n_offsets, channels)

        out_idx = torch.zeros(n_points, k, dtype=torch.long, device=indices.device)
        out_mask = torch.zeros(n_points, k, dtype=torch.bool, device=indices.device)

        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            query_indices = indices[start:end]
            query_features = features_norm[start:end]
            self_idx, self_mask = self._select_one_side(
                query_indices,
                query_features,
                features_norm,
                index_map,
                temporal_offset=0,
            )
            out_idx[start:end] = self_idx
            out_mask[start:end] = self_mask

        return out_idx, out_mask

    def _apply_self_attention(self, features_in, indices, index_map):
        if not self.add_sa:
            return features_in

        n_points, channels = features_in.shape
        if n_points == 0:
            return features_in

        features_norm = self.norm_sa_attn(features_in)
        idx_self, mask_self = self._select_self_topk_by_feature_distance(
            indices,
            features_norm,
            index_map,
        )
        valid_idx = mask_self.any(dim=1).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return features_in

        update = features_in.new_zeros(n_points, channels)
        k = self.topk
        pair_chunk = self._adaptive_chunk_size(valid_idx.numel(), k, channels, pair_factor=1)
        for start in range(0, valid_idx.numel(), pair_chunk):
            end = min(start + pair_chunk, valid_idx.numel())
            rows = valid_idx[start:end]
            n_valid = rows.shape[0]

            idx_v = idx_self[rows]
            mask_v = mask_self[rows]
            f_cur_res = features_in[rows]
            f_cur = features_norm[rows]
            f_nei = features_norm[idx_v.reshape(-1)].view(n_valid, k, channels)

            q = self.sa_q_proj(f_cur).view(n_valid, 1, channels)
            key = self.sa_k_proj(f_nei)
            value = self.sa_v_proj(f_nei)

            attn = (q * key).sum(dim=-1) * self.sa_scale
            attn = attn.masked_fill(~mask_v, -1e4)
            attn = torch.softmax(attn, dim=-1)
            attn = attn * mask_v.to(attn.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

            sa_update = (attn.unsqueeze(-1) * value).sum(dim=1)
            sa_update = self.sa_out_proj(sa_update)
            x_sa = f_cur_res + sa_update
            out_v = x_sa + self.sa_ffn(self.norm_sa_ffn(x_sa)) if self.useFFN else x_sa
            update[rows] = out_v - f_cur_res

        return features_in + update

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
        if self.quanjuduiji:
            B, T, _, _ = dense_index_map.shape
            global_deltas = [
                self._predict_global_jitter_by_frame(features_norm, indices, B, T, dilation)
                for dilation in self.tpdilation
            ]
        else:
            global_deltas = [(None, None) for _ in self.tpdilation]

        attn_update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)
        valid_any = torch.zeros(n_points, dtype=torch.bool, device=features.device)
        spatial = indices[:, 2:4].to(features.dtype)

        for branch_id, dilation in enumerate(self.tpdilation):
            ch_start, ch_end = self.channel_offsets[branch_id]
            split_channels = ch_end - ch_start
            features_part = features_norm[:, ch_start:ch_end].contiguous()

            idx_prev, idx_next, mask_prev, mask_next = self._select_topk_by_feature_distance(
                indices,
                features_part,
                dense_index_map,
                dilation,
            )

            has_pair = mask_prev.any(dim=1) & mask_next.any(dim=1)
            valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            valid_any[valid_idx] = True
            prev_global_delta, next_global_delta = global_deltas[branch_id]
            pair_chunk = self._adaptive_chunk_size(valid_idx.numel(), k * k, split_channels, pair_factor=1)
            for start in range(0, valid_idx.numel(), pair_chunk):
                end = min(start + pair_chunk, valid_idx.numel())
                rows = valid_idx[start:end]
                n_valid = rows.shape[0]

                idx_prev_v = idx_prev[rows]
                idx_next_v = idx_next[rows]
                mask_prev_v = mask_prev[rows]
                mask_next_v = mask_next[rows]
                pair_mask = mask_prev_v.unsqueeze(2) & mask_next_v.unsqueeze(1)

                f_cur = features_part[rows]
                f_prev = features_part[idx_prev_v.reshape(-1)].view(n_valid, k, split_channels)
                f_next = features_part[idx_next_v.reshape(-1)].view(n_valid, k, split_channels)

                q = self.q_proj[branch_id](f_cur).view(n_valid, 1, 1, split_channels)
                cur_value = self.cur_proj[branch_id](f_cur)
                prev_value = self.prev_proj[branch_id](f_prev)
                next_value = self.next_proj[branch_id](f_next)

                s_cur = spatial[rows].view(n_valid, 1, 1, 2)
                s_prev = spatial[idx_prev_v].view(n_valid, k, 1, 2)
                s_next = spatial[idx_next_v].view(n_valid, 1, k, 2)

                d_minus = s_cur - s_prev
                d_plus = s_next - s_cur
                if prev_global_delta is not None and next_global_delta is not None:
                    row_coords = indices[rows].long()
                    batch_ids = row_coords[:, 0]
                    time_ids = row_coords[:, 1]
                    prev_delta = prev_global_delta[batch_ids, time_ids].view(n_valid, 1, 1, 2)
                    next_delta = next_global_delta[batch_ids, time_ids].view(n_valid, 1, 1, 2)
                    d_minus = d_minus + prev_delta
                    d_plus = d_plus - next_delta
                accel = d_plus - d_minus
                motion_input = torch.cat(
                    [
                        d_minus.expand(-1, -1, k, -1),
                        d_plus.expand(-1, k, -1, -1),
                        accel,
                    ],
                    dim=-1,
                )
                motion_input = motion_input / max(self.pos_scale, 1e-6)

                if self.k_laiyuan == 'coords':
                    traj_key = self.traj_mlp[branch_id](motion_input.reshape(-1, 6)).view(
                        n_valid, k, k, split_channels
                    )
                elif self.k_laiyuan == 'features':
                    cur_key = self.k_cur_proj[branch_id](f_cur).view(n_valid, 1, 1, split_channels)
                    prev_key = self.k_prev_proj[branch_id](f_prev).view(n_valid, k, 1, split_channels)
                    next_key = self.k_next_proj[branch_id](f_next).view(n_valid, 1, k, split_channels)
                    traj_key = cur_key + prev_key + next_key
                else:
                    raise ValueError(f"Unsupported k_laiyuan: {self.k_laiyuan}")
                attn = (q * traj_key).sum(dim=-1) * self.scales[branch_id]
                attn = attn.masked_fill(~pair_mask, -1e4)
                attn = torch.softmax(attn.view(n_valid, k * k), dim=-1).view(n_valid, k, k)
                attn = attn * pair_mask.to(attn.dtype)
                attn = attn / attn.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

                if self.wbianyuanhua:
                    w_prev = attn.sum(dim=2)
                    w_next = attn.sum(dim=1)
                    h_prev = (w_prev.unsqueeze(-1) * prev_value).sum(dim=1)
                    h_next = (w_next.unsqueeze(-1) * next_value).sum(dim=1)
                    value = cur_value + h_prev + h_next
                else:
                    pair_value = (
                        cur_value.view(n_valid, 1, 1, split_channels)
                        + prev_value.view(n_valid, k, 1, split_channels)
                        + next_value.view(n_valid, 1, k, split_channels)
                    )
                    value = (attn.unsqueeze(-1) * pair_value).sum(dim=(1, 2))

                branch_update = self.out_proj[branch_id](value)
                attn_update[rows, ch_start:ch_end] = branch_update
                branch_score = attn.amax(dim=(1, 2), keepdim=False).unsqueeze(1)
                score[rows] = torch.maximum(score[rows], branch_score)

        valid_idx = valid_any.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            output_features = input_features
        else:
            update = input_features.new_zeros(n_points, channels)
            x_attn = input_features[valid_idx] + attn_update[valid_idx]
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn)) if self.useFFN else x_attn
            update[valid_idx] = out_v - input_features[valid_idx]
            output_features = input_features + update

        output_features = self._apply_self_attention(output_features, indices, dense_index_map)
        return output_features, score


TripletMotionConsistencyMotionPairSparseConvV24 = TripletMotionConsistencyMotionPairSparseConv
