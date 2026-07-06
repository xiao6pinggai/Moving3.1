import torch
import torch.nn as nn

from lib.models.feature_topk_v23_loader import load_feature_topk_v23_exact

feature_topk_v23_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V23: feature-nearest prev/next candidates with marginalized trajectory attention.

    For each current sparse point, candidates are selected independently from the
    previous and next temporal windows by the smallest feature-distance to the
    current point. The K x K pair attention is used only to produce marginal
    weights for O(K) prev/next value aggregation.
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
        self.quanjuduiji = True
        del kernel_size, alpha, valid_norm, topk_relu

        kwargs.pop("ffn_position", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv23 kwargs: {sorted(kwargs)}")

        self.in_channels = int(in_channels)
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv23 requires tmc_topk > 0")

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

        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)

        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.traj_mlp = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, in_channels),
        )

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

    def _select_topk_by_feature_distance(self, indices, features_norm, index_map):
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
                    temporal_dilation=self.temporal_dilation,
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
                temporal_offset=-self.temporal_dilation,
            )
            next_idx, next_mask = self._select_one_side(
                query_indices,
                query_features,
                features_norm,
                index_map,
                temporal_offset=self.temporal_dilation,
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

        has_pair = mask_prev.any(dim=1) & mask_next.any(dim=1)
        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)

        if valid_idx.numel() == 0:
            return input_features, features.new_zeros((n_points, 1))

        update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)
        spatial = indices[:, 2:4].to(features.dtype)

        pair_chunk = self._adaptive_chunk_size(valid_idx.numel(), k * k, channels, pair_factor=1)
        for start in range(0, valid_idx.numel(), pair_chunk):
            end = min(start + pair_chunk, valid_idx.numel())
            rows = valid_idx[start:end]
            n_valid = rows.shape[0]

            idx_prev_v = idx_prev[rows]
            idx_next_v = idx_next[rows]
            mask_prev_v = mask_prev[rows]
            mask_next_v = mask_next[rows]
            pair_mask = mask_prev_v.unsqueeze(2) & mask_next_v.unsqueeze(1)

            f_cur_res = input_features[rows]
            f_cur = features_norm[rows]
            f_prev = features_norm[idx_prev_v.reshape(-1)].view(n_valid, k, channels)
            f_next = features_norm[idx_next_v.reshape(-1)].view(n_valid, k, channels)

            q = self.q_proj(f_cur).view(n_valid, 1, 1, channels)
            cur_value = self.cur_proj(f_cur)
            prev_value = self.prev_proj(f_prev)
            next_value = self.next_proj(f_next)

            s_cur = spatial[rows].view(n_valid, 1, 1, 2)
            s_prev = spatial[idx_prev_v].view(n_valid, k, 1, 2)
            s_next = spatial[idx_next_v].view(n_valid, 1, k, 2)

            d_minus = s_cur - s_prev
            d_plus = s_next - s_cur
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

            traj_key = self.traj_mlp(motion_input.reshape(-1, 6)).view(n_valid, k, k, channels)
            attn = (q * traj_key).sum(dim=-1) * self.scale
            attn = attn.masked_fill(~pair_mask, -1e4)
            attn = torch.softmax(attn.view(n_valid, k * k), dim=-1).view(n_valid, k, k)
            attn = attn * pair_mask.to(attn.dtype)
            attn = attn / attn.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

            w_prev = attn.sum(dim=2)
            w_next = attn.sum(dim=1)
            h_prev = (w_prev.unsqueeze(-1) * prev_value).sum(dim=1)
            h_next = (w_next.unsqueeze(-1) * next_value).sum(dim=1)

            attn_update = self.out_proj(cur_value + h_prev + h_next)
            x_attn = f_cur_res + attn_update
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn))

            update[rows] = out_v - f_cur_res
            score[rows] = attn.amax(dim=(1, 2), keepdim=False).unsqueeze(1)

        return input_features + update, score


TripletMotionConsistencyMotionPairSparseConvV23 = TripletMotionConsistencyMotionPairSparseConv
