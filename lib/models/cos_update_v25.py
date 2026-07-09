import torch
import torch.nn as nn

from lib.models.feature_topk_v25_loader import load_feature_topk_v25_exact

feature_topk_v25_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V25 single-dilation feature-neighbor attention.

    This implementation is specialized for tpdilation=[1]. It always uses the
    CUDA top-k extension to select K previous-frame neighbors, one current self
    slot plus K-1 current-frame neighbors, and K next-frame neighbors.
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
        self.k_laiyuan = kwargs.pop("k_laiyuan", "coords")
        self.useFFN = bool(kwargs.pop("useFFN", True))
        kwargs.pop("ffn_position", None)
        kwargs.pop("topk_relu", None)
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv25 kwargs: {sorted(kwargs)}")
        if self.k_laiyuan not in ("coords", "features"):
            raise ValueError(f"Unsupported k_laiyuan: {self.k_laiyuan}")

        del kernel_size, alpha, pos_scale
        self.in_channels = int(in_channels)
        self.temporal_dilation = self._validate_single_dilation(tpdilation, temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv25 requires tmc_topk > 0")

        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        self.window_extent = 2 * self.window_radius + 1
        self.center_offset_pos = (self.window_extent * self.window_extent) // 2
        self.pos_scale = float(max(self.window_radius, 1))
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = max(4, int(round(self.in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))
        use_coord_keys = self.k_laiyuan == "coords"
        use_feature_keys = self.k_laiyuan == "features"

        self.norm_attn = nn.LayerNorm(self.in_channels)
        self.norm_ffn = nn.LayerNorm(self.in_channels) if self.useFFN else None
        self.in_proj = nn.Linear(self.in_channels, 4 * self.in_channels, bias=False)
        self.key_proj = nn.Linear(self.in_channels, 3 * self.in_channels, bias=False) if use_feature_keys else None
        self.traj_mlp = (
            nn.Sequential(
                nn.Linear(4, pos_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(pos_hidden, self.in_channels),
            )
            if use_coord_keys
            else None
        )
        self.out_proj = nn.Linear(self.in_channels, self.in_channels, bias=False)
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
            raise ValueError(f"cosv25 is specialized for tpdilation=[1], got {values}")
        return 1

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
        q_key = prefix + "q_proj.0.weight"
        prev_key = prefix + "prev_proj.0.weight"
        cur_key = prefix + "cur_proj.0.weight"
        next_key = prefix + "next_proj.0.weight"
        in_key = prefix + "in_proj.weight"
        if in_key not in state_dict and all(k in state_dict for k in (q_key, prev_key, cur_key, next_key)):
            state_dict[in_key] = torch.cat(
                [state_dict[q_key], state_dict[prev_key], state_dict[cur_key], state_dict[next_key]],
                dim=0,
            )

        kp_key = prefix + "k_prev_proj.0.weight"
        kc_key = prefix + "k_cur_proj.0.weight"
        kn_key = prefix + "k_next_proj.0.weight"
        key_key = prefix + "key_proj.weight"
        if self.key_proj is not None and key_key not in state_dict and all(k in state_dict for k in (kp_key, kc_key, kn_key)):
            state_dict[key_key] = torch.cat([state_dict[kp_key], state_dict[kc_key], state_dict[kn_key]], dim=0)

        if prefix + "out_proj.weight" not in state_dict and prefix + "out_proj.0.weight" in state_dict:
            state_dict[prefix + "out_proj.weight"] = state_dict[prefix + "out_proj.0.weight"]
        migrated_keys = [q_key, prev_key, cur_key, next_key, kp_key, kc_key, kn_key, prefix + "out_proj.0.weight"]
        if self.traj_mlp is not None:
            for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
                old_key = prefix + "traj_mlp.0." + suffix
                new_key = prefix + "traj_mlp." + suffix
                if new_key not in state_dict and old_key in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                migrated_keys.append(old_key)
        for old_key in migrated_keys:
            if old_key in state_dict:
                state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        migrated_prefixes = (
            prefix + "q_proj.0.",
            prefix + "prev_proj.0.",
            prefix + "cur_proj.0.",
            prefix + "next_proj.0.",
            prefix + "k_prev_proj.0.",
            prefix + "k_cur_proj.0.",
            prefix + "k_next_proj.0.",
            prefix + "out_proj.0.",
            prefix + "traj_mlp.0.",
        )
        disabled_prefixes = () if self.key_proj is not None else (prefix + "key_proj.",)
        disabled_prefixes += () if self.traj_mlp is not None else (prefix + "traj_mlp.",)
        missing_keys[:] = [key for key in missing_keys if not key.startswith(disabled_prefixes)]
        unexpected_keys[:] = [key for key in unexpected_keys if not key.startswith(migrated_prefixes + disabled_prefixes)]

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
                raise RuntimeError("cosv25 requires CUDA index_map; no CPU fallback is available")
            if index_map.dtype != torch.int32:
                index_map = index_map.to(dtype=torch.int32)
            return index_map.contiguous()
        B, T, H, W = self._infer_dense_shape(x_conv, indices)
        return self._build_index_map(indices, B, T, H, W)

    @staticmethod
    def _check_conv_contract(input_features, x_conv):
        if x_conv.features.shape != input_features.shape:
            raise RuntimeError(
                "cosv25 requires conv to preserve sparse point count, order, and channels; "
                f"got input features {tuple(input_features.shape)} and conv features {tuple(x_conv.features.shape)}"
            )

    def _load_cuda_topk(self):
        global feature_topk_v25_exact
        if feature_topk_v25_exact is None:
            try:
                feature_topk_v25_exact = load_feature_topk_v25_exact()
            except ImportError as exc:
                raise ImportError(
                    "feature_topk_v25_cuda_ext is required for cosv25. Compile it with: "
                    "cd lib/feature_topk_v25_cuda && python setup.py build_ext --inplace"
                ) from exc
        return feature_topk_v25_exact

    def _select_topk_neighbors(self, indices, features_norm, index_map):
        if not indices.is_cuda or not features_norm.is_cuda:
            raise RuntimeError("cosv25 Top-K requires CUDA indices and features; no Python fallback is available")
        return self._load_cuda_topk()(
            indices=indices,
            index_map=index_map,
            features=features_norm,
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=1,
            topk=self.topk,
        )

    def _build_coord_keys(self, q_all, top_dist, top_pos, mask_nei):
        n_points = q_all.shape[0]
        k = self.topk
        device = q_all.device
        dtype = q_all.dtype
        slots = 3 * k

        rel_t = torch.cat(
            [
                torch.full((n_points, k, 1), -1.0, device=device, dtype=dtype),
                torch.zeros((n_points, k, 1), device=device, dtype=dtype),
                torch.full((n_points, k, 1), 1.0, device=device, dtype=dtype),
            ],
            dim=1,
        )
        flat_pos = top_pos.reshape(-1).clamp_min(0)
        rel_yx = self.window_offsets.to(device=device)[flat_pos].to(dtype=dtype).view(n_points, slots, 2)
        rel_yx = rel_yx / max(self.pos_scale, 1e-6)
        dist_norm = torch.sqrt(top_dist.clamp_min(0).to(dtype=dtype) / float(self.in_channels)).unsqueeze(-1)
        key_input = torch.cat([rel_t, rel_yx, dist_norm], dim=-1)
        key_input = key_input.masked_fill(~mask_nei.unsqueeze(-1), 0)
        return self.traj_mlp(key_input.reshape(-1, 4)).view(n_points, slots, self.in_channels)

    def _build_feature_keys(self, features_norm, idx_prev, idx_cur, idx_next):
        k_prev_all, k_cur_all, k_next_all = self.key_proj(features_norm).chunk(3, dim=-1)
        return torch.cat([k_prev_all[idx_prev], k_cur_all[idx_cur], k_next_all[idx_next]], dim=1)

    def forward(self, x, index_map=None):
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))
        if not input_features.is_cuda:
            raise RuntimeError("cosv25 requires CUDA sparse features; no CPU fallback is available")

        x_conv = self.conv(x)
        self._check_conv_contract(input_features, x_conv)
        indices = x_conv.indices
        features = x_conv.features
        if not indices.is_cuda or not features.is_cuda:
            raise RuntimeError("cosv25 requires CUDA indices and features; no Python fallback is available")
        if indices.shape[0] != features.shape[0] or features.shape[1] != self.in_channels:
            raise RuntimeError("cosv25 received inconsistent sparse indices/features shape")

        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        features_norm = self.norm_attn(features)

        (
            idx_prev,
            idx_cur,
            idx_next,
            mask_prev,
            mask_cur,
            mask_next,
            dist_prev,
            dist_cur,
            dist_next,
            pos_prev,
            pos_cur,
            pos_next,
        ) = self._select_topk_neighbors(indices, features_norm, dense_index_map)

        mask_nei = torch.cat([mask_prev, mask_cur, mask_next], dim=1)
        top_dist = torch.cat([dist_prev, dist_cur, dist_next], dim=1)
        top_pos = torch.cat([pos_prev, pos_cur, pos_next], dim=1)

        q_all, v_prev_all, v_cur_all, v_next_all = self.in_proj(features_norm).chunk(4, dim=-1)
        nei_value = torch.cat([v_prev_all[idx_prev], v_cur_all[idx_cur], v_next_all[idx_next]], dim=1)

        q = q_all.unsqueeze(1)
        if self.k_laiyuan == "coords":
            traj_key = self._build_coord_keys(q_all, top_dist, top_pos, mask_nei)
        elif self.k_laiyuan == "features":
            traj_key = self._build_feature_keys(features_norm, idx_prev, idx_cur, idx_next)
        else:
            raise ValueError(f"Unsupported k_laiyuan: {self.k_laiyuan}")

        attn = (q * traj_key).sum(dim=-1) * self.scale
        attn = attn.masked_fill(~mask_nei, -1e4)
        attn = torch.softmax(attn, dim=1)
        attn = attn * mask_nei.to(attn.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        neighbor_context = (attn.unsqueeze(-1) * nei_value).sum(dim=1)
        attn_update = self.out_proj(neighbor_context)
        x_attn = features + attn_update
        output_features = x_attn + self.ffn(self.norm_ffn(x_attn)) if self.useFFN else x_attn
        score = attn.amax(dim=1, keepdim=True)
        return output_features, score


TripletMotionConsistencyMotionPairSparseConvV25 = TripletMotionConsistencyMotionPairSparseConv
