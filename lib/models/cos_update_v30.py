import torch
import torch.nn as nn

from lib.models.feature_topk_v23_loader import load_feature_topk_v23_exact
from lib.models.spconv_utils import replace_feature, spconv

feature_topk_v23_exact = None


class SparseAvgPool(nn.Module):
    def forward(self, x):
        features = x.features
        batch_ids = x.indices[:, 0].long()
        batch_size = int(x.batch_size)
        channels = features.shape[1]

        pooled = features.new_zeros(batch_size, channels)
        counts = features.new_zeros(batch_size, 1)
        pooled.index_add_(0, batch_ids, features)
        counts.index_add_(0, batch_ids, features.new_ones(features.shape[0], 1))
        return pooled / counts.clamp_min(1.0)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(1, channels // int(reduction))
        self.avg_pool = SparseAvgPool()
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se = self.fc(self.avg_pool(x)).to(dtype=x.features.dtype)
        point_se = se[x.indices[:, 0].long()]
        return replace_feature(x, point_se * x.features)




class GroupedDilatedSparseConv(spconv.SparseModule):
    def __init__(self, channels, dilations, indice_key=None, se_reduction=4):
        super().__init__()
        self.dilations = list(dilations)
        self.groups = len(self.dilations)
        if self.groups <= 0:
            raise ValueError("GDilation must contain at least one dilation")
        if channels % self.groups != 0:
            raise ValueError(
                f"Grouped dilation requires channels divisible by {self.groups}, got {channels}"
            )

        self.group_channels = channels // self.groups
        self.convs = nn.ModuleList()
        for dilation in self.dilations:
            branch_key = f"{indice_key}_d{dilation}" if indice_key is not None else None
            self.convs.append(
                spconv.SubMConv3d(
                    self.group_channels,
                    self.group_channels,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                    bias=False,
                    indice_key=branch_key,
                )
            )

        pw_key = f"{indice_key}_pw" if indice_key is not None else None
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pw = spconv.SparseSequential(
                        spconv.SubMConv3d(
                        channels,
                        channels,
                        kernel_size=1,
                        padding=0,
                        bias=False,
                        indice_key=pw_key,
                    ),
                        nn.BatchNorm1d(channels),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        out_chunks = []
        for chunk, conv in zip(x.features.chunk(self.groups, dim=1), self.convs):
            out_chunks.append(conv(replace_feature(x, chunk)).features)

        x = replace_feature(x, torch.cat(out_chunks, dim=1))
        x = replace_feature(x, self.relu(self.bn(x.features)))
        return self.pw(x)


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V30: V29 with 9D trajectory keys.

    Compared with V29, the trajectory key input is extended from the 6D
    coordinate motion pattern [d_minus, d_plus, accel] to a 9D descriptor by
    appending feature-distance motion terms [f_minus, f_plus, delta_f].
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
        chunk_size=0,
        ffn_position=None,
        use_qkv=False,
        use_maxpool=False,
        use_biqkv=False,
    ):
        super().__init__()
        del kernel_size, alpha, ffn_position, use_qkv, use_maxpool, use_biqkv

        self.GD = False
        self.GDilation = [1, 2]

        self.in_channels = int(in_channels)
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv30 requires tmc_topk > 0")

        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        del pos_scale
        self.pos_scale = float(max(self.window_radius, 1))
        self.chunk_size = int(chunk_size)
        self.gd_conv = (
            GroupedDilatedSparseConv(
                self.in_channels,
                self.GDilation,
                indice_key=f"cosv30_gd_c{self.in_channels}",
            )
            if self.GD
            else None
        )
        self.conv = conv if conv is not None else nn.Identity()
        self.tao = 1.0

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))
        # pos_hidden = int(round(in_channels * float(hidden_ratio)))
        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)

        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.traj_mlp = nn.Sequential(
            nn.Linear(9, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, in_channels),
        )
       
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )

        self.scale = in_channels ** -0.5

        coords = torch.arange(-self.window_radius, self.window_radius + 1, dtype=torch.int32)
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1).contiguous()
        offset_cost = offsets.to(torch.long).square().sum(dim=-1)
        offset_order = torch.argsort(offset_cost, stable=True).to(torch.long)
        self.register_buffer("window_offsets", offsets[offset_order].contiguous(), persistent=False)

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
                raise RuntimeError("cosv30 requires CUDA index_map; no Python fallback is available")
            if index_map.dtype != torch.int32:
                index_map = index_map.to(dtype=torch.int32)
            return index_map.contiguous()

        B, T, H, W = self._infer_dense_shape(x_conv, indices)
        return self._build_index_map(indices, B, T, H, W)

    @staticmethod
    def _check_conv_contract(input_features, x_conv, in_channels):
        if x_conv.features.shape != input_features.shape or x_conv.features.shape[1] != in_channels:
            raise RuntimeError(
                "cosv30 requires conv to preserve sparse point count, order, and channels; "
                f"got input features {tuple(input_features.shape)} and conv features {tuple(x_conv.features.shape)}"
            )

    def _adaptive_chunk_size(self, n_points, n_offsets, channels, pair_factor=1):
        if self.chunk_size > 0:
            return max(1, self.chunk_size)
        denom = max(1, n_offsets * channels * pair_factor)
        return max(1, min(int(n_points), 4_000_000 // denom))

    def _load_cuda_topk(self):
        global feature_topk_v23_exact
        if feature_topk_v23_exact is None:
            feature_topk_v23_exact = load_feature_topk_v23_exact()
        return feature_topk_v23_exact

    def _select_topk_by_feature_distance(self, indices, features_norm, index_map):
        if not indices.is_cuda or not features_norm.is_cuda or not index_map.is_cuda:
            raise RuntimeError("cosv30 Top-K requires CUDA tensors; no Python fallback is available")
        return self._load_cuda_topk()(
            indices=indices,
            index_map=index_map,
            features=features_norm,
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=self.temporal_dilation,
            topk=self.topk,
        )

    # def _masked_temperature_softmax(self, logits, mask):
    #     attn = torch.sigmoid(logits / self.tao)
    #     attn = attn * mask.to(attn.dtype)
    #     return attn / (attn.sum(dim=-1, keepdim=True) + 1.0)
    def _masked_temperature_softmax(self, logits, mask):
        mask_f = mask.to(logits.dtype)
        attn = torch.sigmoid(logits / self.tao) * mask_f
        denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return attn / denom

    def forward(self, x, index_map=None):
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))
        if not input_features.is_cuda:
            raise RuntimeError("cosv30 requires CUDA sparse features; no Python fallback is available")

        x_conv = self.gd_conv(x) if self.GD else x
        x_conv = self.conv(x_conv)
        self._check_conv_contract(input_features, x_conv, self.in_channels)
        indices = x_conv.indices
        features = x_conv.features
        if not indices.is_cuda or not features.is_cuda:
            raise RuntimeError("cosv30 requires CUDA indices and features; no Python fallback is available")

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
        update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)
        if valid_idx.numel() == 0:
            return input_features, score

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
            f_cur_pair = f_cur.view(n_valid, 1, 1, channels)
            f_prev_pair = f_prev.view(n_valid, k, 1, channels)
            f_next_pair = f_next.view(n_valid, 1, k, channels)
            f_minus = (f_cur_pair - f_prev_pair).norm(dim=-1, keepdim=True)
            f_plus = (f_next_pair - f_cur_pair).norm(dim=-1, keepdim=True)
            delta_f = f_plus - f_minus
            motion_input = torch.cat(
                [
                    d_minus.expand(-1, -1, k, -1),
                    d_plus.expand(-1, k, -1, -1),
                    accel,
                    f_minus.expand(-1, -1, k, -1),
                    f_plus.expand(-1, k, -1, -1),
                    delta_f,
                ],
                dim=-1,
            )
            

            traj_key = self.traj_mlp(motion_input.reshape(-1, 9)).view(n_valid, k, k, channels)
            logits = (q * traj_key).sum(dim=-1) * self.scale
            attn = self._masked_temperature_softmax(
                logits.view(n_valid, k * k),
                pair_mask.view(n_valid, k * k),
            ).view(n_valid, k, k)

            w_prev = attn.sum(dim=2)
            w_next = attn.sum(dim=1)
            h_prev = (w_prev.unsqueeze(-1) * prev_value).sum(dim=1)
            h_next = (w_next.unsqueeze(-1) * next_value).sum(dim=1)
            value = cur_value + h_prev + h_next

            attn_update = self.out_proj(value)
            x_attn = f_cur_res + attn_update
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn))

            update[rows] = out_v - f_cur_res
            score[rows] = attn.amax(dim=(1, 2), keepdim=False).unsqueeze(1)

        return input_features + update, score


TripletMotionConsistencyMotionPairSparseConvV30 = TripletMotionConsistencyMotionPairSparseConv
