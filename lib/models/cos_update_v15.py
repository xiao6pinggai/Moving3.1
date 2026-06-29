import math
import os
import sys

import torch
import torch.nn as nn

_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)


class SparseTrajectoryTokenModule(nn.Module):
    """v15 sparse trajectory token feature enhancement module.

    The module keeps sparse topology unchanged:
    x.features [N, C] -> enhanced_features [N, C].

    Each cube is one 256 x 256 spatial window over all frames. Inside every
    cube, v15 runs local sparse spatiotemporal attention, foreground trajectory
    token read/fusion, global-pooled background token competition, and a
    no-chunk point write-back.
    """

    def __init__(
        self,
        in_channels,
        num_frames=10,
        num_traj=32,
        window_size=256,
        num_heads=4,
        alpha=0.5,
        local_temporal_radius=1,
        local_spatial_radius=1,
        conv=None,
        indice_key="subm0",
        kernel_size=5,
        stride=1,
        use_qkv=False,
        use_biqkv=False,
        use_maxpool=False,
        temporal_dilation=1,
        write_chunk_size=None,
        **kwargs,
    ):
        super().__init__()
        if in_channels % num_heads != 0:
            raise ValueError(
                f"in_channels must be divisible by num_heads, got "
                f"in_channels={in_channels}, num_heads={num_heads}"
            )
        if num_traj < 1:
            raise ValueError(f"num_traj must be >= 1, got {num_traj}")

        self.channels = in_channels
        self.num_frames = num_frames
        self.num_traj = num_traj
        self.window_size = window_size
        self.alpha = alpha
        self.local_temporal_radius = local_temporal_radius
        self.local_spatial_radius = local_spatial_radius

        # Compatibility-only arguments. v15 intentionally does not use the
        # old large kernel_size values passed by earlier MFE builders.
        self.conv = conv
        self.indice_key = indice_key
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_qkv = use_qkv
        self.use_biqkv = use_biqkv
        self.use_maxpool = use_maxpool
        self.temporal_dilation = temporal_dilation
        self.write_chunk_size = write_chunk_size

        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self._build_local_offsets()

        self.traj_embed = nn.Parameter(torch.empty(num_traj, in_channels))
        self.time_embed = nn.Parameter(torch.empty(num_frames, in_channels))
        self.bg_time_embed = nn.Parameter(torch.empty(num_frames, in_channels))

        self.feat_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        self.local_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.local_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.local_v_proj = nn.Linear(in_channels, in_channels, bias=False)
        local_hidden = max(8, min(32, in_channels))
        self.local_rel_bias = nn.Sequential(
            nn.Linear(3, local_hidden),
            nn.GELU(),
            nn.Linear(local_hidden, 1),
        )

        self.global_proj = nn.Linear(in_channels, in_channels)
        self.read_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.temporal_fusion = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=4 * in_channels,
            batch_first=True,
        )

        self.bg_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        self.write_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.out_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        self.reset_parameters()

    def _build_local_offsets(self):
        rt = int(self.local_temporal_radius)
        rs = int(self.local_spatial_radius)

        dt = torch.arange(-rt, rt + 1, dtype=torch.long)
        dy = torch.arange(-rs, rs + 1, dtype=torch.long)
        dx = torch.arange(-rs, rs + 1, dtype=torch.long)
        mesh_t, mesh_y, mesh_x = torch.meshgrid(dt, dy, dx, indexing='ij')
        offsets_t = mesh_t.reshape(-1)
        offsets_y = mesh_y.reshape(-1)
        offsets_x = mesh_x.reshape(-1)

        offset_keys = offsets_t * self.scale_t + offsets_y * self.scale_y + offsets_x
        t_norm = offsets_t.to(torch.float32) / max(float(rt), 1.0)
        y_norm = offsets_y.to(torch.float32) / max(float(rs), 1.0)
        x_norm = offsets_x.to(torch.float32) / max(float(rs), 1.0)
        offset_pos = torch.stack([t_norm, y_norm, x_norm], dim=1)

        self.register_buffer("local_offset_keys", offset_keys)
        self.register_buffer("local_offset_pos", offset_pos)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.traj_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.bg_time_embed, std=0.02)

        nn.init.eye_(self.feat_proj.weight)
        nn.init.eye_(self.local_v_proj.weight)
        nn.init.eye_(self.read_v_proj.weight)
        nn.init.eye_(self.write_v_proj.weight)

        nn.init.zeros_(self.pos_mlp[2].weight)
        nn.init.zeros_(self.pos_mlp[2].bias)
        nn.init.zeros_(self.out_mlp[2].weight)
        nn.init.zeros_(self.out_mlp[2].bias)

    def _cube_keys(self, indices):
        b = indices[:, 0].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        wy = torch.div(y, self.window_size, rounding_mode='floor')
        wx = torch.div(x_coord, self.window_size, rounding_mode='floor')
        return b * 10_000_000_000 + wy * 100_000 + wx

    def _position_encoding(self, indices):
        dtype = torch.float32
        t = indices[:, 1].to(dtype).clamp(0, self.num_frames - 1)
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        wy = torch.div(y, self.window_size, rounding_mode='floor')
        wx = torch.div(x_coord, self.window_size, rounding_mode='floor')
        local_y = (y - wy * self.window_size).to(dtype)
        local_x = (x_coord - wx * self.window_size).to(dtype)

        t_norm = t / max(self.num_frames - 1, 1)
        y_norm = local_y / max(self.window_size - 1, 1)
        x_norm = local_x / max(self.window_size - 1, 1)
        return torch.stack([t_norm, y_norm, x_norm], dim=1)

    @staticmethod
    def _edge_softmax(logits, row_idx, row_count):
        max_per_row = logits.new_full((row_count,), -float("inf"))
        if hasattr(max_per_row, "scatter_reduce_"):
            max_per_row.scatter_reduce_(
                0,
                row_idx,
                logits,
                reduce="amax",
                include_self=True,
            )
        else:
            unique_rows = torch.unique(row_idx)
            for row in unique_rows.tolist():
                mask = row_idx == row
                max_per_row[row] = logits[mask].max()

        weights = torch.exp(logits - max_per_row[row_idx])
        denom = logits.new_zeros((row_count,))
        denom.index_add_(0, row_idx, weights)
        return weights / denom[row_idx].clamp_min(1e-6)

    def _local_sparse_attention(self, point_tokens, indices):
        n_points, channels = point_tokens.shape
        if n_points == 0:
            return point_tokens

        dtype = point_tokens.dtype
        scale = math.sqrt(channels)

        t = indices[:, 1].long().clamp(0, self.num_frames - 1)
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        current_keys = t * self.scale_t + y * self.scale_y + x_coord
        sorted_keys, sort_idx = torch.sort(current_keys)

        q = self.local_q_proj(point_tokens)
        k = self.local_k_proj(point_tokens)
        v = self.local_v_proj(point_tokens)
        sorted_k = k[sort_idx]
        sorted_v = v[sort_idx]

        neighbor_keys = current_keys[:, None] + self.local_offset_keys[None, :]
        flat_neighbor_keys = neighbor_keys.reshape(-1)
        ptr = torch.searchsorted(sorted_keys, flat_neighbor_keys).clamp(max=n_points - 1)
        valid = sorted_keys[ptr] == flat_neighbor_keys

        if not bool(valid.any()):
            return point_tokens

        valid_flat = valid.nonzero(as_tuple=False).squeeze(1)
        offset_count = self.local_offset_keys.numel()
        row_idx = torch.div(valid_flat, offset_count, rounding_mode='floor')
        offset_idx = valid_flat % offset_count
        neighbor_ptr = ptr[valid_flat]

        q_edge = q[row_idx]
        k_edge = sorted_k[neighbor_ptr]
        bias = self.local_rel_bias(
            self.local_offset_pos[offset_idx].to(device=point_tokens.device, dtype=dtype)
        ).squeeze(1)
        logits = ((q_edge * k_edge).sum(dim=1) / scale + bias).float()
        attn = self._edge_softmax(logits, row_idx, n_points).to(dtype)

        context = point_tokens.new_zeros((n_points, channels))
        context.index_add_(0, row_idx, sorted_v[neighbor_ptr] * attn[:, None])
        return point_tokens + context

    def _temporal_fuse(self, traj_tokens, valid_frames):
        m_count = traj_tokens.shape[0]
        if not bool(valid_frames.any()):
            return traj_tokens

        key_padding_mask = ~valid_frames[None, :].expand(m_count, -1)
        fused = self.temporal_fusion(
            traj_tokens,
            src_key_padding_mask=key_padding_mask,
        )
        return fused * valid_frames[None, :, None].to(fused.dtype)

    def forward_cube(self, residual_w, features_w, indices_w):
        n_points, channels = features_w.shape
        device = features_w.device
        dtype = features_w.dtype
        m_count = self.num_traj
        l_count = self.num_frames
        scale = math.sqrt(channels)

        pos = self._position_encoding(indices_w).to(device=device, dtype=dtype)
        point_tokens = self.feat_proj(features_w) + self.pos_mlp(pos)
        e = self._local_sparse_attention(point_tokens, indices_w)

        cube_cond = self.global_proj(e.mean(dim=0, keepdim=True))
        q_traj = (
            self.traj_embed[:, None, :]
            + self.time_embed[None, :, :]
            + cube_cond[None, :, :]
        ).to(device=device, dtype=dtype)

        t_local = indices_w[:, 1].long().clamp(0, l_count - 1)
        q_read = self.read_q_proj(q_traj)
        k_all = self.read_k_proj(e)
        v_all = self.read_v_proj(e)

        traj_tokens = e.new_zeros((m_count, l_count, channels))
        valid_frames = torch.zeros(l_count, device=device, dtype=torch.bool)

        for frame_id in range(l_count):
            frame_mask = t_local == frame_id
            if not bool(frame_mask.any()):
                continue

            score = q_read[:, frame_id, :] @ k_all[frame_mask].T / scale
            attn = torch.softmax(score.float(), dim=1).to(dtype)
            traj_tokens[:, frame_id, :] = attn @ v_all[frame_mask]
            valid_frames[frame_id] = True

        traj_hat = self._temporal_fuse(traj_tokens, valid_frames)

        bg_base = self.bg_mlp(e.mean(dim=0, keepdim=True))
        bg_tokens = bg_base + self.bg_time_embed.to(device=device, dtype=dtype)

        k_fg = self.write_k_proj(traj_hat)
        v_fg = self.write_v_proj(traj_hat)
        k_bg = self.write_k_proj(bg_tokens)
        q_points = self.write_q_proj(e)

        # No chunk write-back by design: build [Nw, M, C] for the whole cube.
        k_fg_point = k_fg[:, t_local, :].permute(1, 0, 2)
        v_fg_point = v_fg[:, t_local, :].permute(1, 0, 2)
        k_bg_point = k_bg[t_local]

        score_bg = (q_points * k_bg_point).sum(dim=1, keepdim=True) / scale
        score_fg = (q_points[:, None, :] * k_fg_point).sum(dim=-1) / scale
        score_all = torch.cat([score_bg, score_fg], dim=1)
        attn_all = torch.softmax(score_all.float(), dim=1).to(dtype)

        fg_weight = attn_all[:, 1:]
        context = (fg_weight[..., None] * v_fg_point).sum(dim=1)
        score_out = fg_weight.sum(dim=1, keepdim=True)

        delta = self.out_mlp(torch.cat([residual_w, context], dim=1))
        out = residual_w + self.alpha * delta
        return out, score_out

    def forward(self, x):
        residual = x.features
        features = x.features
        indices = x.indices
        n_points = features.shape[0]

        out = torch.empty_like(residual)
        score = features.new_empty((n_points, 1))
        if n_points == 0:
            return out, score

        cube_keys = self._cube_keys(indices)
        sorted_keys, sort_idx = torch.sort(cube_keys)
        counts = torch.unique_consecutive(sorted_keys, return_counts=True)[1]

        start = 0
        for count in counts.tolist():
            end = start + count
            point_idx = sort_idx[start:end]
            out_w, score_w = self.forward_cube(
                residual[point_idx],
                features[point_idx],
                indices[point_idx],
            )
            out[point_idx] = out_w
            score[point_idx] = score_w
            start = end

        return out, score


SparseSymmetricCosineAttention = SparseTrajectoryTokenModule
