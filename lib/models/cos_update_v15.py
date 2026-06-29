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
    """v15 稀疏轨迹 token 特征增强模块。

    模块保持稀疏拓扑不变：
    x.features [N, C] -> enhanced_features [N, C]。

    每个 cube 表示跨越全部时间帧的一个 256 x 256 空间窗口。v15 在每个
    cube 内部依次执行：局部稀疏时空注意力、前景轨迹 token 读取与融合、
    全局池化背景 token 竞争，以及整块点集的写回更新。
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
        """初始化 v15 所需的轨迹 token、位置编码和读写注意力子模块。"""
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
        self.local_temporal_radius = temporal_dilation
        self.local_spatial_radius = kernel_size // 2

        # 兼容旧版 MFE 构建器保留下来的参数，v15 只复用其中一部分配置含义。
        self.conv = conv
        self.indice_key = indice_key
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_qkv = use_qkv
        self.use_biqkv = use_biqkv
        self.use_maxpool = use_maxpool
        self.temporal_dilation = temporal_dilation
        self.write_chunk_size = write_chunk_size

        # 将 (t, y, x) 编码成一维 key，便于快速检索局部邻域点。
        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self._build_local_offsets()

        # 可学习的前景轨迹 token、前景时间编码、背景时间编码。
        self.traj_embed = nn.Parameter(torch.empty(num_traj, in_channels))
        self.time_embed = nn.Parameter(torch.empty(num_frames, in_channels))
        self.bg_time_embed = nn.Parameter(torch.empty(num_frames, in_channels))

        # 点特征投影与归一化位置编码，用于构造初始点 token。
        self.feat_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        # 局部稀疏注意力的 Q/K/V 与相对位置偏置网络。
        self.local_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.local_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.local_v_proj = nn.Linear(in_channels, in_channels, bias=False)
        local_hidden = max(8, min(32, in_channels))
        self.local_rel_bias = nn.Sequential(
            nn.Linear(3, local_hidden),
            nn.GELU(),
            nn.Linear(local_hidden, 1),
        )

        # 轨迹 token 从点 token 读取信息时所需的条件投影与 Q/K/V。
        self.global_proj = nn.Linear(in_channels, in_channels)
        self.read_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        # 沿时间维融合每条轨迹在不同帧上的响应。
        self.temporal_fusion = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=4 * in_channels,
            batch_first=True,
        )

        # 背景 token 由 cube 全局均值生成，再与时间编码结合。
        self.bg_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        # 将前景/背景 token 重新写回点特征时使用的 Q/K/V。
        self.write_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        # 将原始残差与聚合上下文拼接后映射为最终增量。
        self.out_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

        self.reset_parameters()

    def _build_local_offsets(self):
        """构建局部时空邻域偏移表，供稀疏注意力查询邻点。"""
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
        """初始化可学习参数，并让部分投影在初始阶段接近恒等映射。"""
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
        """根据 batch 和窗口坐标生成 cube key，用于把点划分到不同空间块。"""
        b = indices[:, 0].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        wy = torch.div(y, self.window_size, rounding_mode='floor')
        wx = torch.div(x_coord, self.window_size, rounding_mode='floor')
        return b * 10_000_000_000 + wy * 100_000 + wx

    def _position_encoding(self, indices):
        """将点的时间和窗口内相对位置归一化为 3 维坐标编码。"""
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
        """对稀疏边集合按源点分组做 softmax，得到归一化注意力权重。"""
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
        """在局部时空邻域内执行稀疏注意力，增强点 token 的局部上下文。"""
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
        """沿时间维融合轨迹 token，只在存在点的帧上保留有效响应。"""
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
        """处理单个 cube 内的点，完成轨迹读写并输出增强后的点特征。"""
        n_points, channels = features_w.shape
        device = features_w.device
        dtype = features_w.dtype
        m_count = self.num_traj
        l_count = self.num_frames
        scale = math.sqrt(channels)

        # 1) 点特征叠加位置编码，并先做一次局部稀疏时空增强。
        pos = self._position_encoding(indices_w).to(device=device, dtype=dtype)
        point_tokens = self.feat_proj(features_w) + self.pos_mlp(pos)
        e = self._local_sparse_attention(point_tokens, indices_w)

        # 2) 依据 cube 全局语义生成当前 cube 的前景轨迹查询 token。
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

        # 3) 每一帧内让多条轨迹 token 从该帧点集中读取前景响应。
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

        # 4) 对轨迹 token 做跨帧融合，补充时间一致性。
        traj_hat = self._temporal_fuse(traj_tokens, valid_frames)

        # 5) 构造背景 token，并与前景轨迹 token 一起准备写回点特征。
        bg_base = self.bg_mlp(e.mean(dim=0, keepdim=True))
        bg_tokens = bg_base + self.bg_time_embed.to(device=device, dtype=dtype)

        k_fg = self.write_k_proj(traj_hat)
        v_fg = self.write_v_proj(traj_hat)
        k_bg = self.write_k_proj(bg_tokens)
        q_points = self.write_q_proj(e)

        # 整个 cube 一次性写回，不再分 chunk 处理。
        k_fg_point = k_fg[:, t_local, :].permute(1, 0, 2)
        v_fg_point = v_fg[:, t_local, :].permute(1, 0, 2)
        k_bg_point = k_bg[t_local]

        # 6) 每个点与背景 token、多个前景轨迹 token 竞争注意力权重。
        score_bg = (q_points * k_bg_point).sum(dim=1, keepdim=True) / scale
        score_fg = (q_points[:, None, :] * k_fg_point).sum(dim=-1) / scale
        score_all = torch.cat([score_bg, score_fg], dim=1)
        attn_all = torch.softmax(score_all.float(), dim=1).to(dtype)

        fg_weight = attn_all[:, 1:]
        context = (fg_weight[..., None] * v_fg_point).sum(dim=1)
        score_out = fg_weight.sum(dim=1, keepdim=True)

        # 7) 将前景上下文写回残差特征，输出当前 cube 的增强结果。
        delta = self.out_mlp(torch.cat([residual_w, context], dim=1))
        out = residual_w + self.alpha * delta
        return out, score_out

    def forward(self, x):
        """按 cube 分组遍历稀疏点云，对每个空间块独立执行前景轨迹增强。"""
        residual = x.features
        features = x.features
        indices = x.indices
        n_points = features.shape[0]

        out = torch.empty_like(residual)
        score = features.new_empty((n_points, 1))
        if n_points == 0:
            return out, score

        # 1) 先按空间窗口生成 cube key，相同 key 的点进入同一个 cube。
        cube_keys = self._cube_keys(indices)
        sorted_keys, sort_idx = torch.sort(cube_keys)
        counts = torch.unique_consecutive(sorted_keys, return_counts=True)[1]

        # 2) 逐个 cube 调用 forward_cube，最后再写回原始点顺序。
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
