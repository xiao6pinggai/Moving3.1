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

class GatedTemporalFusion(nn.Module):
    """轨迹 token 的轻量时序融合模块。

    输入形状为 [M, L, C]：
    - M：单个空间 cube 内的轨迹槽数量
    - L：帧数
    - C：特征通道数

    这里故意不用完整 TransformerEncoderLayer，而是用更轻的 depthwise Conv1d
    沿时间维混合每条轨迹，再用门控控制注入多少时序上下文。
    """

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Linear(channels, channels)
        self.gate = nn.Linear(channels, channels)

    def forward(self, h, valid_mask):
        # valid_mask 用来屏蔽空帧轨迹 token。空帧在 read attention 中是零 token，
        # 不应该通过时序卷积污染相邻帧。
        valid = valid_mask[None, :, None].to(h.dtype)
        h = h * valid

        # depthwise 时序卷积：每个通道只沿帧维混合。
        # Conv1d 需要 [M, C, L]，所以这里先转置再转回来。
        z = self.dw(h.transpose(1, 2)).transpose(1, 2)
        z = self.pw(z)

        # 残差门控融合，让模块初始化时更保守。
        out = h + torch.sigmoid(self.gate(h)) * z
        return out * valid


class FrameRestrictedSparseTrajectoryTransformer(nn.Module):
    """帧内读取限制的稀疏轨迹 token 特征增强模块。

    模块保持稀疏点坐标和数量不变，只更新特征：
    x.features [N, C] -> enhanced_features [N, C].

    在每个空间 cube 内，每一帧先独立读取成 M 个轨迹 token；
    然后轨迹 token 沿时间维融合；最后每个点只从自身所在帧的 M 个轨迹
    token 中写回上下文。
    """

    def __init__(
        self,
        in_channels,
        num_frames=5,
        num_traj=16,
        window_size=256,
        num_heads=4,
        alpha=0.5,
        conv=None,
        indice_key="subm0",
        write_chunk_size=4096,
        use_temporal_transformer=True,
        kernel_size=5,
        stride=1,
        use_qkv=False,
        use_biqkv=False,
        use_maxpool=False,
        temporal_dilation=1,
        **kwargs,
    ):
        super().__init__()
        self.channels = in_channels
        self.num_frames = num_frames
        self.num_traj = num_traj
        self.window_size = window_size
        self.alpha = alpha
        self.write_chunk_size = write_chunk_size

        # 可学习的轨迹槽身份和帧身份。所有 cube 共享同一组 M 个轨迹身份；
        # cube 自身的信息稍后通过 global_proj(mean(E)) 注入。
        self.traj_embed = nn.Parameter(torch.empty(num_traj, in_channels))
        self.time_embed = nn.Parameter(torch.empty(num_frames, in_channels))

        # 点 token 构造：特征投影 + 局部 (t, y, x) 位置编码。
        # batch id 只用于分组，不进入位置编码。
        self.feat_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        self.global_proj = nn.Linear(in_channels, in_channels) # 全局均值池化

        # read attention：轨迹 query 只读取同一帧的稀疏点 token。
        # 这样避免原始 STT 中较重的 [M * L, Nw] attention。
        self.read_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        if use_temporal_transformer:
            self.temporal_fusion = nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=4 * in_channels,
                batch_first=True,
            )
            self.use_temporal_transformer = True
        else:
            self.temporal_fusion = GatedTemporalFusion(in_channels)
            self.use_temporal_transformer = False

        # write-back attention：每个点只查询自身帧的 M 个轨迹 token。
        # 为了速度，先对轨迹 token 做 K/V 投影，再按点的帧号 gather。
        self.write_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.write_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        # 残差门控输出，沿用 cos_update_v12 更稳定的更新方式。
        # 模块只预测增量 delta，不直接覆盖原特征。
        self.update_proj = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        self.gate_proj = nn.Linear(2 * in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.traj_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        # 负 gate bias 让初始增强幅度较小，插入已有 backbone 时更稳。
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _cube_keys(self, indices):
        # 按 batch 和粗空间窗口分组。时间不参与 cube key：
        # 一个 cube 包含同一空间窗口内的所有帧。
        # 这里用整数打包 key，实验代码里简单且速度足够。
        b = indices[:, 0].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        wy = torch.div(y, self.window_size, rounding_mode='floor')
        wx = torch.div(x_coord, self.window_size, rounding_mode='floor')
        return b * 10_000_000_000 + wy * 100_000 + wx

    def _position_encoding(self, indices):
        # 空间 cube 内的局部归一化坐标。
        # 位置先用 fp32 计算，forward_cube 中再转成特征 dtype。
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

    def _temporal_fuse(self, h, valid_frames):
        # valid_frames 标记当前 cube 中哪些帧至少有一个稀疏点。
        # 空帧 token 是零，融合后也应该保持为零。
        if not self.use_temporal_transformer:
            return self.temporal_fusion(h, valid_frames)

        key_padding_mask = ~valid_frames[None, :].expand(h.shape[0], -1)
        out = self.temporal_fusion(h, src_key_padding_mask=key_padding_mask)
        return out * valid_frames[None, :, None].to(out.dtype)

    def forward_cube(self, residual_w, features_w, indices_w):
        # 处理单个空间 cube。features_w / indices_w 包含该 cube 内所有帧的点，
        # 不包含其他 cube 或其他 batch 的点。
        n_points, channels = features_w.shape
        device = features_w.device
        dtype = features_w.dtype
        m = self.num_traj
        l_count = self.num_frames
        scale = math.sqrt(channels)

        # 构造稀疏点 token：E = 特征投影 + 局部位置编码。
        pos = self._position_encoding(indices_w).to(device=device, dtype=dtype)
        e = self.feat_proj(features_w) + self.pos_mlp(pos)

        # 构造带 cube 条件的轨迹 query。g 给每个 cube 一个内容相关偏置，
        # 同时轨迹身份 traj_embed 仍然在所有 cube 间共享。
        g = self.global_proj(e.mean(dim=0, keepdim=True))
        q_traj = self.traj_embed[:, None, :] + self.time_embed[None, :, :] + g[None, :, :]

        t_local = indices_w[:, 1].long().clamp(0, l_count - 1)

        # read 侧投影只做一次。下面逐帧循环时只切片对应帧点，
        # 避免对同一个点 token 重复做 Linear。
        q_read = self.read_q_proj(q_traj)
        k_all = self.read_k_proj(e)
        v_all = self.read_v_proj(e)

        # H 保存每一帧的 M 个轨迹 token。由于每帧稀疏点数 Nl 不同，
        # 简洁实验版采用逐帧循环，每次 attention 形状为 [M, Nl]。
        h = e.new_zeros((m, l_count, channels))
        valid_frames = torch.zeros(l_count, device=device, dtype=torch.bool)

        for frame_id in range(l_count):
            frame_mask = t_local == frame_id
            if not frame_mask.any():
                continue

            # 帧内限制 read：当前帧的轨迹槽只关注当前帧稀疏点，
            # 不读取 cube 内其他时间帧的点。
            score = q_read[:, frame_id, :] @ k_all[frame_mask].T / scale
            attn = torch.softmax(score.float(), dim=1).to(dtype)
            h[:, frame_id, :] = attn @ v_all[frame_mask]
            valid_frames[frame_id] = True

        # 每条轨迹槽沿时间维融合。这里不做不同轨迹槽之间的 attention。
        h_hat = self._temporal_fuse(h, valid_frames)

        # 先投影轨迹 K/V，再按点 gather。这样比对重复后的 [Nw, M, C]
        # 张量做 Linear 更快、更省显存。
        k_hat = self.write_k_proj(h_hat)
        v_hat = self.write_v_proj(h_hat)
        q_points = self.write_q_proj(e)

        # 分块写回，限制 K/V gather 的峰值显存：
        # 每个 chunk 构造 [chunk, M, C]，而不是一次性构造 [Nw, M, C]。
        context = e.new_empty((n_points, channels))
        score_out = e.new_empty((n_points, 1))
        chunk_size = max(int(self.write_chunk_size), 1)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            t_chunk = t_local[start:end]
            q_chunk = q_points[start:end]

            # 按每个点自身帧号 gather 对应的 M 个轨迹 token。
            # permute 后形状为 [chunk, M, C]。
            k_chunk = k_hat[:, t_chunk, :].permute(1, 0, 2)
            v_chunk = v_hat[:, t_chunk, :].permute(1, 0, 2)

            # 点到轨迹的写回 attention，只在 M 个轨迹槽上 softmax。
            score = (q_chunk[:, None, :] * k_chunk).sum(dim=-1) / scale
            beta = torch.softmax(score.float(), dim=1).to(dtype)
            context[start:end] = (beta[..., None] * v_chunk).sum(dim=1)

            # score_out 主要用于可视化或调试，作用类似早期 cosine MFE 返回的 score。
            score_out[start:end] = beta.max(dim=1, keepdim=True).values

        # 保守的残差增强。residual_w 是模块输入原特征，
        # features_w 是本模块中用于构造 token 的点特征。
        update_input = torch.cat([features_w, context], dim=1)
        gate = torch.sigmoid(self.gate_proj(update_input))
        delta = self.update_proj(update_input)
        out = residual_w + self.alpha * gate * delta
        return out, score_out

    def forward(self, x):
        # 保持稀疏结构不变，只返回增强后的 features 和 point score，
        # 与 cosv10/v11/v12 的调用约定一致。
        residual = x.features
        indices = x.indices
        features = x.features
        n_points = features.shape[0]

        out = torch.empty_like(residual)
        score = features.new_empty((n_points, 1))
        if n_points == 0:
            return out, score

        # 按 cube key 排序，使同一个 cube 的点成为连续片段。
        # 这样处理每个 cube 时不用为每个 unique key 构造全局 bool mask。
        cube_keys = self._cube_keys(indices)
        sorted_keys, sort_idx = torch.sort(cube_keys)
        counts = torch.unique_consecutive(sorted_keys, return_counts=True)[1]

        start = 0
        for count in counts.tolist():
            end = start + count
            point_idx = sort_idx[start:end]

            # 每个 cube 独立处理：attention 不跨 batch，也不跨空间窗口。
            out_w, score_w = self.forward_cube(
                residual[point_idx],
                features[point_idx],
                indices[point_idx],
            )
            out[point_idx] = out_w
            score[point_idx] = score_w
            start = end

        return out, score


SparseSymmetricCosineAttention = FrameRestrictedSparseTrajectoryTransformer
