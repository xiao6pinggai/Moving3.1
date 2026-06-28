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


class ObjectCentricAssociationTokenFusion(nn.Module):
    """面向小目标稀疏点特征的 object-centric token 融合模块。

    输入和输出都保持 spconv 的稀疏拓扑不变：
    x.features [N, C] -> enhanced_features [N, C]

    核心思路：
    1. 按 batch + 空间窗口把稀疏点分成多个 cube。
    2. 在每个 cube 的每一帧里，用 num_traj 个可学习 object slots 聚合点特征。
    3. 让同一个 cube 内的 object tokens 沿时间维做 attention 融合。
    4. 再把融合后的 object tokens 写回每个稀疏点。
    """

    def __init__(
        self,
        in_channels,
        num_frames=5,
        num_traj=16,
        topk=2,
        window_size=256,
        num_heads=4,
        alpha=0.5,
        assignment_temperature=0.7,
        min_mass_ratio=0.05,
        conv=None,
        indice_key="subm0",
        kernel_size=5,
        stride=1,
        use_qkv=False,
        use_biqkv=False,
        use_maxpool=False,
        temporal_dilation=1,
        **kwargs,
    ):
        super().__init__()
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if topk > num_traj:
            raise ValueError(f"topk must be <= num_traj, got topk={topk}, num_traj={num_traj}")
        if assignment_temperature <= 0:
            raise ValueError(
                f"assignment_temperature must be > 0, got {assignment_temperature}"
            )
        if in_channels % num_heads != 0:
            raise ValueError(
                f"in_channels must be divisible by num_heads, got "
                f"in_channels={in_channels}, num_heads={num_heads}"
            )

        self.channels = in_channels
        self.num_frames = num_frames
        self.num_traj = num_traj
        self.topk = topk
        self.window_size = window_size
        self.alpha = alpha
        self.assignment_temperature = assignment_temperature
        self.min_mass_ratio = min_mass_ratio

        # 这些参数是为了兼容 cosv10/v11/v12 的构造接口。
        # 当前 v14 主要使用 token 聚合，不额外使用 conv/kernel temporal_dilation。
        self.conv = conv
        self.indice_key = indice_key
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_qkv = use_qkv
        self.use_biqkv = use_biqkv
        self.use_maxpool = use_maxpool
        self.temporal_dilation = temporal_dilation

        # slot_embed: M 个可学习 object slots，表示局部窗口内最多聚合 M 个目标假设。
        # time_embed: 给不同帧的 object slots 加时间身份。
        # temporal_pos_embed + slot_id_embed: temporal_attn 中使用的位置/slot 身份编码。
        self.slot_embed = nn.Parameter(torch.empty(num_traj, in_channels))
        self.time_embed = nn.Parameter(torch.empty(num_frames, in_channels))
        self.temporal_pos_embed = nn.Parameter(torch.empty(num_frames, in_channels))
        self.slot_id_embed = nn.Parameter(torch.empty(num_traj, in_channels))

        # 点特征先投影成 token 表达；位置 MLP 注入局部 (t, y, x) 信息。
        self.feat_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )
        self.read_q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_k_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.read_v_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.slot_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.slot_id_embed, std=0.02)

        nn.init.eye_(self.feat_proj.weight)
        nn.init.eye_(self.read_v_proj.weight)
        nn.init.xavier_uniform_(self.read_q_proj.weight)
        nn.init.xavier_uniform_(self.read_k_proj.weight)

        nn.init.xavier_uniform_(self.pos_mlp[0].weight)
        nn.init.zeros_(self.pos_mlp[0].bias)
        nn.init.zeros_(self.pos_mlp[2].weight)
        nn.init.zeros_(self.pos_mlp[2].bias)

        # 初始时让 temporal attention 的输出为 0，避免刚开始训练时扰动过大。
        nn.init.zeros_(self.temporal_attn.out_proj.weight)
        nn.init.zeros_(self.temporal_attn.out_proj.bias)

    def _cube_keys(self, indices):
        """把稀疏点映射到 batch + 空间窗口 key。

        indices 形状为 [N, 4]，约定为 [batch, t, y, x]。
        这里不把 t 放进 key，因为同一个空间窗口内的多帧点需要一起做时间融合。
        """
        b = indices[:, 0].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        wy = torch.div(y, self.window_size, rounding_mode='floor')
        wx = torch.div(x_coord, self.window_size, rounding_mode='floor')
        return b * 10_000_000_000 + wy * 100_000 + wx

    def _position_encoding(self, indices):
        """构造每个稀疏点的局部归一化位置编码 [t_norm, y_norm, x_norm]。"""
        dtype = torch.float32
        t = indices[:, 1].to(dtype)
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

    def _temporal_fuse(self, tokens, valid):
        """单个 cube 的时间融合参考实现。

        tokens: [L, M, C]，L=num_frames，M=num_traj。
        valid:  [L, M]，标记哪些帧/slot 真正聚合到了足够点。

        这个函数保留给 forward_cube 使用，便于和批量 forward 做等价验证。
        训练主路径现在走 _temporal_fuse_batched。
        """
        l_count, m_count, channels = tokens.shape
        values = tokens.reshape(l_count * m_count, channels)
        valid_flat = valid.reshape(l_count * m_count)
        valid_count = int(valid_flat.sum().item())

        out = values.new_zeros(values.shape)
        if valid_count == 0:
            return out.reshape(l_count, m_count, channels)
        if valid_count == 1:
            out[valid_flat] = values[valid_flat]
            return out.reshape(l_count, m_count, channels)

        pos = (
            self.temporal_pos_embed[:, None, :]
            + self.slot_id_embed[None, :, :]
        ).reshape(l_count * m_count, channels)

        attn_in = (values + pos)[valid_flat].unsqueeze(0)
        attn_out, _ = self.temporal_attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=False,
        )
        out[valid_flat] = values[valid_flat] + attn_out.squeeze(0)
        return out.reshape(l_count, m_count, channels)

    def _temporal_fuse_batched(self, tokens, valid):
        """批量版本的时间融合，避免 Python 按 cube 循环。

        tokens: [Bcube, L, M, C]
        valid:  [Bcube, L, M]

        做法是把每个 cube 的 L*M 个 object tokens 看成一条短序列，
        用 key_padding_mask 屏蔽无效 token，然后一次性送进 MultiheadAttention。
        """
        cube_count, l_count, m_count, channels = tokens.shape
        values = tokens.reshape(cube_count, l_count * m_count, channels)
        valid_flat = valid.reshape(cube_count, l_count * m_count)
        valid_counts = valid_flat.sum(dim=1)

        out = values.new_zeros(values.shape)
        single_mask = valid_counts == 1
        if bool(single_mask.any()):
            out[single_mask] = values[single_mask]

        attn_mask = valid_counts > 1
        if bool(attn_mask.any()):
            pos = (
                self.temporal_pos_embed[:, None, :]
                + self.slot_id_embed[None, :, :]
            ).reshape(l_count * m_count, channels)
            pos = pos.to(device=values.device, dtype=values.dtype)

            attn_in = values[attn_mask] + pos.unsqueeze(0)
            key_padding_mask = ~valid_flat[attn_mask]
            attn_out, _ = self.temporal_attn(
                attn_in,
                attn_in,
                attn_in,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            fused = values[attn_mask] + attn_out
            fused = fused * valid_flat[attn_mask].unsqueeze(-1).to(values.dtype)
            out[attn_mask] = fused

        return out.reshape(cube_count, l_count, m_count, channels)

    def _build_frame_tokens(self, q_frame, k_frame, v_frame, dtype):
        """旧的逐帧 token 聚合函数。

        q_frame: [M, C]，当前帧的 M 个 object slot query。
        k_frame/v_frame: [N_frame, C]，当前帧内的稀疏点 key/value。

        返回：
        token:      [M, C]，每个 slot 聚合后的 token。
        valid:      [M]，该 slot 是否聚合到足够质量/数量的点。
        assign_idx: [N_frame, topk]，每个点分配到的 topk slots。
        assign_val: [N_frame, topk]，对应分配权重。
        """
        n_frame = k_frame.shape[0]
        m_count = self.num_traj
        k_count = self.topk
        scale = math.sqrt(k_frame.shape[1])

        logits = q_frame @ k_frame.T / scale
        logits = logits / self.assignment_temperature
        assign = torch.softmax(logits.float(), dim=0).to(dtype)
        top_val, top_idx = torch.topk(assign, k=k_count, dim=0)

        assign_idx = top_idx.transpose(0, 1).contiguous()
        assign_val = top_val.transpose(0, 1).contiguous()

        slot_idx = assign_idx.reshape(-1)
        slot_w = assign_val.reshape(-1, 1)
        point_idx = torch.arange(
            n_frame,
            device=k_frame.device,
            dtype=torch.long,
        ).repeat_interleave(k_count)

        token_sum = v_frame.new_zeros((m_count, v_frame.shape[1]))
        mass = v_frame.new_zeros((m_count, 1))
        token_sum.index_add_(0, slot_idx, v_frame[point_idx] * slot_w)
        mass.index_add_(0, slot_idx, slot_w)

        token = token_sum / mass.clamp_min(1e-6)
        expected_mass = max(float(n_frame * k_count) / float(m_count), 1.0)
        min_mass = self.min_mass_ratio * expected_mass
        valid = mass.squeeze(1) > min_mass
        token = token * valid[:, None].to(dtype)

        return token, valid, assign_idx, assign_val

    def forward_cube(self, residual_w, features_w, indices_w):
        """旧版单 cube 前向，保留用于阅读和等价测试。

        这个函数逻辑更直观：先按帧循环生成 tokens，再做时间融合，最后按帧写回。
        但它会导致大量 Python 循环，训练速度慢；实际训练走下面的批量 forward。
        """
        n_points, channels = features_w.shape
        device = features_w.device
        dtype = features_w.dtype
        l_count = self.num_frames
        m_count = self.num_traj

        # 单 cube 参考路径：先为当前 cube 内所有点构造基础点 token E。
        pos = self._position_encoding(indices_w).to(device=device, dtype=dtype)
        e = self.feat_proj(features_w) + self.pos_mlp(pos)

        t_local = indices_w[:, 1].long()
        k_all = self.read_k_proj(e)
        v_all = self.read_v_proj(e)

        tokens = e.new_zeros((l_count, m_count, channels))
        valid = torch.zeros((l_count, m_count), device=device, dtype=torch.bool)
        assignments = []

        for frame_id in range(l_count):
            frame_idx = (t_local == frame_id).nonzero(as_tuple=False).squeeze(1)
            if frame_idx.numel() == 0:
                assignments.append(None)
                continue

            q_frame = self.slot_embed + self.time_embed[frame_id]
            q_frame = self.read_q_proj(q_frame.to(device=device, dtype=dtype))
            token, valid_frame, assign_idx, assign_val = self._build_frame_tokens(
                q_frame,
                k_all[frame_idx],
                v_all[frame_idx],
                dtype,
            )
            tokens[frame_id] = token
            valid[frame_id] = valid_frame
            assignments.append((frame_idx, assign_idx, assign_val))

        fused_tokens = self._temporal_fuse(tokens, valid)
        context = e.new_zeros((n_points, channels))
        score = e.new_zeros((n_points, 1))

        for frame_id, assignment in enumerate(assignments):
            if assignment is None:
                continue

            frame_idx, assign_idx, assign_val = assignment
            selected_tokens = fused_tokens[frame_id][assign_idx]
            context[frame_idx] = (
                selected_tokens * assign_val[..., None]
            ).sum(dim=1)
            score[frame_idx] = assign_val.max(dim=1, keepdim=True).values

        out = residual_w + self.alpha * context
        return out, score

    def forward(self, x):
        """批量前向：当前训练实际使用的高速路径。

        主要张量约定：
        N: 当前 sparse tensor 中的点数。
        C: 通道数。
        L: num_frames，也就是 opt.seqLen 传进来的帧数。
        M: num_traj，局部窗口内 object slots 数量。
        K: topk，每个点分配给 K 个 object slots。
        """
        residual = x.features
        features = x.features
        indices = x.indices
        n_points = features.shape[0]

        out = residual.clone()
        score = features.new_zeros((n_points, 1))
        if n_points == 0:
            return out, score

        device = features.device
        dtype = features.dtype
        l_count = self.num_frames
        m_count = self.num_traj
        k_count = self.topk
        channels = features.shape[1]
        scale = math.sqrt(channels)

        # 直接使用输入帧号；若越界，后续张量索引会抛出 PyTorch 原生错误。
        t_all = indices[:, 1].long()

        work_idx = torch.arange(n_points, device=device, dtype=torch.long)
        residual_w = residual
        features_w = features
        indices_w = indices
        t_local = t_all

        # cube_inverse: 每个点属于第几个空间 cube，范围 [0, cube_count)。
        # segment_ids: 把 (cube, frame) 压成一维编号，后面用它批量聚合每帧 tokens。
        cube_keys = self._cube_keys(indices_w)
        unique_cube_keys, cube_inverse = torch.unique(
            cube_keys,
            sorted=True,
            return_inverse=True,
        )
        cube_count = unique_cube_keys.shape[0]
        segment_ids = cube_inverse * l_count + t_local
        segment_count = cube_count * l_count

        # 批量主路径：E = 点特征投影 + 局部位置编码，是后续 read/write 的基础点 token。
        pos = self._position_encoding(indices_w).to(device=device, dtype=dtype)
        e = self.feat_proj(features_w) + self.pos_mlp(pos)
        k_all = self.read_k_proj(e)
        v_all = self.read_v_proj(e)

        # 为每一帧构造 M 个 object slot query，形状 [L, M, C]。
        # 每个点根据自己的帧号 t_local 取出对应帧的 M 个 query。
        q_frame = self.slot_embed[None, :, :] + self.time_embed[:, None, :]
        q_frame = self.read_q_proj(q_frame.to(device=device, dtype=dtype))
        q_points = q_frame[t_local]
        logits = (q_points * k_all[:, None, :]).sum(dim=-1) / scale
        logits = logits / self.assignment_temperature
        assign = torch.softmax(logits.float(), dim=1).to(dtype)
        assign_val, assign_idx = torch.topk(assign, k=k_count, dim=1)

        # target 把 (cube, frame, slot) 压成一维编号。
        # 通过 index_add_，所有点可以一次性累加到对应 object slot token 上。
        target = (segment_ids[:, None] * m_count + assign_idx).reshape(-1)
        point_idx = torch.arange(
            features_w.shape[0],
            device=device,
            dtype=torch.long,
        ).repeat_interleave(k_count)
        weights = assign_val.reshape(-1, 1)

        # token_sum/mass 分别累计每个 slot 的加权 value 和总权重。
        # 最终 tokens = token_sum / mass，得到 [cube, frame, slot, C] 的 object tokens。
        token_sum = v_all.new_zeros((segment_count * m_count, channels))
        mass = v_all.new_zeros((segment_count * m_count, 1))
        token_sum.index_add_(0, target, v_all[point_idx] * weights)
        mass.index_add_(0, target, weights)

        tokens = token_sum / mass.clamp_min(1e-6)
        # valid 用于过滤“几乎没有点贡献”的 slot，减少噪声 token 参与时间 attention。
        segment_sizes = torch.bincount(segment_ids, minlength=segment_count).to(dtype)
        expected_mass = torch.clamp(
            segment_sizes * float(k_count) / float(m_count),
            min=1.0,
        )
        min_mass = self.min_mass_ratio * expected_mass[:, None]
        valid = mass.view(segment_count, m_count) > min_mass

        tokens = tokens.view(cube_count, l_count, m_count, channels)
        valid = valid.view(cube_count, l_count, m_count)
        tokens = tokens * valid.unsqueeze(-1).to(dtype)

        # 时间融合后，每个 cube 内同一批 object slots 能感知跨帧信息。
        fused_tokens = self._temporal_fuse_batched(tokens, valid)
        flat_fused = fused_tokens.reshape(cube_count * l_count * m_count, channels)
        gather_idx = (segment_ids[:, None] * m_count + assign_idx).reshape(-1)
        selected_tokens = flat_fused[gather_idx].view(
            features_w.shape[0],
            k_count,
            channels,
        )
        # 写回：每个点只读取自己所在 (cube, frame) 的 topk slots，并按分配权重加权求和。
        context = (selected_tokens * assign_val.unsqueeze(-1)).sum(dim=1)

        # 保持稀疏坐标不变，只更新 features；score 主要用于可视化/调试响应强度。
        out[work_idx] = residual_w + self.alpha * context
        score[work_idx] = assign_val.max(dim=1, keepdim=True).values

        return out, score


SparseSymmetricCosineAttention = ObjectCentricAssociationTokenFusion
