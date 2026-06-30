import torch
import torch.nn as nn


class TripletMotionConsistencySparseConv(nn.Module):
    """稀疏时空点的三帧运动一致性更新模块。

    改动点：
    1. top-k 邻居搜索改为真正的窗口内查找，不再对整帧候选点做 dense pairwise distance。
    2. 去掉 GPU -> CPU 的 .item() / .cpu() / .tolist() 同步。
    3. 使用窗口 offset + hash key + searchsorted 做 CUDA 并行查找。
    4. 只对具有 previous-next 有效 triplet 的点计算三元特征与 FFN。
    5. prev / cur / next 特征先各自 projection。
    6. prev / next 分支使用共享相对位置 MLP，分别加入 Δ- / Δ+ embedding。
    7. cat 得到 triplet_feat，由 FFN 完成 3C -> C 降维。
    8. FFN 后乘 scalar pos_gate。
    9. BatchNorm1d 改为 LayerNorm。
    10. 去掉 self.alpha 对输出的作用。
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
        **kwargs,
    ):
        super().__init__()
        del kernel_size, chunk_size, kwargs

        self.in_channels = in_channels

        # 保留 alpha 属性以兼容旧接口，但 forward 中不再使用。
        self.alpha = alpha

        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        self.window_size = int(window_size)
        self.window_radius = self.window_size // 2
        self.pos_scale = float(pos_scale)
        self.valid_norm = bool(valid_norm)
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = int(round(in_channels * float(hidden_ratio)))

        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)

        # scalar gate：仍然使用完整 triplet 运动输入 [Δ-, Δ+, acc]，维度为 6。
        self.pos_mlp = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, 1),
        )

        # prev / next 分支共享同一个相对位置 embedding MLP：
        # prev 分支输入 Δ- = current - previous。
        # next 分支输入 Δ+ = next - current。
        self.rel_pos_mlp = nn.Sequential(
            nn.Linear(2, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, in_channels),
        )

        # prev / cur / next 相加后仍为 C 维，由 FFN 更新。
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )

        # 用 LayerNorm 替代 BatchNorm1d，避免无效点污染 batch 统计。
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU(inplace=True)

        # 初始位置门控保持中性：2 * sigmoid(0) = 1。
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

        # 初始共享相对位置 embedding 为 0，避免刚加入时破坏原始特征分支。
        nn.init.zeros_(self.rel_pos_mlp[-1].weight)
        nn.init.zeros_(self.rel_pos_mlp[-1].bias)

        # 预先构造窗口内 offset，并按距离从近到远排序。
        # 后续只需要保留前 K 个有效 offset，即是真正窗口内 top-k。
        coords = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            dtype=torch.long,
        )
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1)
        dist2 = offsets[:, 0].square() + offsets[:, 1].square()
        order = torch.argsort(dist2, stable=True)

        self.register_buffer("window_offsets", offsets[order], persistent=False)
        self.register_buffer("window_dist2", dist2[order], persistent=False)

    @staticmethod
    def _encode_key(b, t, y, x, t_stride, y_stride, x_stride):
        """把 batch/time/y/x 编码成一维 key，用于 CUDA 上的排序和查找。"""
        return (((b * t_stride + t) * y_stride + y) * x_stride + x)

    def _find_triplet_neighbors(self, indices):
        """真正的窗口内 top-k 邻居查找。

        输入:
            indices: [N, 4], 格式为 [batch, time, y, x]

        输出:
            idx_prev:  [N, K]
            idx_next:  [N, K]
            mask_prev: [N, K]
            mask_next: [N, K]

        计算方式:
            1. 对所有稀疏点坐标编码成 hash key。
            2. 对 key 排序。
            3. 对每个点并行枚举窗口内所有 offset。
            4. 使用 searchsorted 查找 previous / next 帧中对应坐标是否存在。
            5. offset 已按距离排序，因此前 K 个有效匹配就是窗口内 top-k。
        """
        device = indices.device
        n_points = indices.shape[0]
        k = self.topk

        coords = indices.long()
        b = coords[:, 0]
        t = coords[:, 1]
        y = coords[:, 2]
        x = coords[:, 3]

        # stride 全部保留为 CUDA tensor，避免 .item() 触发 GPU/CPU 同步。
        t_stride = t.max() + self.temporal_dilation + 3
        y_stride = y.max() + self.window_radius + 3
        x_stride = x.max() + self.window_radius + 3

        all_keys = self._encode_key(
            b=b,
            t=t,
            y=y,
            x=x,
            t_stride=t_stride,
            y_stride=y_stride,
            x_stride=x_stride,
        )

        # 对所有已有 sparse 坐标建立可搜索的一维 key 表。
        sorted_keys, sorted_order = torch.sort(all_keys)

        offsets = self.window_offsets.to(device=device)
        off_y = offsets[:, 0]
        off_x = offsets[:, 1]
        n_offsets = offsets.shape[0]

        # 同时构造 previous 和 next 两个方向的目标坐标。
        target_t = torch.stack(
            [
                t - self.temporal_dilation,
                t + self.temporal_dilation,
            ],
            dim=1,
        ).view(n_points, 2, 1)

        target_b = b.view(n_points, 1, 1)
        target_y = y.view(n_points, 1, 1) + off_y.view(1, 1, n_offsets)
        target_x = x.view(n_points, 1, 1) + off_x.view(1, 1, n_offsets)

        target_keys = self._encode_key(
            b=target_b,
            t=target_t,
            y=target_y,
            x=target_x,
            t_stride=t_stride,
            y_stride=y_stride,
            x_stride=x_stride,
        )

        flat_target_keys = target_keys.reshape(-1)

        # CUDA 并行二分查找：每个目标窗口坐标查一次是否存在。
        pos = torch.searchsorted(sorted_keys, flat_target_keys)
        pos_safe = pos.clamp_max(n_points - 1)

        matched_keys = sorted_keys[pos_safe]
        matched_mask = (pos < n_points) & (matched_keys == flat_target_keys)
        matched_index = sorted_order[pos_safe]

        matched_mask = matched_mask.view(n_points, 2, n_offsets)
        matched_index = matched_index.view(n_points, 2, n_offsets)

        # offsets 已经按距离升序排列。
        # 对每个点、每个方向，前 K 个有效匹配就是窗口内 top-k。
        rank = matched_mask.long().cumsum(dim=-1) - 1
        keep = matched_mask & (rank < k)

        out_idx = torch.zeros(n_points, 2, k, dtype=torch.long, device=device)
        out_mask = torch.zeros(n_points, 2, k, dtype=torch.bool, device=device)

        row_ids = torch.arange(n_points, device=device).view(n_points, 1, 1)
        row_ids = row_ids.expand(n_points, 2, n_offsets)

        dir_ids = torch.arange(2, device=device).view(1, 2, 1)
        dir_ids = dir_ids.expand(n_points, 2, n_offsets)

        out_idx[row_ids[keep], dir_ids[keep], rank[keep]] = matched_index[keep]
        out_mask[row_ids[keep], dir_ids[keep], rank[keep]] = True

        idx_prev = out_idx[:, 0]
        idx_next = out_idx[:, 1]
        mask_prev = out_mask[:, 0]
        mask_next = out_mask[:, 1]

        return idx_prev, idx_next, mask_prev, mask_next

    def forward(self, x):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features

        n_points, channels = features.shape
        k = self.topk

        # 1. 真正窗口内 top-k previous / next 邻居查找。
        idx_prev, idx_next, mask_prev, mask_next = self._find_triplet_neighbors(indices)

        # 2. 构造 previous-next triplet 的有效 mask。
        # mask_pair[i, u, v] 表示第 i 个点的第 u 个 prev 和第 v 个 next 是否同时有效。
        mask_pair = mask_prev.unsqueeze(2) & mask_next.unsqueeze(1)
        has_pair = mask_pair.flatten(1).any(dim=1)

        # 只对存在合法 triplet 的点计算后续三元特征，减少无效 FFN 计算。
        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)

        idx_prev_v = idx_prev[valid_idx]
        idx_next_v = idx_next[valid_idx]
        mask_pair_v = mask_pair[valid_idx]

        n_valid = valid_idx.shape[0]

        # 3. 构造基于运动一致性的 6 维位置编码。
        # Δ- = current - previous
        # Δ+ = next - current
        # acc = Δ+ - Δ- = next - 2 * current + previous
        spatial = indices[:, 2:4].to(features.dtype)

        s_cur = spatial[valid_idx].view(n_valid, 1, 1, 2)
        s_prev = spatial[idx_prev_v].unsqueeze(2)
        s_next = spatial[idx_next_v].unsqueeze(1)

        s_minus = (s_cur - s_prev).expand(-1, -1, k, -1)
        s_plus = (s_next - s_cur).expand(-1, k, -1, -1)
        accel = s_plus - s_minus

        pos_scale = max(self.pos_scale, 1e-6)
        s_minus_norm = s_minus / pos_scale
        s_plus_norm = s_plus / pos_scale

        motion_input = torch.cat(
            [s_minus, s_plus, accel],
            dim=-1,
        ) / pos_scale

        # 4. 位置编码生成 scalar gate。
        # 初始时 pos_gate ≈ 1，训练后根据运动一致性调制 FFN 后的 triplet message。
        pos_score = self.pos_mlp(motion_input.reshape(-1, 6))
        pos_score = pos_score.view(n_valid, k, k, 1)

        pos_gate = 2.0 * torch.sigmoid(pos_score)
        pos_gate = pos_gate * mask_pair_v.unsqueeze(-1).to(pos_gate.dtype)

        # 5. prev / next 原始特征先加入相对位置 embedding，再做 point-wise projection。
        prev_branch = features[idx_prev_v.reshape(-1)].view(n_valid, k, 1, channels)
        prev_branch = prev_branch.expand(-1, -1, k, -1)
        next_branch = features[idx_next_v.reshape(-1)].view(n_valid, 1, k, channels)
        next_branch = next_branch.expand(-1, k, -1, -1)

        prev_pos_emb = self.rel_pos_mlp(s_minus_norm.reshape(-1, 2))
        prev_pos_emb = prev_pos_emb.view(n_valid, k, k, channels)

        next_pos_emb = self.rel_pos_mlp(s_plus_norm.reshape(-1, 2))
        next_pos_emb = next_pos_emb.view(n_valid, k, k, channels)

        prev_branch = self.prev_proj((prev_branch + prev_pos_emb).reshape(-1, channels))
        prev_branch = prev_branch.view(n_valid, k, k, channels)

        cur_branch = self.cur_proj(features[valid_idx]).view(n_valid, 1, 1, channels)
        cur_branch = cur_branch.expand(-1, k, k, -1)

        next_branch = self.next_proj((next_branch + next_pos_emb).reshape(-1, channels))
        next_branch = next_branch.view(n_valid, k, k, channels)

        # 6. 保留 587 版 add 约束：三支 projection 后相加，再由 FFN 更新。
        triplet_feat = prev_branch + cur_branch + next_branch

        # 7. FFN 完成 C -> C 更新。
        triplet_msg = self.ffn(triplet_feat.reshape(-1, channels))
        triplet_msg = triplet_msg.view(n_valid, k, k, channels)

        # 8. FFN 后乘 scalar pos_gate。
        # pos_gate 已包含有效 triplet mask。
        triplet_msg = triplet_msg * pos_gate.to(triplet_msg.dtype)

        # 9. 对 K × K 个 triplet message 聚合。
        msg_sum = triplet_msg.sum(dim=2).sum(dim=1)

        if self.valid_norm:
            denom = mask_pair_v.flatten(1).sum(dim=1)
            denom = denom.to(features.dtype).clamp_min(1.0).unsqueeze(1)
        else:
            denom = features.new_full((n_valid, 1), float(k * k))

        update_v = msg_sum / denom

        # 12. 原有 norm 不变：LayerNorm + ReLU。
        update_v = self.act(self.norm(update_v))

        # 13. 写回完整 N 个点的 update。
        update = features.new_zeros(n_points, channels)
        update[valid_idx] = update_v

        # 14. score 表示有效 triplet 的平均位置门控强度。
        score_v = pos_gate.sum(dim=2).sum(dim=1) / denom.to(pos_gate.dtype)

        score = features.new_zeros(n_points, 1)
        score[valid_idx] = score_v

        # 15. 去掉 self.alpha 的作用，直接 residual update。
        output = input_features.clone()
        # masked 点保持 identity；
        # valid_idx 中的点使用 0.5 * input + 0.5 * update。
        output[valid_idx] = 0.5 * input_features[valid_idx] + 0.5 * update_v
        return output, score