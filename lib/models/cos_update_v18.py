import torch
import torch.nn as nn

class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """稀疏时空点的三帧运动一致性更新模块。

    改动点：
    1. top-k 邻居搜索改为真正的窗口内查找，并按 previous-next 成对运动一致性距离选择。
    2. 去掉 GPU -> CPU 的 .item() / .cpu() / .tolist() 同步。
    3. 使用窗口 offset + hash key + searchsorted 做 CUDA 并行查找。
    4. 只对具有 previous-next 有效 triplet 的点计算三元特征与 FFN。
    5. FFN 放在 “三元特征 * 位置门控” 之后。
    6. BatchNorm1d 改为 LayerNorm。
    7. 去掉 self.alpha 对输出的作用。
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
        self.topk = max(1, int(topk))
        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        self.pos_scale = float(pos_scale)
        self.valid_norm = bool(valid_norm)
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio)))) # 可以增加参数量
        pos_hidden = max(4, int(pos_hidden))

        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.pos_mlp = nn.Sequential( # 可以增加参数量
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, 1),
        )

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

        # 预先构造窗口内 offset；邻居最终按 previous-next 成对运动一致性距离取 top-k。
        coords = torch.arange(
            -self.window_radius,
            self.window_radius + 1,
            dtype=torch.long,
        )
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1)
        self.register_buffer("window_offsets", offsets, persistent=False)

        n_offsets = offsets.shape[0]
        op = offsets.view(n_offsets, 1, 2)
        on = offsets.view(1, n_offsets, 2)
        pair_cost = (op + on).square().sum(dim=-1).reshape(-1)
        pair_order = torch.argsort(pair_cost, stable=True)
        pair_prev_offset_id = torch.div(pair_order, n_offsets, rounding_mode="floor")
        pair_next_offset_id = pair_order - pair_prev_offset_id * n_offsets

        self.register_buffer("pair_prev_offset_id", pair_prev_offset_id, persistent=False)
        self.register_buffer("pair_next_offset_id", pair_next_offset_id, persistent=False)

    @staticmethod
    def _encode_key(b, t, y, x, t_stride, y_stride, x_stride):
        """把 batch/time/y/x 编码成一维 key，用于 CUDA 上的排序和查找。"""
        return (((b * t_stride + t) * y_stride + y) * x_stride + x)

    def _find_triplet_neighbors(self, indices, features):
        """窗口内基于 previous-next 成对运动一致性的 top-k 邻居查找。

        输入:
            indices: [N, 4], 格式为 [batch, time, y, x]
            features: 保留旧接口参数，本函数不使用 feature 相似度。

        输出:
            idx_prev:  [N, K]
            idx_next:  [N, K]
            mask_pair: [N, K]

        计算方式:
            1. 对所有稀疏点坐标编码成 hash key。
            2. 对 key 排序。
            3. 对每个点并行枚举 previous / next 窗口内所有 offset。
            4. 使用 searchsorted 查找 previous / next 帧中对应坐标是否存在。
            5. 对所有有效 previous-next pair 按运动一致性 cost 取 top-k。

        运动一致性 cost:
            previous offset: op = p - c
            next offset:     on = n - c

            d_prev = c - p = -op
            d_next = n - c = on

            cost = ||d_next - d_prev||^2
                 = ||on + op||^2
                 = ||p + n - 2c||^2

        注意:
            idx_prev[:, q] 和 idx_next[:, q] 是第 q 个 pair。
            这是放回式 pair top-k：
                - 同一个 previous 点可以出现在多个 pair 中；
                - 同一个 next 点可以出现在多个 pair 中；
                - 不会因为某点被选中过就删除该点。
        """
        del features

        device = indices.device
        n_points = indices.shape[0]
        k = self.topk

        coords = indices.long()
        b = coords[:, 0]
        t = coords[:, 1]
        y = coords[:, 2]
        x = coords[:, 3]

        # stride 全部保留为 CUDA tensor，避免 .item() 触发 GPU/CPU 同步。
        t_max = t.max()
        y_max = y.max()
        x_max = x.max()

        t_stride = t_max + self.temporal_dilation + 3
        y_stride = y_max + self.window_radius + 3
        x_stride = x_max + self.window_radius + 3

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

        # 防止负坐标或越界坐标的 key 偶然碰撞。
        valid_coord = (
            (target_t >= 0)
            & (target_t <= t_max)
            & (target_y >= 0)
            & (target_y <= y_max)
            & (target_x >= 0)
            & (target_x <= x_max)
        )
        matched_mask = matched_mask & valid_coord.reshape(-1)

        matched_index = sorted_order[pos_safe]
        matched_index = torch.where(
            matched_mask,
            matched_index,
            torch.zeros_like(matched_index),
        )

        matched_mask = matched_mask.view(n_points, 2, n_offsets)
        matched_index = matched_index.view(n_points, 2, n_offsets)

        prev_mask_by_offset = matched_mask[:, 0]     # [N, O]
        next_mask_by_offset = matched_mask[:, 1]     # [N, O]
        prev_idx_by_offset = matched_index[:, 0]     # [N, O]
        next_idx_by_offset = matched_index[:, 1]     # [N, O]

        out_prev = torch.zeros(n_points, k, dtype=torch.long, device=device)
        out_next = torch.zeros(n_points, k, dtype=torch.long, device=device)
        mask_pair = torch.zeros(n_points, k, dtype=torch.bool, device=device)

        prev_count = prev_mask_by_offset.long().sum(dim=1)
        next_count = next_mask_by_offset.long().sum(dim=1)
        target_count = (prev_count * next_count).clamp_max(k)
        if not bool(target_count.any()):
            return out_prev, out_next, mask_pair

        fill_count = torch.zeros(n_points, dtype=torch.long, device=device)

        pair_prev_offset_id = self.pair_prev_offset_id.to(device=device)
        pair_next_offset_id = self.pair_next_offset_id.to(device=device)
        # Speed-first: scan only the lowest-cost offset pairs; topk stays 9.
        pair_block = 128
        pair_scan_count = min(pair_prev_offset_id.shape[0], 128)

        for start in range(0, pair_scan_count, pair_block):
            active_idx = (fill_count < target_count).nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                break

            end = min(start + pair_block, pair_scan_count)
            pid_prev = pair_prev_offset_id[start:end]
            pid_next = pair_next_offset_id[start:end]
            active_count = active_idx.shape[0]

            prev_mask_active = prev_mask_by_offset[active_idx]
            next_mask_active = next_mask_by_offset[active_idx]
            fill_active = fill_count[active_idx]
            target_active = target_count[active_idx]

            valid_s = prev_mask_active[:, pid_prev] & next_mask_active[:, pid_next]

            rank_s = valid_s.long().cumsum(dim=1) - 1
            dst_rank = fill_active.view(active_count, 1) + rank_s
            keep = valid_s & (dst_rank < target_active.view(active_count, 1))

            if bool(keep.any()):
                prev_s = prev_idx_by_offset[active_idx][:, pid_prev]
                next_s = next_idx_by_offset[active_idx][:, pid_next]
                active_rows = active_idx.view(active_count, 1).expand_as(keep)
                keep_rows = active_rows[keep]
                keep_rank = dst_rank[keep]

                out_prev[keep_rows, keep_rank] = prev_s[keep]
                out_next[keep_rows, keep_rank] = next_s[keep]
                mask_pair[keep_rows, keep_rank] = True

            fill_count[active_idx] = torch.minimum(
                fill_active + valid_s.long().sum(dim=1),
                target_active,
            )

        return out_prev, out_next, mask_pair

    def forward(self, x):
        input_features = x.features

        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features

        n_points, channels = features.shape
        k = self.topk

        # 1. 窗口内按 previous-next 成对运动一致性 cost 取 top-k。
        # idx_prev[:, q] 和 idx_next[:, q] 是第 q 个 pair。
        idx_prev, idx_next, mask_pair = self._find_triplet_neighbors(indices, features)

        # 2. 只对存在合法 pair 的点计算后续三元特征，减少无效 FFN 计算。
        has_pair = mask_pair.any(dim=1)

        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)

        idx_prev_v = idx_prev[valid_idx]
        idx_next_v = idx_next[valid_idx]
        mask_pair_v = mask_pair[valid_idx]

        n_valid = valid_idx.shape[0]

        # 3. 只投影参与有效更新的点及其邻居。
        prev_feat = self.prev_proj(features[idx_prev_v.reshape(-1)]).view(n_valid, k, channels)
        cur_feat = self.cur_proj(features[valid_idx]).view(n_valid, 1, channels)
        next_feat = self.next_proj(features[idx_next_v.reshape(-1)]).view(n_valid, k, channels)

        # 4. 构造三元特征。
        # z_{iq} = Wp h_prev_q + Wc h_cur + Wn h_next_q
        triplet_feat = prev_feat + cur_feat + next_feat

        # 5. 构造基于运动一致性的 6 维位置编码。
        # Δ- = current - previous
        # Δ+ = next - current
        # acc = Δ+ - Δ- = next - 2 * current + previous
        spatial = indices[:, 2:4].to(features.dtype)

        s_cur = spatial[valid_idx].view(n_valid, 1, 2)
        s_prev = spatial[idx_prev_v]
        s_next = spatial[idx_next_v]

        s_minus = s_cur - s_prev
        s_plus = s_next - s_cur
        accel = s_plus - s_minus

        motion_input = torch.cat(
            [s_minus, s_plus, accel],
            dim=-1,
        ) / max(self.pos_scale, 1e-6)

        # 6. 位置编码生成 scalar gate。
        # 初始时 pos_gate ≈ 1，训练后根据运动一致性调制三元特征。
        pos_score = self.pos_mlp(motion_input.reshape(-1, 6))
        pos_score = pos_score.view(n_valid, k, 1)

        pos_gate = 2.0 * torch.sigmoid(pos_score)
        pos_gate = pos_gate * mask_pair_v.unsqueeze(-1).to(pos_gate.dtype)

        # 7. 按你的原代码逻辑：FFN 输入不乘 pos_gate。
        # 注意：这里没有新增运动一致性权重；运动 cost 只用于 top-k 筛选。
        triplet_input = triplet_feat * pos_gate
        # triplet_input = triplet_feat
        triplet_msg = self.ffn(triplet_input.reshape(-1, channels))
        triplet_msg = triplet_msg.view(n_valid, k, channels)
        triplet_msg = triplet_msg * mask_pair_v.unsqueeze(-1).to(triplet_msg.dtype)

        # 8. 对 K 个 pair triplet message 聚合。
        msg_sum = triplet_msg.sum(dim=1)

        if self.valid_norm:
            denom = mask_pair_v.sum(dim=1)
            denom = denom.to(features.dtype).clamp_min(1.0).unsqueeze(1)
        else:
            denom = features.new_full((n_valid, 1), float(k))

        update_v = msg_sum / denom

        # 9. 使用 LayerNorm 替代 BatchNorm。
        update_v = self.act(self.norm(update_v))

        # 10. 写回完整 N 个点的 update。
        update = features.new_zeros(n_points, channels)
        update[valid_idx] = update_v

        # 11. score 表示有效 triplet 的平均位置门控强度。
        score_v = pos_gate.sum(dim=1) / denom.to(pos_gate.dtype)

        score = features.new_zeros(n_points, 1)
        score[valid_idx] = score_v

        # 12. 去掉 self.alpha 的作用，直接 residual update。
        return input_features + update, score