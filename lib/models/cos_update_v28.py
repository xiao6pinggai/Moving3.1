import torch
import torch.nn as nn

from lib.models.feature_topk_v23_loader import load_feature_topk_v23_exact

feature_topk_v23_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V28：带单侧兜底和 garbage bin 的 V27 运动对注意力。

    输入是 spconv 稀疏张量，坐标格式沿用项目里的 [batch, time, y, x]，
    特征格式为 [N, C]。模块会对每个当前帧稀疏点分别寻找前一帧和后一帧
    的 K 个特征最近邻，形成 K x K 个 prev-cur-next 候选运动三元组。

    注意力 key 使用前后两侧的 delta_xyt 和时空距离 dis 组成 8 维输入：
    前帧候选到当前点的 [dy, dx, dt, dis]，以及当前点到后帧候选的
    [dy, dx, dt, dis]。delta_xy 按空间窗口归一化，delta_t 按时间窗口归一化，
    dis 按时空窗口对角线归一化。

    如果某一侧完全没有邻居点，该侧补一个自身点参与计算，其 delta_xyt/dis
    自然为 0；不足 K 的占位点仍然作为非法点被 mask 掉。softmax 中额外加入
    一条可学习 garbage 轨迹，归一化后丢弃该 bin，因此有效轨迹注意力和可以小于 1。
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
        **kwargs,
    ):
        super().__init__()
        # kernel_size 和 alpha 是早期 MFE 接口遗留参数，v28 不参与计算；
        # 保留在函数签名中是为了兼容 build_mfe_module 的统一调用方式。
        del kernel_size, alpha
        # 这些开关属于更早版本的注意力实现，v28 固定为坐标运动注意力。
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv28 kwargs: {sorted(kwargs)}")

        # 基础超参数：C 是稀疏点特征维度，K 是每个时间方向保留的候选数。
        self.in_channels = int(in_channels)
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv28 requires tmc_topk > 0")

        # 空间搜索窗口只用于 top-k 候选选择；坐标归一化尺度使用窗口半径，
        # 避免命令行 pos_scale 改动引入额外变量。
        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        del pos_scale
        self.pos_scale = float(max(self.window_radius, 1))
        self.time_scale = float(max(abs(self.temporal_dilation), 1))
        self.spacetime_scale = float((2.0 * self.pos_scale * self.pos_scale + self.time_scale * self.time_scale) ** 0.5)
        self.chunk_size = int(chunk_size)
        self.conv = conv if conv is not None else nn.Identity()
        self.tao = 2.0  # 归一化权重的温度系数：固定在模块内部，不从命令行传入。

        # FFN 使用较小隐藏层即可，主要承担注意力更新后的通道混合。
        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))

        # 注意力前归一化让相似度尺度更稳定，FFN 前再单独做一次归一化。
        self.norm_attn = nn.LayerNorm(in_channels)
        self.norm_ffn = nn.LayerNorm(in_channels)

        # q 来自当前点；value 投影与 v23 保持一致，cur/prev/next 各自独立。
        self.q_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels, bias=False)

        # 坐标运动 key：[前侧 delta_yx/dis，后侧 delta_yx/dis] -> C 维注意力 key。
        self.traj_mlp = nn.Sequential(
            nn.Linear(8, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, in_channels),
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.scale = in_channels ** -0.5
        self.garbage_traj_key = nn.Parameter(torch.zeros(in_channels))

        # window_offsets 按离中心点的距离排序，CUDA top-k 会按这些偏移枚举窗口内候选。
        # 先访问近邻位置有利于稳定同距离候选的顺序。
        coords = torch.arange(-self.window_radius, self.window_radius + 1, dtype=torch.int32)
        oy, ox = torch.meshgrid(coords, coords, indexing="ij")
        offsets = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1).contiguous()
        offset_cost = offsets.to(torch.long).square().sum(dim=-1)
        offset_order = torch.argsort(offset_cost, stable=True).to(torch.long)
        self.register_buffer("window_offsets", offsets[offset_order].contiguous(), persistent=False)

    @staticmethod
    def _infer_dense_shape(x_conv, indices):
        """推断稀疏张量对应的密集空间尺寸 (B, T, H, W)。

        优先使用 spconv 对象携带的 batch_size 和 spatial_shape；如果当前对象
        没有这些元信息，就从稀疏坐标最大值反推尺寸。这里不做空张量兜底，
        因为 forward 已经在入口处处理了空特征。
        """
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
        """构建 dense index map，用于从 (b,t,y,x) 快速查到稀疏点行号。

        不存在稀疏点的位置填 -1；存在的位置填 features/indices 中的行号。
        CUDA top-k kernel 依赖这个表在窗口搜索时把空间坐标映射回稀疏特征。
        """
        index_map = torch.full((B, T, H, W), -1, device=indices.device, dtype=torch.int32)
        coords = indices.long()
        row_ids = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32)
        index_map[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = row_ids
        return index_map.contiguous()

    @staticmethod
    def _extract_external_index_map(x_conv):
        """复用外部已经缓存的 index_map，避免重复构建密集索引表。"""
        for name in ("index_map", "sparse_index_map", "dense_index_map"):
            value = getattr(x_conv, name, None)
            if value is not None:
                return value
        return None

    def _get_index_map(self, x_conv, indices, index_map=None):
        """取得 CUDA int32 index_map；没有外部缓存时按当前稀疏坐标新建。

        v28 明确不提供 CPU fallback，因此外部传入的 index_map 必须已经在 CUDA
        上。dtype 允许自动转为 int32，以匹配 CUDA 扩展的输入约定。
        """
        if index_map is None:
            index_map = self._extract_external_index_map(x_conv)
        if index_map is not None:
            if not index_map.is_cuda:
                raise RuntimeError("cosv28 requires CUDA index_map; no CPU fallback is available")
            if index_map.dtype != torch.int32:
                index_map = index_map.to(dtype=torch.int32)
            return index_map.contiguous()
        B, T, H, W = self._infer_dense_shape(x_conv, indices)
        return self._build_index_map(indices, B, T, H, W)

    @staticmethod
    def _check_conv_contract(input_features, x_conv, in_channels):
        """检查外部 conv 是否保持稀疏点顺序、数量和通道数不变。

        当前注意力更新按原始行号写回 update，因此 conv 只能做逐点特征变换，
        不能改变稀疏点集合或重排 indices。
        """
        if x_conv.features.shape != input_features.shape or x_conv.features.shape[1] != in_channels:
            raise RuntimeError(
                "cosv28 requires conv to preserve sparse point count, order, and channels; "
                f"got input features {tuple(input_features.shape)} and conv features {tuple(x_conv.features.shape)}"
            )

    def _adaptive_chunk_size(self, n_points, n_offsets, channels, pair_factor=1):
        """根据点数、窗口大小和通道数估计分块大小，控制显存峰值。

        chunk_size > 0 时使用用户指定值；否则用一个经验上限估计每块能处理的
        点数。pair_factor 用来粗略表达 K x K 这类成对张量的额外开销。
        """
        if self.chunk_size > 0:
            return max(1, self.chunk_size)
        denom = max(1, n_offsets * channels * pair_factor)
        return max(1, min(int(n_points), 4_000_000 // denom))

    def _load_cuda_topk(self):
        """延迟加载 v23 的 CUDA top-k 扩展。

        v28 沿用 v23 的前后帧候选搜索 kernel：输入当前点特征和窗口索引，
        输出每个点在前一帧、后一帧的 top-k 候选行号及有效 mask。
        """
        global feature_topk_v23_exact
        if feature_topk_v23_exact is None:
            feature_topk_v23_exact = load_feature_topk_v23_exact()
        return feature_topk_v23_exact

    def _select_topk_by_feature_distance(self, indices, features_norm, index_map):
        """按归一化特征距离选择前后帧候选点。

        返回 idx_prev/idx_next 和 mask_prev/mask_next，形状均为 [N, K]。
        mask 表示对应候选是否真实存在；无效位置的 idx 只作为占位，后续会被
        pair_mask 屏蔽掉，不参与注意力。
        """
        if not indices.is_cuda or not features_norm.is_cuda:
            raise RuntimeError("cosv28 Top-K requires CUDA indices and features; no Python fallback is available")
        return self._load_cuda_topk()(
            indices=indices,
            index_map=index_map,
            features=features_norm,
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=self.temporal_dilation,
            topk=self.topk,
        )

    @staticmethod
    def _inject_self_for_missing_side(idx_side, mask_side, row_ids):
        """某一侧完全无邻居时，补一个自身点；不足 K 的其它位置继续非法。"""
        missing = ~mask_side.any(dim=1)
        if not missing.any():
            return idx_side, mask_side
        idx_side = idx_side.clone()
        mask_side = mask_side.clone()
        idx_side[missing] = row_ids[missing].view(-1, 1).expand(-1, idx_side.shape[1])
        mask_side[missing] = False
        idx_side[missing, 0] = row_ids[missing]
        mask_side[missing, 0] = True
        return idx_side, mask_side

    def _masked_temperature_softmax_with_bin(self, logits, mask, q_flat):
        """一次 softmax：真实轨迹和可学习 garbage 轨迹共同竞争。

        q_flat 形状为 [Bv, C]。garbage_traj_key 是一条可学习 key，
        其 logit 同样由 q dot key 产生；softmax 后丢弃该 bin，不再重新归一化。
        """
        logits = logits.masked_fill(~mask, -1e4)
        bin_key = self.garbage_traj_key.to(dtype=q_flat.dtype, device=q_flat.device).view(1, -1)
        bin_logit = (q_flat * bin_key).sum(dim=-1, keepdim=True) * self.scale
        attn_with_bin = torch.softmax(torch.cat([logits, bin_logit], dim=-1) / self.tao, dim=-1)
        return attn_with_bin[:, :-1] * mask.to(attn_with_bin.dtype)

    def forward(self, x, index_map=None):
        """执行一次 v28 运动一致性特征增强。

        返回 output_features 和 score：output_features 与输入 features 同形状 [N, C]；
        score 是每个点最大轨迹注意力，形状 [N, 1]，主要用于沿用 MFE 接口或调试
        注意力强度。
        """
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))
        if not input_features.is_cuda:
            raise RuntimeError("cosv28 requires CUDA sparse features; no CPU fallback is available")

        # 先执行传入的局部卷积分支。v28 后续按稀疏点行号写回结果，
        # 因此这里要求 conv 不改变点数、顺序和通道数。
        x_conv = self.conv(x)
        self._check_conv_contract(input_features, x_conv, self.in_channels)
        indices = x_conv.indices
        features = x_conv.features
        if not indices.is_cuda or not features.is_cuda:
            raise RuntimeError("cosv28 requires CUDA indices and features; no Python fallback is available")

        n_points, channels = features.shape
        k = self.topk
        dense_index_map = self._get_index_map(x_conv, indices, index_map=index_map)
        # top-k 和注意力打分都基于归一化后的特征，残差则加回原始 input_features。
        features_norm = self.norm_attn(features)

        # 在当前点的时间前后各搜索一次，得到 [N, K] 的候选行号和有效标记。
        idx_prev, idx_next, mask_prev, mask_next = self._select_topk_by_feature_distance(
            indices,
            features_norm,
            dense_index_map,
        )

        # 单侧完全无邻居时补一个自身点；这样所有当前点都能进入注意力计算。
        all_rows = torch.arange(n_points, device=indices.device, dtype=idx_prev.dtype)
        idx_prev, mask_prev = self._inject_self_for_missing_side(idx_prev, mask_prev, all_rows)
        idx_next, mask_next = self._inject_self_for_missing_side(idx_next, mask_next, all_rows)
        valid_idx = all_rows.to(dtype=torch.long)
        update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)

        # 空间坐标和时间坐标一起构造 delta_xyt/dis。
        spatial = indices[:, 2:4].to(features.dtype)
        times = indices[:, 1:2].to(features.dtype)
        pair_chunk = self._adaptive_chunk_size(valid_idx.numel(), k * k, channels, pair_factor=1)
        for start in range(0, valid_idx.numel(), pair_chunk):
            end = min(start + pair_chunk, valid_idx.numel())
            rows = valid_idx[start:end]
            n_valid = rows.shape[0]

            idx_prev_v = idx_prev[rows]
            idx_next_v = idx_next[rows]
            mask_prev_v = mask_prev[rows]
            mask_next_v = mask_next[rows]
            # pair_mask: [Bv, K, K]，表示每一条 prev-cur-next 轨迹是否有效。
            pair_mask = mask_prev_v.unsqueeze(2) & mask_next_v.unsqueeze(1)

            # f_cur_res 用于残差连接；f_cur/f_prev/f_next 用于注意力投影和打分。
            f_cur_res = input_features[rows]
            f_cur = features_norm[rows]
            f_prev = features_norm[idx_prev_v.reshape(-1)].view(n_valid, k, channels)
            f_next = features_norm[idx_next_v.reshape(-1)].view(n_valid, k, channels)

            # q: [Bv, 1, 1, C]，通过广播同时匹配 K x K 条候选轨迹。
            q = self.q_proj(f_cur).view(n_valid, 1, 1, channels)
            cur_value = self.cur_proj(f_cur)
            prev_value = self.prev_proj(f_prev)
            next_value = self.next_proj(f_next)

            # 使用 y/x/t 坐标构造前后两侧的 [dy, dx, dt, dis]。
            s_cur = spatial[rows].view(n_valid, 1, 1, 2)
            s_prev = spatial[idx_prev_v].view(n_valid, k, 1, 2)
            s_next = spatial[idx_next_v].view(n_valid, 1, k, 2)
            t_cur = times[rows].view(n_valid, 1, 1, 1)
            t_prev = times[idx_prev_v].view(n_valid, k, 1, 1)
            t_next = times[idx_next_v].view(n_valid, 1, k, 1)

            dxy_prev = s_cur - s_prev
            dt_prev = t_cur - t_prev
            dis_prev = torch.sqrt(dxy_prev.square().sum(dim=-1, keepdim=True) + dt_prev.square())
            prev_input = torch.cat(
                [
                    dxy_prev / max(self.pos_scale, 1e-6),
                    dt_prev / max(self.time_scale, 1e-6),
                    dis_prev / max(self.spacetime_scale, 1e-6),
                ],
                dim=-1,
            )

            dxy_next = s_next - s_cur
            dt_next = t_next - t_cur
            dis_next = torch.sqrt(dxy_next.square().sum(dim=-1, keepdim=True) + dt_next.square())
            next_input = torch.cat(
                [
                    dxy_next / max(self.pos_scale, 1e-6),
                    dt_next / max(self.time_scale, 1e-6),
                    dis_next / max(self.spacetime_scale, 1e-6),
                ],
                dim=-1,
            )
            motion_input = torch.cat(
                [
                    prev_input.expand(-1, -1, k, -1),
                    next_input.expand(-1, k, -1, -1),
                ],
                dim=-1,
            )

            # 每条轨迹生成一个 C 维 key，再与当前点 q 做点积得到轨迹分数。
            traj_key = self.traj_mlp(motion_input.reshape(-1, 8)).view(n_valid, k, k, channels)
            logits = (q * traj_key).sum(dim=-1) * self.scale
            # 在 K x K 条真实轨迹和一条 garbage 轨迹上做一次 softmax，随后丢弃 bin。
            attn = self._masked_temperature_softmax_with_bin(
                logits.view(n_valid, k * k),
                pair_mask.view(n_valid, k * k),
                q.view(n_valid, channels),
            )
            attn = attn.view(n_valid, k, k)

            # 将 K x K 轨迹注意力边缘化为前帧权重和后帧权重，使 value 聚合保持 O(K)。
            w_prev = attn.sum(dim=2)
            w_next = attn.sum(dim=1)
            h_prev = (w_prev.unsqueeze(-1) * prev_value).sum(dim=1)
            h_next = (w_next.unsqueeze(-1) * next_value).sum(dim=1)
            # 当前点 value 加上前后帧上下文，得到本次注意力的消息。
            value = cur_value + h_prev + h_next

            # 标准残差结构：注意力消息先残差加回当前点，再经过 FFN 残差细化。
            attn_update = self.out_proj(value)
            x_attn = f_cur_res + attn_update
            out_v = x_attn + self.ffn(self.norm_ffn(x_attn))
            # update 只记录增量，循环结束后统一加回 input_features，便于未命中轨迹的点保持原样。
            # score 使用最大轨迹权重，数值越大表示该点的最可信运动三元组越集中。
            update[rows] = out_v - f_cur_res
            score[rows] = attn.amax(dim=(1, 2), keepdim=False).unsqueeze(1)

        return input_features + update, score


TripletMotionConsistencyMotionPairSparseConvV28 = TripletMotionConsistencyMotionPairSparseConv
