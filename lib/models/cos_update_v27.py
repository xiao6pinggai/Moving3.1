import torch
import torch.nn as nn

from lib.models.feature_topk_v23_loader import load_feature_topk_v23_exact

feature_topk_v23_exact = None


class TripletMotionConsistencyMotionPairSparseConv(nn.Module):
    """V27：精简版 V23 运动对注意力。

    输入是 spconv 稀疏张量，坐标格式沿用项目里的 [batch, time, y, x]，
    特征格式为 [N, C]。模块会对每个当前帧稀疏点分别寻找前一帧和后一帧
    的 K 个特征最近邻，形成 K x K 个 prev-cur-next 候选运动三元组。

    注意力 key 不再使用 v23 里可选的 feature-key 分支，而是固定由坐标运动量
    生成：当前点到前帧候选的位移、后帧候选到当前点的位移，以及二者差值
    表示的加速度。这样保留了 v23 的运动一致性建模，同时让实验变量更干净。

    本版本明确删除 v23 中已经关闭或兜底的路径：自注意力 SA、全局对齐、
    Python fallback、旧权重兼容填充、未启用的非边缘化 value 聚合。主干路径
    因此只包含 CUDA top-k、坐标运动注意力、三帧独立 value 投影、FFN 残差更新。
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
        # kernel_size 和 alpha 是早期 MFE 接口遗留参数，v27 不参与计算；
        # 保留在函数签名中是为了兼容 build_mfe_module 的统一调用方式。
        del kernel_size, alpha
        # 这些开关属于更早版本的注意力实现，v27 固定为坐标运动注意力。
        for legacy_key in ("use_qkv", "use_maxpool", "use_biqkv"):
            kwargs.pop(legacy_key, None)
        if kwargs:
            raise TypeError(f"Unexpected cosv27 kwargs: {sorted(kwargs)}")

        # 基础超参数：C 是稀疏点特征维度，K 是每个时间方向保留的候选数。
        self.in_channels = int(in_channels)
        self.temporal_dilation = int(temporal_dilation)
        self.topk = int(topk)
        if self.topk <= 0:
            raise ValueError("cosv27 requires tmc_topk > 0")

        # 空间搜索窗口只用于 top-k 候选选择；坐标归一化尺度使用窗口半径，
        # 避免命令行 pos_scale 改动引入额外变量。
        self.window_size = max(1, int(window_size))
        self.window_radius = self.window_size // 2
        del pos_scale
        self.pos_scale = float(max(self.window_radius, 1))
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

        # 坐标运动 key：[当前-前帧，后帧-当前，加速度] -> C 维注意力 key。
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

        v27 明确不提供 CPU fallback，因此外部传入的 index_map 必须已经在 CUDA
        上。dtype 允许自动转为 int32，以匹配 CUDA 扩展的输入约定。
        """
        if index_map is None:
            index_map = self._extract_external_index_map(x_conv)
        if index_map is not None:
            if not index_map.is_cuda:
                raise RuntimeError("cosv27 requires CUDA index_map; no CPU fallback is available")
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
                "cosv27 requires conv to preserve sparse point count, order, and channels; "
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

        v27 沿用 v23 的前后帧候选搜索 kernel：输入当前点特征和窗口索引，
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
            raise RuntimeError("cosv27 Top-K requires CUDA indices and features; no Python fallback is available")
        return self._load_cuda_topk()(
            indices=indices,
            index_map=index_map,
            features=features_norm,
            window_offsets=self.window_offsets.to(device=indices.device),
            temporal_dilation=self.temporal_dilation,
            topk=self.topk,
        )

    def _masked_temperature_softmax(self, logits, mask):
        """带 mask 和温度系数的归一化。

        logits 形状通常是 [有效点数, K*K]。先把无效轨迹填为极小值，再除以
        self.tao 做温度缩放；最后二次乘 mask 并重新归一化，避免所有无效位置
        的数值残留影响边缘化权重。
        """
        logits = logits.masked_fill(~mask, -1e4)
        attn = torch.softmax(logits / self.tao, dim=-1)
        attn = attn * mask.to(attn.dtype)
        return attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x, index_map=None):
        """执行一次 v27 运动一致性特征增强。

        返回 output_features 和 score：output_features 与输入 features 同形状 [N, C]；
        score 是每个点最大轨迹注意力，形状 [N, 1]，主要用于沿用 MFE 接口或调试
        注意力强度。
        """
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))
        if not input_features.is_cuda:
            raise RuntimeError("cosv27 requires CUDA sparse features; no CPU fallback is available")

        # 先执行传入的局部卷积分支。v27 后续按稀疏点行号写回结果，
        # 因此这里要求 conv 不改变点数、顺序和通道数。
        x_conv = self.conv(x)
        self._check_conv_contract(input_features, x_conv, self.in_channels)
        indices = x_conv.indices
        features = x_conv.features
        if not indices.is_cuda or not features.is_cuda:
            raise RuntimeError("cosv27 requires CUDA indices and features; no Python fallback is available")

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

        # 只有同时存在前帧和后帧候选点的当前点，才能构成完整的 prev-cur-next 轨迹。
        has_pair = mask_prev.any(dim=1) & mask_next.any(dim=1)
        valid_idx = has_pair.nonzero(as_tuple=False).squeeze(1)
        update = input_features.new_zeros(n_points, channels)
        score = features.new_zeros(n_points, 1)
        if valid_idx.numel() == 0:
            return input_features, score

        # 只取空间坐标 y/x 参与运动建模，时间方向已经由 prev/next 搜索确定。
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

            # 使用 y/x 坐标构造轨迹几何量。
            s_cur = spatial[rows].view(n_valid, 1, 1, 2)
            s_prev = spatial[idx_prev_v].view(n_valid, k, 1, 2)
            s_next = spatial[idx_next_v].view(n_valid, 1, k, 2)
            # d_minus 表示从前帧候选到当前点的位移，d_plus 表示从当前点到后帧候选的位移。
            # accel 越小，说明前后两段运动越接近匀速直线运动。
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
            # 归一化坐标量，避免窗口尺寸变化直接放大 MLP 输入尺度。
            motion_input = motion_input / max(self.pos_scale, 1e-6)

            # 每条轨迹生成一个 C 维 key，再与当前点 q 做点积得到轨迹分数。
            traj_key = self.traj_mlp(motion_input.reshape(-1, 6)).view(n_valid, k, k, channels)
            logits = (q * traj_key).sum(dim=-1) * self.scale
            # 在 K x K 条轨迹上归一化，self.tao 控制分布尖锐程度。
            attn = self._masked_temperature_softmax(logits.view(n_valid, k * k), pair_mask.view(n_valid, k * k))
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


TripletMotionConsistencyMotionPairSparseConvV27 = TripletMotionConsistencyMotionPairSparseConv
