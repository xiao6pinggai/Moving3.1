from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

ROOT_DIR = "/root/autodl-tmp/Moving3.1"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from lib.models.spconv_utils import replace_feature, spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.spconv_utils import replace_feature, spconv
 # 改进了时域dilation，防止目标运动慢时只会取中心点
 # 新增响应强度反比增益，增强暗目标，不过度增强亮目标
class SparseSEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 定义SE模块的MLP部分，输入输出都是channel维度的向量
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        indices, features = x.indices, x.features
        N, C = features.shape
        
        # --- 1. Squeeze: Global Average Pooling over (T, H, W) ---
        # 稀疏张量的indices第一列通常是Batch Index
        # 我们需要根据Batch Index对特征进行聚合 (Scatter Mean)
        
        batch_indices = indices[:, 0].long() # [N], 确保是Long类型用于索引
        
        # 获取当前Batch的大小 (假设batch index从0开始且连续)
        batch_size = int(batch_indices.max()) + 1
        
        # 初始化聚合容器 [B, C]
        pooled_sum = torch.zeros(batch_size, C, device=features.device, dtype=features.dtype)
        
        # 将特征按Batch Index累加
        # index_add_(dim, index, source)
        pooled_sum.index_add_(0, batch_indices, features)
        
        # 计算每个Batch由多少个点组成，用于求平均
        counts = torch.zeros(batch_size, 1, device=features.device, dtype=features.dtype)
        ones = torch.ones(N, 1, device=features.device, dtype=features.dtype)
        counts.index_add_(0, batch_indices, ones)
        
        # 求平均值 [B, C]
        # clamp防止除以0 (虽然稀疏张量里通常不会有空batch进入forward，但为了鲁棒性)
        avg_pool = pooled_sum / counts.clamp(min=1.0)
        
        # --- 2. Excitation: 计算通道注意力权重 ---
        # 输入 [B, C] -> 输出 [B, C]
        channel_attention = self.fc(avg_pool)
        
        # --- 3. Scale: 将Batch级别的权重广播回点级别 ---
        # 利用 batch_indices 作为索引，将 [B, C] 的权重“查表”回 [N, C]
        # 这意味着同一个Batch内的所有点，都会获得相同的通道权重
        point_wise_attention = channel_attention[batch_indices] # [N, C]
        
        # 原地特征增强
        new_features = features * point_wise_attention
        
        # 使用replace_feature更新稀疏张量的内容
        x = x.replace_feature(new_features)
        
        return x
class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, alpha=1.0, stride=1, 
                 indice_key="subm0", use_qkv=False, use_biqkv=False, 
                 use_maxpool=False, temporal_dilation=2):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        self.use_qkv = use_qkv
        self.use_biqkv = use_biqkv
        self.use_maxpool = use_maxpool
        self.temporal_dilation = temporal_dilation
        
        # 哈希参数
        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self.scale_b = 4096 * 4096 * 100
        
        # 预先生成偏移量
        dy = torch.arange(-self.r, self.r + 1)
        dx = torch.arange(-self.r, self.r + 1)
        mesh_y, mesh_x = torch.meshgrid(dy, dx, indexing='ij')
        self.register_buffer('offsets_y', mesh_y.reshape(-1)) 
        self.register_buffer('offsets_x', mesh_x.reshape(-1))
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.bn = norm_fn(in_channels)
        self.relu = nn.ReLU()
        # --- 通用投影层 (QKV / BiQKV) ---
        if self.use_qkv or self.use_biqkv:
            # 使用 LayerNorm 稳定特征分布
            self.pre_norm = nn.LayerNorm(in_channels)
            
            # Q/K 共享投影 (基于你提供的代码逻辑，此处保持共享)
            self.proj_qk = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            # V 投影
            self.proj_v = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            
            # 缩放因子
            self.scale_factor = in_channels ** -0.5

    def _compute_keys(self, b, t, y, x):
        return (b.long() * self.scale_b + 
                t.long() * self.scale_t + 
                y.long() * self.scale_y + 
                x.long())

    def forward(self, x):
        indices, features = x.indices, x.features
        N, C = features.shape
        K2 = self.k * self.k
        
        # -----------------------------------------------------------
        # 1. 准备数据 & 哈希索引
        # -----------------------------------------------------------
        b, t, y, x_coord = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
        current_keys = self._compute_keys(b, t, y, x_coord)
        
        sorted_keys, sort_idx = torch.sort(current_keys)
        
        # 构造查询 Key (Broadcasting preparation)
        t_col = t.unsqueeze(1)
        y_col = y.unsqueeze(1)
        x_col = x_coord.unsqueeze(1)
        b_col = b.unsqueeze(1)
        
        off_y = self.offsets_y.unsqueeze(0)
        off_x = self.offsets_x.unsqueeze(0)
        
        # [修改点] 引入 temporal_dilation (t-2, t+2)
        # 空间偏移保持原样，以便在更大时间跨度下捕捉偏离中心的运动
        keys_prev = self._compute_keys(
            b_col, 
            t_col - self.temporal_dilation, 
            y_col - off_y, 
            x_col - off_x
        ).reshape(-1)

        keys_next = self._compute_keys(
            b_col, 
            t_col + self.temporal_dilation, 
            y_col + off_y, 
            x_col + off_x
        ).reshape(-1)
        
        # 批量搜索
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev).clamp(max=N-1)
        mask_prev = (sorted_keys[ptr_prev] == keys_prev)
        
        ptr_next = torch.searchsorted(sorted_keys, keys_next).clamp(max=N-1)
        mask_next = (sorted_keys[ptr_next] == keys_next)

        # -----------------------------------------------------------
        # 2. 计算相似度矩阵 (Score Map)
        # -----------------------------------------------------------
        
        # 占位符初始化
        s_prev = None
        s_next = None
        s_prev_raw = None
        s_next_raw = None
        
        # 准备投影特征
        if self.use_qkv or self.use_biqkv:
            # 先归一化
            feat_norm = self.pre_norm(features)
            
            # 为了节省显存，构造一个临时的 sparse tensor 做卷积
            x_norm = x.replace_feature(feat_norm)
            qk_feat = self.proj_qk(x_norm).features
            v_feat = self.proj_v(x_norm).features
            
            sorted_qk = qk_feat[sort_idx]
            sorted_v = v_feat[sort_idx]
            
            # Gather Neighbors (K & V)
            k_prev = sorted_qk[ptr_prev].view(N, K2, C)
            k_next = sorted_qk[ptr_next].view(N, K2, C)
            
            v_prev = sorted_v[ptr_prev].view(N, K2, C)
            v_next = sorted_v[ptr_next].view(N, K2, C)
            
            # Current Query
            q_curr = qk_feat.view(N, 1, C)

        # === [改进版] Bi-QKV: 轨迹门控注意力 ===
        if self.use_biqkv:
            # 1. 标准 Attention 分数 (Query-Key Matching)
            # 增加额外的缩放系数 0.5 以防止数值饱和
            attn_prev = torch.matmul(q_curr, k_prev.transpose(1, 2)).squeeze(1) * (self.scale_factor * 0.5)
            attn_next = torch.matmul(q_curr, k_next.transpose(1, 2)).squeeze(1) * (self.scale_factor * 0.5)
            
            # 2. 轨迹一致性分数 (Trajectory Consistency)
            # 计算 t-prev 和 t-next 的一致性
            traj_consistency = (k_prev * k_next).sum(dim=-1) * (self.scale_factor * 0.5)
            
            # 3. 门控融合
            consistency_bias = torch.tanh(traj_consistency) 
            
            s_prev = torch.sigmoid(attn_prev + consistency_bias)
            s_next = torch.sigmoid(attn_next + consistency_bias)
            
        # === [原版] QKV ===
        elif self.use_qkv:
            score_prev_raw = torch.matmul(q_curr, k_prev.transpose(1, 2)).squeeze(1)
            score_next_raw = torch.matmul(q_curr, k_next.transpose(1, 2)).squeeze(1)
            
            s_prev = torch.sigmoid(score_prev_raw * self.scale_factor)
            s_next = torch.sigmoid(score_next_raw * self.scale_factor)
            
        # === [原版] Cosine ===
        else:
            features_norm = F.normalize(features, p=2, dim=1)
            sorted_features = features_norm[sort_idx]
            
            feat_prev = sorted_features[ptr_prev].view(N, K2, C)
            feat_next = sorted_features[ptr_next].view(N, K2, C)
            
            feat_curr = features_norm.view(N, 1, C)
            score_prev = (feat_curr * feat_prev).sum(dim=2)
            score_next = (feat_curr * feat_next).sum(dim=2)
            
            s_prev = F.relu(score_prev)
            s_next = F.relu(score_next)

        # -----------------------------------------------------------
        # 3. 统一的对称处理
        # -----------------------------------------------------------
        
        # 1. 基础 Masking
        s_prev = s_prev * mask_prev.view(N, K2).float()
        s_next = s_next * mask_next.view(N, K2).float()






        # # =======================================================
        # # [新增逻辑] 对 s_prev 和 s_next 分别进行中心抑制
        # # 代码不注释表示fill0
        # # =======================================================
        # center_idx = K2 // 2
        # # 构造一个形状为 (1, K2) 的掩码，中心为 0，其余为 1
        # spatial_suppress_mask = torch.ones((1, K2), device=s_prev.device)
        # spatial_suppress_mask[:, center_idx] = 0.0
        
        # # 直接将中心点的响应降为 0
        # s_prev = s_prev * spatial_suppress_mask
        # s_next = s_next * spatial_suppress_mask
        # # =======================================================





        s_prev_raw = s_prev
        s_next_raw = s_next

        idx_prev_local = None
        idx_next_local = None

        # 2. 轨迹容差处理 (MaxPool)
        if self.use_maxpool:
            s_prev_map = s_prev.view(N, 1, self.k, self.k)
            s_next_map = s_next.view(N, 1, self.k, self.k)
            
            s_prev_dilated, idx_prev_linear = F.max_pool2d(
                s_prev_map, kernel_size=3, stride=1, padding=1, return_indices=True
            )
            s_next_dilated, idx_next_linear = F.max_pool2d(
                s_next_map, kernel_size=3, stride=1, padding=1, return_indices=True
            )
            
            s_prev = s_prev_dilated.view(N, K2)
            s_next = s_next_dilated.view(N, K2)
            
            idx_prev_local = idx_prev_linear.view(N, K2) % K2
            idx_next_local = idx_next_linear.view(N, K2) % K2

        # 3. 边界处理 (考虑 temporal_dilation)
        min_t, max_t = t.min(), t.max()
        is_start = (t_col < min_t + self.temporal_dilation).float()
        is_end   = (t_col > max_t - self.temporal_dilation).float()
        is_mid   = 1.0 - is_start - is_end

        # 4. 对称乘积融合
        score_mid   = s_prev * s_next
        score_start = s_next * s_next
        score_end   = s_prev * s_prev
        
        total_scores = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)

        # 取最大值及其索引
        max_scores, max_indices = total_scores.max(dim=1, keepdim=True)

        # -----------------------------------------------------------
        # [修改点] 中心抑制 (Center Suppression)
        # -----------------------------------------------------------
        # 如果最佳匹配点是 kernel 的中心（即物体在 t-2, t, t+2 都在原地），
        # 我们认为这是背景或静止物体，不进行增强。
        # center_idx = K2 // 2
        # is_center_mask = (max_indices == center_idx)
        # max_scores = max_scores.masked_fill(is_center_mask, 0.0)

        # -----------------------------------------------------------
        # [修改点] 自适应强度反比增益 (Adaptive Intensity Inverse Gain)
        # -----------------------------------------------------------
        # 1. 计算当前原始特征的模长 (Intensity)
        # 使用 detach 切断梯度，只作为增益系数参考
        intensity = torch.norm(features, p=2, dim=1, keepdim=True).detach() # [N, 1] L2范数
        
        # 2. 计算当前 Batch 的平均强度
        mean_intensity = intensity.mean() # 标量batch的平均强度
        
        # 3. 计算反比增益
        # 公式: Gain = (Mean / (Intensity + eps)) ^ gamma
        # 暗目标 (Intensity < Mean) -> Gain > 1 (增强)
        # 亮目标 (Intensity > Mean) -> Gain < 1 (抑制/保持)
        eps = 1e-5
        gamma = 0.5 
        adaptive_gain = (mean_intensity / (intensity + eps)).pow(gamma)
        
        # 4. 限制增益范围 [0.5, 3.0] 以防止极暗噪点爆炸或亮目标过分抑制
        adaptive_gain = adaptive_gain.clamp(min=0.5, max=3.0)
        
        # 5. 计算最终注入的 Alpha
        final_alpha = self.alpha * adaptive_gain

        # -----------------------------------------------------------
        # 4. 特征增强与注入 (Feature Injection)
        # -----------------------------------------------------------

        if self.use_qkv or self.use_biqkv:
            # Gather 最佳邻居特征 (Value)
            if self.use_maxpool:
                real_idx_prev = idx_prev_local.gather(1, max_indices)
                real_idx_next = idx_next_local.gather(1, max_indices)
                
                gather_idx_prev = real_idx_prev.unsqueeze(2).expand(-1, -1, C)
                gather_idx_next = real_idx_next.unsqueeze(2).expand(-1, -1, C)
                
                best_v_prev = v_prev.gather(1, gather_idx_prev).squeeze(1)
                best_v_next = v_next.gather(1, gather_idx_next).squeeze(1)
            else:
                gather_idx = max_indices.unsqueeze(2).expand(-1, -1, C)
                best_v_prev = v_prev.gather(1, gather_idx).squeeze(1)
                best_v_next = v_next.gather(1, gather_idx).squeeze(1)
            
            # 融合前后邻居
            injected_v = (best_v_prev * (is_mid + is_end).view(N, 1) + 
                          best_v_next * (is_mid + is_start).view(N, 1))
            
            # 使用 adaptive_gain 调节后的 final_alpha 进行注入
            enhanced_features = features + (injected_v * max_scores * final_alpha)
            
        else:
            # Cosine 支路
            if self.use_maxpool:
                score_mid_raw   = s_prev_raw * s_next_raw
                score_start_raw = s_next_raw * s_next_raw
                score_end_raw   = s_prev_raw * s_prev_raw
                total_scores_raw = (score_mid_raw * is_mid) + \
                                   (score_start_raw * is_start) + \
                                   (score_end_raw * is_end)
            else:
                total_scores_raw = total_scores

            # Cosine 支路同样应用自适应增益
            # score_std = torch.std(total_scores_raw, dim=1, keepdim=True)
            max_scores = max_scores  * final_alpha # * score_std
            enhanced_features = self.relu(self.bn(features + (features * max_scores )))

        return enhanced_features, max_scores
# ==========================================
# 模拟数据生成器
# ==========================================
def generate_simulation_data(num_frames=10, device='cuda'):
    """
    生成 10 帧数据，包含 3 个匀速运动目标和随机噪声
    """
    indices_list = []
    features_list = []
    ground_truth_labels = [] # 1=目标, 0=噪声
    descriptions = []        # 用于打印描述

    # --- 定义 3 个运动目标 ---
    # 格式: {起点(y,x), 速度(vy,vx), 基础特征, 名称}
    targets = [
        {
            "start": (10, 10), "vel": (0, 2),  # 水平向右, 速度2
            "feat": [0.1, 0.1, 0.2,0.3], "name": "目标A(水平)"
        },
        {
            "start": (20, 50), "vel": (2, 0),  # 垂直向下, 速度2
            "feat": [0.8, 1.0, 0.2,0.8], "name": "目标B(垂直)"
        },
        {
            "start": (10, 80), "vel": (1, -1), # 左下对角, 速度1.414
            "feat": [0.5, 0.3, 0.5,0.9], "name": "目标C(对角)"
        }
    ]

    print(f"生成 {num_frames} 帧数据...")

    for t in range(num_frames):
        # 1. 生成目标点
        for tgt in targets:
            # 计算当前位置 P_t = P_0 + v * t
            y = tgt["start"][0] + t * tgt["vel"][0]
            x = tgt["start"][1] + t * tgt["vel"][1]
            
            indices_list.append([0, t, y, x])
            
            # 特征添加微小扰动 (模拟传感器波动)
            base_feat = torch.tensor(tgt["feat"], device=device)
            noise_feat = torch.randn(4, device=device) * 0.1
            features_list.append(base_feat + noise_feat)
            
            ground_truth_labels.append(1) # 标记为真目标
            descriptions.append(f"帧{t} - {tgt['name']}")

        # 2. 生成随机噪声点 (每帧 2 个)
        for _ in range(2):
            # 随机位置
            ry = random.randint(0, 100)
            rx = random.randint(0, 100)
            
            # 确位置不会碰巧和目标重合 (简单略过)
            indices_list.append([0, t, ry, rx])
            
            # 随机特征
            features_list.append(torch.randn(4, device=device))
            
            ground_truth_labels.append(0) # 标记为噪声
            descriptions.append(f"帧{t} - 噪声点")

    # 转为 Tensor
    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    features = torch.stack(features_list)
    labels = torch.tensor(ground_truth_labels, device=device)
    
    return indices, features, labels, descriptions

# ==========================================
# 主函数
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # 确保使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = np.array([64, 64, 10 - 1])
    # 1. 生成数据
    # T=10, 每帧3目标+2噪声 = 5点, 总共 50 个点
    indices, features, gt_labels, descs = generate_simulation_data(num_frames=10, device=device)
    input_sp_tensor = spconv.SparseConvTensor(
            features=features,
            indices=indices.int(),
            spatial_shape=grid_size[::-1] + [1, 0, 0],
            batch_size=1
        )
    # 2. 初始化模型
    # 注意：目标最大速度是 2 (像素/帧)
    # 窗口半径 r 必须 >= 2。
    # kernel_size = 2*r + 1。如果不设为 5 或更大，就会漏检！
    # 这里设为 7 (半径3)，足以覆盖速度为 0, 1, 2, 3 的运动
    model = SparseSymmetricCosineAttention(in_channels=4, kernel_size=9, use_qkv=False, use_maxpool=False, alpha=1, use_biqkv=False, temporal_dilation=1).to(device)
    se = SparseSEModule(channel=4,reduction=2).to(device)
    # 3. 运行推理
    print("-" * 60)
    print("开始推理 (Kernel Size = 7)...")
    # input_sp_tensor = se(input_sp_tensor)
    enhanced_feat, scores = model(input_sp_tensor)
    print("推理完成。")
    print("-" * 60)

    # 4. 结果统计与验证
    scores_cpu = scores.detach().cpu().numpy().flatten()
    gt_cpu = gt_labels.cpu().numpy()
    
    print(f"{'索引':<6} | {'帧号':<4} | {'类型':<12} | {'位置(y,x)':<12} | {'得分 (Score)':<12} | {'状态'}")
    print("-" * 80)
    
    tp_count = 0 # 真阳性
    fn_count = 0 # 漏检
    fp_count = 0 # 虚警 (噪声被误报)
    
    # 阈值设定 (score > 0.5 视为检测到)
    threshold = 0.5
    
    for i in range(len(indices)):
        t = indices[i, 1].item()
        y = indices[i, 2].item()
        x = indices[i, 3].item()
        score = scores_cpu[i]
        label = gt_cpu[i]
        desc = descs[i]
        
        # 简化描述，只提取类型
        is_target = (label == 1)
        type_str = "✅ 目标" if is_target else "Testing 噪声"
        
        # 判定状态
        status = ""
        if is_target:
            if score > threshold:
                status = "DETECTED (成功)"
                tp_count += 1
            else:
                status = "MISSED (漏检)"
                fn_count += 1
        else:
            if score < threshold:
                status = "SUPPRESSED (抑制成功)"
            else:
                status = "FALSE ALARM (虚警)"
                fp_count += 1

        # 为了不刷屏，我们只打印 前2帧、中间帧、最后2帧 的部分数据
        if t <= 1 or t == 5 or t >= 8:
            print(f"{i:<6} | {t:<4} | {type_str:<12} | {f'({y},{x})':<12} | {score:.4f}       | {status}")
            
    print("-" * 80)
    print("【最终统计】")
    total_targets = gt_labels.sum().item()
    total_noise = len(gt_labels) - total_targets
    
    print(f"目标总数: {total_targets}")
    print(f"  - 成功检出: {tp_count}")
    print(f"  - 漏检:     {fn_count}")
    
    print(f"噪声总数: {total_noise}")
    print(f"  - 抑制成功: {total_noise - fp_count}")
    print(f"  - 虚警:     {fp_count}")
    
    if fn_count == 0 and fp_count == 0:
        print("\n🏆 完美通过！所有轨迹（含首尾帧）均被检测，所有噪声均被抑制。")
    else:
        print("\n⚠️ 存在部分误判，请检查 Kernel Size 是否覆盖了目标最大速度。")