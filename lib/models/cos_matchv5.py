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
 # 改进了maxpool的策略，qkv取值取值注入到当前点的时候，使用maxpool后的索引去v中取值，而不是直接用maxpool后的分数去乘v，这样更精确一些，对cosine相似度没有影响，因为它是自增强
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, alpha=1.0, stride=1, 
                 indice_key="subm0", use_qkv=False, use_biqkv=False, use_maxpool=False):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        self.use_qkv = use_qkv
        self.use_biqkv = use_biqkv
        self.use_maxpool = use_maxpool
        
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

        # --- 通用投影层 (QKV / BiQKV) ---
        if self.use_qkv or self.use_biqkv:
            # 使用 LayerNorm 稳定特征分布，防止点积数值爆炸
            self.pre_norm = nn.LayerNorm(in_channels)
            
            # Q/K 共享投影
            self.proj_qk = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            # V 投影
            self.proj_v = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            
            # 缩放因子 (Attention is all you need paper: 1 / sqrt(d_k))
            self.scale_factor = in_channels ** -0.5

            # if self.use_biqkv:
            #     # 专门为轨迹一致性项学习一个极小的门控参数，而不是硬编码
            #     self.traj_gate = nn.Sequential(
            #         nn.Linear(1, 1),
            #         nn.Sigmoid()
            #     )

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
        # 1. 准备数据 & 哈希索引 (通用步骤)
        # -----------------------------------------------------------
        b, t, y, x_coord = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
        current_keys = self._compute_keys(b, t, y, x_coord)
        
        sorted_keys, sort_idx = torch.sort(current_keys)
        
        # 构造查询 Key
        t_col = t.unsqueeze(1)
        y_col = y.unsqueeze(1)
        x_col = x_coord.unsqueeze(1)
        b_col = b.unsqueeze(1)
        
        off_y = self.offsets_y.unsqueeze(0)
        off_x = self.offsets_x.unsqueeze(0)
        
        # 利用对称性：key_prev 对应 -Delta, key_next 对应 +Delta
        keys_prev = self._compute_keys(b_col, t_col - 1, y_col - off_y, x_col - off_x).reshape(-1)
        keys_next = self._compute_keys(b_col, t_col + 1, y_col + off_y, x_col + off_x).reshape(-1)
        
        # 批量搜索
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev).clamp(max=N-1)
        mask_prev = (sorted_keys[ptr_prev] == keys_prev)
        
        ptr_next = torch.searchsorted(sorted_keys, keys_next).clamp(max=N-1)
        mask_next = (sorted_keys[ptr_next] == keys_next)

        # -----------------------------------------------------------
        # 2. 计算相似度矩阵 (Score Map)
        # -----------------------------------------------------------
        
        if self.use_qkv or self.use_biqkv:
            # 先归一化再投影，保证 Q/K 的模长在可控范围内
            feat_norm = self.pre_norm(features)
            
            # 为了节省显存，构造一个临时的 sparse tensor 做卷积
            x_norm = x.replace_feature(feat_norm)
            qk_feat = self.proj_qk(x_norm).features
            v_feat = self.proj_v(x_norm).features
            
            sorted_qk = qk_feat[sort_idx]
            sorted_v = v_feat[sort_idx]
            
            # Gather Neighbors
            k_prev = sorted_qk[ptr_prev].view(N, K2, C)
            k_next = sorted_qk[ptr_next].view(N, K2, C)
            v_prev = sorted_v[ptr_prev].view(N, K2, C)
            v_next = sorted_v[ptr_next].view(N, K2, C)
            
            q_curr = qk_feat.view(N, 1, C)

        # === [改进版] Bi-QKV: 轨迹门控注意力 ===
        if self.use_biqkv:
            # 1. 标准 Attention 分数 (Query-Key Matching)
            # score ~ (N, K2)
            attn_prev = torch.matmul(q_curr, k_prev.transpose(1, 2)).squeeze(1) * self.scale_factor
            attn_next = torch.matmul(q_curr, k_next.transpose(1, 2)).squeeze(1) * self.scale_factor
            
            # 2. 轨迹一致性分数 (Trajectory Consistency)
            # 计算 t-1 和 t+1 对应位置特征的余弦相似度（或点积）
            # 这代表了: "如果不看当前帧，仅凭前后帧，这个位置是否连贯？"
            traj_consistency = (k_prev * k_next).sum(dim=-1) * self.scale_factor # (N, K2)
            
            # 3. 门控融合 (Gated Fusion)
            # 逻辑：最终分数 = 标准注意力 + (一致性权重 * 轨迹分)
            # 使用 tanh 限制轨迹分范围在 [-1, 1] 之间，防止过大干扰
            # 这样做的目的是：如果 Query 很弱(attn低)，但 traj 高，总分会被拉高。
            
            # 这里的 0.5 是一个先验超参，也可以设为可学习
            consistency_bias = torch.tanh(traj_consistency) 
            
            s_prev = torch.sigmoid(attn_prev + consistency_bias)
            s_next = torch.sigmoid(attn_next + consistency_bias)
            
            # 注意：这里输出的 s_prev/s_next 已经包含了轨迹先验。
            # 在后续步骤 4 中，s_prev * s_next 会进一步强化这种约束。

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
        # 3. 统一的对称处理 (保持不变)
        # -----------------------------------------------------------
        s_prev = s_prev * mask_prev.view(N, K2).float()
        s_next = s_next * mask_next.view(N, K2).float()

        s_prev_raw = s_prev
        s_next_raw = s_next
        
        idx_prev_local = None
        idx_next_local = None

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

        min_t, max_t = t.min(), t.max()
        is_start = (t_col == min_t).float()
        is_end   = (t_col == max_t).float() 
        is_mid   = 1.0 - is_start - is_end

        # 这里再次利用了对称性：(s_prev_new * s_next_new)
        # 现在的 s 已经包含了轨迹先验，所以这里是 "Enhanced Symmetry"
        score_mid   = s_prev * s_next
        score_start = s_next * s_next
        score_end   = s_prev * s_prev
        
        total_scores = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)
        max_scores, max_indices = total_scores.max(dim=1, keepdim=True)

        # -----------------------------------------------------------
        # 4. 特征增强与注入
        # -----------------------------------------------------------

        if self.use_qkv or self.use_biqkv:
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
            
            # 特征注入逻辑：
            # 我们根据 max_scores 的强度，将邻居特征加回到原始特征上
            # 这里的 alpha 控制注入强度
            injected_v = (best_v_prev * (is_mid + is_end).view(N, 1) + 
                          best_v_next * (is_mid + is_start).view(N, 1))
            
            # 残差连接：Original + Attention * Neighbor
            enhanced_features = features + (features * max_scores * self.alpha) # 输出方式
            
        else:
            # Cosine 支路 (保持不变)
            if self.use_maxpool:
                score_mid_raw   = s_prev_raw * s_next_raw
                score_start_raw = s_next_raw * s_next_raw
                score_end_raw   = s_prev_raw * s_prev_raw
                total_scores_raw = (score_mid_raw * is_mid) + (score_start_raw * is_start) + (score_end_raw * is_end)
            else:
                total_scores_raw = total_scores

            score_std = torch.std(total_scores_raw, dim=1, keepdim=True)
            max_scores = max_scores * score_std
            enhanced_features = features + (features * max_scores * self.alpha)

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
            "feat": [1.0, 0.2, 0.2], "name": "目标A(水平)"
        },
        {
            "start": (20, 50), "vel": (2, 0),  # 垂直向下, 速度2
            "feat": [0.2, 1.0, 0.2], "name": "目标B(垂直)"
        },
        {
            "start": (10, 80), "vel": (1, -1), # 左下对角, 速度1.414
            "feat": [0.2, 0.2, 1.0], "name": "目标C(对角)"
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
            noise_feat = torch.randn(3, device=device) * 0.05
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
            features_list.append(torch.randn(3, device=device))
            
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
    model = SparseSymmetricCosineAttention(in_channels=3, kernel_size=7, use_qkv=False, use_maxpool=False, alpha=1, use_biqkv=True).to(device)
    
    # 3. 运行推理
    print("-" * 60)
    print("开始推理 (Kernel Size = 7)...")
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