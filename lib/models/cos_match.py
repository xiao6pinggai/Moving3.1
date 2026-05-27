import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from lib.models.spconv_utils import replace_feature, spconv

class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, kernel_size=3, alpha=1.0):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        
        # 哈希参数
        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self.scale_b = 4096 * 4096 * 100
        
        # 预先生成所有可能的偏移量 (Offsets)
        # Shape: (K*K, 2) -> [[-r, -r], ..., [0,0], ..., [r, r]]
        dy = torch.arange(-self.r, self.r + 1)
        dx = torch.arange(-self.r, self.r + 1)
        # meshgrid 生成网格，stack 堆叠，reshape 展平
        mesh_y, mesh_x = torch.meshgrid(dy, dx, indexing='ij')
        self.register_buffer('offsets_y', mesh_y.reshape(-1)) # (K*K,)
        self.register_buffer('offsets_x', mesh_x.reshape(-1)) # (K*K,)

    def _compute_keys(self, b, t, y, x):
        return (b.long() * self.scale_b + 
                t.long() * self.scale_t + 
                y.long() * self.scale_y + 
                x.long())

    def forward(self, x):
        indices, features = x.indices, x.features
        N, C = features.shape
        K2 = self.k * self.k # Total shifts
        device = features.device

        # 1. 归一化 (N, C)
        features_norm = F.normalize(features, p=2, dim=1)

        # 2. 构建哈希表
        b, t, y, x = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
        current_keys = self._compute_keys(b, t, y, x)
        
        # 排序 Key (N,)
        sorted_keys, sort_idx = torch.sort(current_keys)
        # 重排特征 (N, C)
        sorted_features = features_norm[sort_idx]

        # 3. 向量化构造查询 Keys
        # 目标: 一次性构造所有 Prev 和 Next 的 Key
        # Input: (N,) -> Expand to (N, K*K)
        
        # (N, 1)
        t_col = t.unsqueeze(1)
        y_col = y.unsqueeze(1)
        x_col = x.unsqueeze(1)
        b_col = b.unsqueeze(1)
        
        # Offsets: (1, K*K)
        off_y = self.offsets_y.unsqueeze(0)
        off_x = self.offsets_x.unsqueeze(0)
        
        # --- 构造 Prev Keys (N, K*K) ---
        # T-1, Y-dy, X-dx
        q_t_prev = t_col - 1
        q_y_prev = y_col - off_y
        q_x_prev = x_col - off_x
        keys_prev_flat = self._compute_keys(b_col, q_t_prev, q_y_prev, q_x_prev).reshape(-1)
        
        # --- 构造 Next Keys (N, K*K) ---
        # T+1, Y+dy, X+dx
        q_t_next = t_col + 1
        q_y_next = y_col + off_y
        q_x_next = x_col + off_x
        keys_next_flat = self._compute_keys(b_col, q_t_next, q_y_next, q_x_next).reshape(-1)

        # 4. 批量查找 (Batch Search)
        # 一次性查找 N * K*K 个点
        # searchsorted: (N) haystack vs (N*K*K) needles
        
        # Prev Search
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev_flat).clamp(max=N-1)
        mask_prev = (sorted_keys[ptr_prev] == keys_prev_flat) # (N*K*K,)
        
        # Next Search
        ptr_next = torch.searchsorted(sorted_keys, keys_next_flat).clamp(max=N-1)
        mask_next = (sorted_keys[ptr_next] == keys_next_flat) # (N*K*K,)
        
        # 5. 批量计算相似度
        # 我们需要从 sorted_features 中取出特征
        # sorted_features: (N, C)
        # indices: ptr_prev (N*K*K)
        # gather 出来的特征: (N*K*K, C)
        
        feat_prev = sorted_features[ptr_prev]
        feat_next = sorted_features[ptr_next]
        
        # 当前特征需要重复 K*K 次以进行对齐: (N, C) -> (N, K*K, C) -> (N*K*K, C)
        feat_curr_expanded = features_norm.repeat_interleave(K2, dim=0)
        
        # 点乘 (N*K*K, 1)
        score_prev = (feat_curr_expanded * feat_prev).sum(dim=1, keepdim=True)
        score_next = (feat_curr_expanded * feat_next).sum(dim=1, keepdim=True)
        
        # 掩码过滤 & ReLU
        s_prev = F.relu(score_prev) * mask_prev.float().unsqueeze(1)
        s_next = F.relu(score_next) * mask_next.float().unsqueeze(1)
        
        # 6. 处理边界帧 & 对称乘积
        # 还原回 (N, K*K, 1) 以便分别处理每种 shift
        s_prev = s_prev.view(N, K2, 1)
        s_next = s_next.view(N, K2, 1)
        
        # 获取时间 mask
        min_t = t.min()
        max_t = t.max()
        is_start = (t_col == min_t).float().unsqueeze(2) # (N, 1, 1)
        is_end   = (t_col == max_t).float().unsqueeze(2)
        is_mid   = 1.0 - is_start - is_end
        
        # 计算不同情况的分数
        score_mid   = s_prev * s_next
        score_start = s_next * s_next
        score_end   = s_prev * s_prev
        
        # 组合 (N, K*K, 1)
        total_score_all_shifts = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)
        
        # 7. 取最大值 (Max Pooling over shifts)
        # dim=1 是 K*K 维度
        max_scores = total_score_all_shifts.max(dim=1)[0] # (N, 1)
        # === [新增] 标准差加权 ===
        # Std Score (N, 1)
        # 乘以 self.k 进行数量级补偿
        score_std = torch.std(total_score_all_shifts, dim=1) 
        
        # 最终分数: 既要匹配度高，又要分布尖锐(非均匀背景)
        final_scores = max_scores * score_std
        # 8. 残差增强
        enhanced_features = features + (features * final_scores * self.alpha)
        return enhanced_features, final_scores

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
    model = SparseSymmetricCosineAttention(kernel_size=7).to(device)
    
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