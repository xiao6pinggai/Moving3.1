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
from lib.models.spconv_backbone import  GroupedDilatedBlock, post_act_block

# ==========================================
# 深度优化的 Attention 模块
# ==========================================

class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=5, alpha=0.5, stride=1, 
                 indice_key="subm0", use_qkv=False, use_biqkv=False, 
                 use_maxpool=False, temporal_dilation=2, conv=None):
        super().__init__()
        # Configuration matches the request: k=5, alpha=0.5, dilation=2
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        self.temporal_dilation = temporal_dilation
        
        # Keep the specific convolution block as requested
        # self.conv = GroupedDilatedBlock(in_channels, in_channels, 3, dilations=[1, 2, 3, 4])

        self.conv = conv if conv is not None else post_act_block(16, 16, (1,3,3), norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), padding=(0,1,1), indice_key='subm1')
        # Hashing scales (using int64 to prevent overflow)
        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self.scale_b = 4096 * 4096 * 100
        
        # --- Pre-compute Spatial Offsets for Hash Keys ---
        # Instead of broadcasting N times, we calculate the integer offset for the kernel once.
        dy = torch.arange(-self.r, self.r + 1)
        dx = torch.arange(-self.r, self.r + 1)
        mesh_y, mesh_x = torch.meshgrid(dy, dx, indexing='ij')
        
        # Calculate flat offsets: offset = y * scale_y + x
        # We perform the subtraction logic (neighbor = center - offset) in forward, 
        # so here we just store the magnitude.
        # 修改时间：2026-05-24
        # 修改内容：按照 MFE 伪代码排除中心偏移 (0,0)，避免增强静止位置的背景杂波。
        # 原代码保留如下：
        # spatial_offsets = (mesh_y.reshape(-1) * self.scale_y + mesh_x.reshape(-1))
        offsets_y = mesh_y.reshape(-1)
        offsets_x = mesh_x.reshape(-1)
        valid_offsets = (offsets_y != 0) | (offsets_x != 0)
        spatial_offsets = (offsets_y[valid_offsets] * self.scale_y + offsets_x[valid_offsets])
        self.register_buffer('spatial_offsets', spatial_offsets)

    def forward(self, x):
        # 1. Feature Extraction
        features_ori = x.features
        x = self.conv(x)
        indices, features = x.indices, x.features
        N, C = features.shape
        K2 = self.spatial_offsets.numel()
        
        # 2. Compute Current Keys (Center Points)
        # Using in-place operations or simple additions is faster/lighter than creating new tensors repeatedly
        b = indices[:, 0].long()
        t = indices[:, 1].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        
        # current_keys: [N]
        current_keys = (b * self.scale_b + t * self.scale_t + y * self.scale_y + x_coord)
        
        # Sort keys to enable binary search (searchsorted)
        sorted_keys, sort_idx = torch.sort(current_keys)
        
        # 3. Prepare Normalized Features for Cosine Similarity
        # We normalize ONLY for the score calculation
        features_norm = F.normalize(features, p=2, dim=1)
        
        # We need features ordered by the sorted keys for fast lookup
        sorted_features = features_norm[sort_idx] # [N, C]

        # 4. Efficient Neighbor Search via Pre-computed Offsets
        # Time offset magnitude
        time_offset = self.temporal_dilation * self.scale_t
        
        # Generate Neighbor Keys directly using broadcasting: [N, 1] +/- [K2] -> [N, K2]
        # Prev: t - dilation, y - dy, x - dx  => Key - time_offset - spatial_offsets
        # Next: t + dilation, y + dy, x + dx  => Key + time_offset + spatial_offsets
        
        # Note: We use sorted_keys for the "base" to ensure alignment with sorted_features later?
        # No, we must search for neighbors of the *current* points.
        # But to use 'sorted_features' as the lookup table, the 'target' is the sorted array.
        # We want to find neighbors for every point in the batch.
        
        # Reshape for broadcasting
        keys_col = current_keys.unsqueeze(1) # [N, 1]
        
        keys_prev = keys_col - time_offset - self.spatial_offsets.unsqueeze(0) # [N, K2]
        keys_next = keys_col + time_offset + self.spatial_offsets.unsqueeze(0) # [N, K2]
        
        # Flatten for search
        keys_prev_flat = keys_prev.reshape(-1)
        keys_next_flat = keys_next.reshape(-1)
        
        # Search in the SORTED keys
        # ptr points to index in 'sorted_keys' (and thus 'sorted_features')
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev_flat).clamp(max=N-1)
        ptr_next = torch.searchsorted(sorted_keys, keys_next_flat).clamp(max=N-1)
        
        # Validation Mask (Exact match check)
        mask_prev = (sorted_keys[ptr_prev] == keys_prev_flat).view(N, K2)
        mask_next = (sorted_keys[ptr_next] == keys_next_flat).view(N, K2)
        
        # 5. Optimized Cosine Similarity (Using Matrix Multiplication)
        # Instead of gathering [N, K2, C], we use matmul.
        # Query: Current points [N, 1, C]
        # Key: Neighbor points gathered from sorted_features [N, K2, C]
        
        # Gather neighbors from sorted array
        feat_prev = sorted_features[ptr_prev].view(N, K2, C)
        feat_next = sorted_features[ptr_next].view(N, K2, C)
        
        # Current features (Unsorted, matching 'keys_col')
        q_curr = features_norm.view(N, 1, C) 
        
        # Cosine Similarity: (N, 1, C) @ (N, C, K2) -> (N, 1, K2)
        # This avoids creating the massive (N, K2, C) intermediate product
        score_prev = torch.matmul(q_curr, feat_prev.transpose(1, 2)).squeeze(1)
        score_next = torch.matmul(q_curr, feat_next.transpose(1, 2)).squeeze(1)
        
        # Apply ReLU and Mask
        s_prev = F.relu(score_prev) * mask_prev.float()
        s_next = F.relu(score_next) * mask_next.float()

        # 6. Symmetric Fusion
        # Determine temporal boundary conditions
        min_t, max_t = t.min(), t.max()
        is_start = (t < min_t + self.temporal_dilation).float().unsqueeze(1)
        is_end   = (t > max_t - self.temporal_dilation).float().unsqueeze(1)
        is_mid   = 1.0 - is_start - is_end
        
        # Symmetric product
        score_mid   = s_prev * s_next
        score_start = s_next * s_next
        score_end   = s_prev * s_prev
        
        total_scores = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)
        
        # Get Max Scores [N, 1]
        max_scores, _ = total_scores.max(dim=1, keepdim=True)

        # 7. Adaptive Intensity Inverse Gain
        # 修改时间：2026-05-24
        # 修改内容：按照 MFE 伪代码，将增益统计从“全体稀疏点的全局 L2 均值”
        # 修改为“同一 batch、同一帧采样集合 V_t 内的 L1 均值”，避免不同帧亮度分布互相影响。
        # 原代码保留如下：
        # intensity = torch.norm(features, p=2, dim=1, keepdim=True).detach()
        # mean_intensity = intensity.mean()
        intensity = torch.norm(features, p=1, dim=1, keepdim=True).detach()
        frame_keys = b * self.scale_t + t
        unique_frames, frame_inverse = torch.unique(frame_keys, sorted=False, return_inverse=True)
        frame_sums = torch.zeros(
            unique_frames.numel(), 1, device=features.device, dtype=features.dtype
        )
        frame_counts = torch.zeros_like(frame_sums)
        frame_sums.index_add_(0, frame_inverse, intensity)
        frame_counts.index_add_(0, frame_inverse, torch.ones_like(intensity))
        mean_intensity = frame_sums[frame_inverse] / frame_counts[frame_inverse].clamp_min(1.0)
        
        eps = 1e-5
        gamma = 0.5
        adaptive_gain = (mean_intensity / (intensity + eps)).pow(gamma)
        adaptive_gain = adaptive_gain.clamp(min=0.5, max=3.0)
        
        final_alpha = self.alpha * adaptive_gain
        
        # 8. Feature Injection
        # We apply the gain to the un-normalized features from the conv block
        # Formula: Out = Original_Input + (Conv_Output * Score * Alpha)
        max_scores = max_scores * final_alpha
        enhanced_features = features_ori + (features * max_scores)

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
            "feat": [0.1, 0.1, 0.2, 0.3,0.1, 0.1, 0.2, 0.3,0.1, 0.1, 0.2, 0.3,0.1, 0.1, 0.2, 0.3,], "name": "目标A(水平)"
        },
        {
            "start": (20, 50), "vel": (2, 0),  # 垂直向下, 速度2
            "feat": [0.8, 1.0, 0.2, 0.5,0.8, 1.0, 0.2, 0.5,0.8, 1.0, 0.2, 0.5,0.8, 1.0, 0.2, 0.5,], "name": "目标B(垂直)"
        },
        {
            "start": (10, 80), "vel": (1, -1), # 左下对角, 速度1.414
            "feat": [0.5, 0.3, 0.5, 0.9,0.5, 0.3, 0.5, 0.9,0.5, 0.3, 0.5, 0.9,0.5, 0.3, 0.5, 0.9,], "name": "目标C(对角)"
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
            noise_feat = torch.randn(16, device=device) * 0.1
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
            features_list.append(torch.randn(16, device=device))
            
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
    model = SparseSymmetricCosineAttention(in_channels=16, kernel_size=9, use_qkv=False, use_maxpool=False, alpha=1, use_biqkv=False, temporal_dilation=1).to(device)
    
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
