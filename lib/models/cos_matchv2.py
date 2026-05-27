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
import torch
import torch.nn as nn
import torch.nn.functional as F
class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, alpha=1.0, stride=1, indice_key="subm0"):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        
        # --- 改进点 1: 使用子流形卷积生成 Q, K, V ---
        # 我们使用一个共享的卷积层一次性生成 Q, K, V，效率更高
        # in_channels -> in_channels * 3
        self.qkv_conv = spconv.SubMConv3d(
            in_channels, 
            in_channels * 3, 
            kernel_size=1, 
            stride=stride, 
            padding=1 // 2, # 保持 spatial shape 不变
            bias=False, 
            indice_key=indice_key
        )
        
        # 输出投影层 (可选，用于融合 Attention 结果)
        self.output_proj = spconv.SubMConv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=False,
            indice_key=indice_key + "_out"
        )

        # --- 哈希搜索相关参数 (逻辑不变) ---
        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self.scale_b = 4096 * 4096 * 100
        
        # 预先生成偏移量
        dy = torch.arange(-self.r, self.r + 1)
        dx = torch.arange(-self.r, self.r + 1)
        mesh_y, mesh_x = torch.meshgrid(dy, dx, indexing='ij')
        self.register_buffer('offsets_y', mesh_y.reshape(-1)) 
        self.register_buffer('offsets_x', mesh_x.reshape(-1))
        
        # 缩放因子 (Scale Dot-Product Attention)
        self.scale_factor = in_channels ** -0.5

    def _compute_keys(self, b, t, y, x):
        return (b.long() * self.scale_b + 
                t.long() * self.scale_t + 
                y.long() * self.scale_y + 
                x.long())

    def forward(self, x):
        """
        x: spconv.SparseConvTensor
        """
        indices = x.indices
        # 1. QKV 投影 (通过稀疏卷积)
        # qkv_features: (N, C*3)
        qkv_features = self.qkv_conv(x).features
        
        N, C3 = qkv_features.shape
        C = C3 // 3
        
        # 拆分 Q, K, V
        # query, key, value: (N, C)
        query, key, value = qkv_features.split(C, dim=1)

        # -----------------------------------------------------------
        # 2. 哈希索引构建 (逻辑保持不变，用于寻找 Time-Shift 邻居)
        # -----------------------------------------------------------
        b, t, y_coord, x_coord = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
        current_keys = self._compute_keys(b, t, y_coord, x_coord)
        
        # 建立 Haystack
        sorted_keys, sort_idx = torch.sort(current_keys)
        
        # 对 Key 和 Value 进行重排，方便后续 Gather
        # 注意：这里我们只重排 K 和 V，Q 不需要重排，因为 Q 是以此为中心的
        sorted_k = key[sort_idx]
        sorted_v = value[sort_idx]

        # -----------------------------------------------------------
        # 3. 构造偏移查询 (N, K*K)
        # -----------------------------------------------------------
        K2 = self.k * self.k
        
        # Expand dims for broadcasting
        t_col = t.unsqueeze(1)
        y_col = y_coord.unsqueeze(1)
        x_col = x_coord.unsqueeze(1)
        b_col = b.unsqueeze(1)
        
        off_y = self.offsets_y.unsqueeze(0)
        off_x = self.offsets_x.unsqueeze(0)

        # Generate Keys for Prev and Next frames
        keys_prev = self._compute_keys(b_col, t_col - 1, y_col - off_y, x_col - off_x).reshape(-1)
        keys_next = self._compute_keys(b_col, t_col + 1, y_col + off_y, x_col + off_x).reshape(-1)

        # -----------------------------------------------------------
        # 4. 批量查找与特征收集 (Gather)
        # -----------------------------------------------------------
        # Search
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev).clamp(max=N-1)
        mask_prev = (sorted_keys[ptr_prev] == keys_prev) # (N*K2,)
        
        ptr_next = torch.searchsorted(sorted_keys, keys_next).clamp(max=N-1)
        mask_next = (sorted_keys[ptr_next] == keys_next) # (N*K2,)

        # Gather K and V from neighbors
        # k_prev: (N*K2, C) -> view -> (N, K2, C)
        k_prev = sorted_k[ptr_prev].view(N, K2, C)
        v_prev = sorted_v[ptr_prev].view(N, K2, C)
        
        k_next = sorted_k[ptr_next].view(N, K2, C)
        v_next = sorted_v[ptr_next].view(N, K2, C)

        # -----------------------------------------------------------
        # 5. QK Attention 计算 (显存优化版)
        # -----------------------------------------------------------
        # Query: (N, C) -> Reshape to (N, 1, C)
        q_curr = query.view(N, 1, C)

        # 利用广播机制计算点乘 (N, 1, C) * (N, C, K2) -> (N, 1, K2)
        # 避免了 repeat_interleave 带来的显存浪费
        
        # Score Prev
        attn_prev = torch.matmul(q_curr, k_prev.transpose(1, 2)).squeeze(1) # (N, K2)
        attn_prev = attn_prev * self.scale_factor # Scaled Dot-Product
        
        # Score Next
        attn_next = torch.matmul(q_curr, k_next.transpose(1, 2)).squeeze(1) # (N, K2)
        attn_next = attn_next * self.scale_factor

        # Masking & Activation
        # 使用 Sigmoid 或 Softmax 归一化注意力权重
        # 这里为了保持稀疏性，我们结合 ReLU 和 Mask
        
        # Mask Shape: (N*K2) -> (N, K2)
        m_prev = mask_prev.view(N, K2).float()
        m_next = mask_next.view(N, K2).float()

        # 这里的逻辑：如果无效，分数设为极小值 (Softmax) 或 0 (ReLU)
        # 沿用你的对称逻辑，我们用 Sigmoid 使得分在 0~1 之间
        scores_prev = torch.sigmoid(attn_prev) * m_prev
        scores_next = torch.sigmoid(attn_next) * m_next

        # -----------------------------------------------------------
        # 6. 对称性约束 (Symmetric Constraint)
        # -----------------------------------------------------------
        # 处理边界时间
        min_t, max_t = t.min(), t.max()
        is_start = (t_col == min_t).float() # (N, 1)
        is_end   = (t_col == max_t).float()
        is_mid   = 1.0 - is_start - is_end

        # 对称乘积：只有当 t-1 和 t+1 对应位置的 key 都匹配较高时，才认为该轨迹有效
        score_mid = scores_prev * scores_next
        score_start = scores_next # 第一帧只看后面
        score_end = scores_prev   # 最后一帧只看前面

        # Final Attention Weights (N, K2)
        # 这是一个门控信号 (Gating)，表示我们在哪个 offset 发现了物体
        attn_weights = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)

        # -----------------------------------------------------------
        # 7. 特征注入 (Value Injection)
        # -----------------------------------------------------------
        # 我们不仅要找到分最高的，还要把对应的 Value 加进来
        # 方法 A: 只取最大值 (Hard Attention) - 类似原代码
        # 方法 B: 加权求和 (Soft Attention) - 更符合 QKV 逻辑
        
        # 这里采用加权求和，能更平滑地聚合邻域特征
        # (N, K2, 1) * (N, K2, C) -> sum dim 1 -> (N, C)
        weights_expanded = attn_weights.unsqueeze(2)
        
        # 融合 Prev 和 Next 的 Value
        # 只有中间帧才会同时融合两边，首尾帧只融合一边
        fused_value = (v_prev * weights_expanded * (is_mid.unsqueeze(2) + is_end.unsqueeze(2)) + 
                       v_next * weights_expanded * (is_mid.unsqueeze(2) + is_start.unsqueeze(2)))
        
        # 聚合所有 offsets 的结果
        context = fused_value.sum(dim=1) # (N, C)

        # -----------------------------------------------------------
        # 8. 残差连接与输出投影
        # -----------------------------------------------------------
        # 将聚合来的上下文特征再次投影 (Linear / 1x1 Conv)
        # 因为我们还在 SparseTensor 体系内，可以使用 replace_feature
        
        # 构造一个新的 SparseTensor 进行投影
        out_tensor = x.replace_feature(context)
        projected_context = self.output_proj(out_tensor).features

        # 最终残差: 原始特征 + alpha * 邻域注入特征
        final_features = x.features + self.alpha * projected_context

        return final_features, attn_weights.max(dim=1)[0].unsqueeze(1)
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