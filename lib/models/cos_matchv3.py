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

class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, alpha=1.0, stride=1, 
                 indice_key="subm0", use_qkv=False, use_maxpool=False):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        self.use_qkv = use_qkv
        self.use_maxpool = use_maxpool # [记录参数]
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

        if self.use_qkv:
            # --- QKV 投影层 ---
            # 1. Q/K 共享投影 (将特征映射到匹配空间)
            # 使用 kernel_size=1 相当于 Point-wise Linear，最省显存
            self.proj_qk = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            
            # 2. V 投影 (将邻居特征映射到注入空间)
            self.proj_v = spconv.SubMConv3d(
                in_channels, in_channels, kernel_size=1, bias=False, indice_key=indice_key
            )
            
            # 缩放因子 (防止点积爆炸)
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
        device = features.device

        # -----------------------------------------------------------
        # 1. 准备数据 & 哈希索引 (通用步骤)
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
        
        if self.use_qkv:
            # === QKV 投影逻辑 ===
            
            # 投影 Q, K, V
            # qk_feat: (N, C) -> Q 和 K 共享特征
            qk_feat = self.proj_qk(x).features
            v_feat = self.proj_v(x).features
            
            # 对 K 和 V 进行重排 (为了 Gather)
            sorted_qk = qk_feat[sort_idx]
            sorted_v = v_feat[sort_idx]
            
            # Gather Neighbors (K & V)
            # k_prev: (N, K2, C)
            k_prev = sorted_qk[ptr_prev].view(N, K2, C)
            k_next = sorted_qk[ptr_next].view(N, K2, C)
            
            # v_prev: (N, K2, C)
            v_prev = sorted_v[ptr_prev].view(N, K2, C)
            v_next = sorted_v[ptr_next].view(N, K2, C)
            
            # 计算注意力分数 (Q * K^T) - 显存优化版
            q_curr = qk_feat.view(N, 1, C) # (N, 1, C)
            
            # (N, 1, C) @ (N, C, K2) -> (N, 1, K2)
            score_prev_raw = torch.matmul(q_curr, k_prev.transpose(1, 2)).squeeze(1)
            score_next_raw = torch.matmul(q_curr, k_next.transpose(1, 2)).squeeze(1)
            
            # 缩放 + Sigmoid
            s_prev = torch.sigmoid(score_prev_raw * self.scale_factor)
            s_next = torch.sigmoid(score_next_raw * self.scale_factor)
            
        else:
            # === 原版余弦相似度逻辑 ===
            features_norm = F.normalize(features, p=2, dim=1)
            sorted_features = features_norm[sort_idx]
            
            feat_prev = sorted_features[ptr_prev].view(N, K2, C)
            feat_next = sorted_features[ptr_next].view(N, K2, C)
            
            # 广播计算点乘: (N, 1, C) * (N, K2, C) -> sum -> (N, K2)
            feat_curr = features_norm.view(N, 1, C)
            score_prev = (feat_curr * feat_prev).sum(dim=2)
            score_next = (feat_curr * feat_next).sum(dim=2)
            
            # ReLU 去负
            s_prev = F.relu(score_prev)
            s_next = F.relu(score_next)

        # -----------------------------------------------------------
        # 3. 统一的对称处理 (Symmetric Process)
        # -----------------------------------------------------------
        
        # 1. 基础 Masking (先把无效的邻居置零)
        s_prev = s_prev * mask_prev.view(N, K2).float()
        s_next = s_next * mask_next.view(N, K2).float()
        # [新增 关键点 A] 保存原始分数引用，用于后续计算 Std
        # 无论后面怎么 MaxPool，这两个变量始终保持“原始、清晰”的状态
        s_prev_raw = s_prev
        s_next_raw = s_next   

        # 2. [新增] 轨迹容差处理 (Trajectory Tolerance via MaxPool)
        if self.use_maxpool:
            # 将分数图恢复为 2D 空间结构: (N, K*K) -> (N, 1, K, K)
            s_prev_map = s_prev.view(N, 1, self.k, self.k)
            s_next_map = s_next.view(N, 1, self.k, self.k)
            
            # 执行 MaxPool (相当于形态学膨胀 Dilation)
            # kernel_size=3, padding=1 保证空间尺寸不变，只让高分向四周扩散一格
            s_prev_dilated = F.max_pool2d(s_prev_map, kernel_size=3, stride=1, padding=1)
            s_next_dilated = F.max_pool2d(s_next_map, kernel_size=3, stride=1, padding=1)
            
            # 展平回原始形状 (N, K*K)
            s_prev = s_prev_dilated.view(N, K2)
            s_next = s_next_dilated.view(N, K2)

        # 3. 边界处理 (保持不变)
        min_t, max_t = t.min(), t.max()
        is_start = (t_col == min_t).float()
        is_end   = (t_col == max_t).float()
        is_mid   = 1.0 - is_start - is_end

        # 4. 对称乘积 (N, K2)
        # 如果开启了 MaxPool，这里的 s_prev 和 s_next 已经是膨胀过的版本
        score_mid   = s_prev * s_next
        score_start = s_next * s_next
        score_end   = s_prev * s_prev
        
        total_scores = (score_mid * is_mid) + (score_start * is_start) + (score_end * is_end)

        # 取最大值及其索引
        # max_scores: (N, 1) -> 最佳匹配的权重
        # max_indices: (N, 1) -> 最佳匹配是第几个邻居
        max_scores, max_indices = total_scores.max(dim=1, keepdim=True)

        # -----------------------------------------------------------
        # 4. 特征增强与注入 (Feature Injection)
        # -----------------------------------------------------------

        if self.use_qkv:
            # === QKV 注入逻辑: 提取 Best V 并注入到 Qt ===
            
            # 构造 gather 索引: (N, 1, C)
            gather_idx = max_indices.unsqueeze(2).expand(-1, -1, C)
            
            # 提取最佳 V 特征 (Hard Attention)
            best_v_prev = v_prev.gather(1, gather_idx).squeeze(1) # (N, C)
            best_v_next = v_next.gather(1, gather_idx).squeeze(1)
            
            # 融合 V: 只有中间帧融合两边，首尾只融合一边
            injected_v = (best_v_prev * (is_mid + is_end).view(N, 1) + 
                          best_v_next * (is_mid + is_start).view(N, 1))
            
            # QKV 输出: 
            # 基础特征为 qk_feat (投影后的 Qt)，注入特征为 injected_v
            # 权重直接使用 max_scores (无需 std，因为 QK 投影模长和 Sigmoid 已包含区分度)
            enhanced_features = qk_feat + (injected_v * max_scores * self.alpha) # 是“搬运工”。它利用 max_indices 把 $t-1/t+1$ 时刻的邻居特征搬过来加给当前点。
            
        else:
            # === 原版增强逻辑 (带 Std) ===
            # [新增 关键点 B] 计算“分布置信度”分数 (必须使用原始 Raw 数据)
            # 目的：为了 Precision (精度)，确保信号是尖锐的
            if self.use_maxpool:
                # 如果开启了 MaxPool，total_scores 已经被模糊了，不能用来算 Std
                # 我们必须用 s_prev_raw 和 s_next_raw 重算一遍
                score_mid_raw   = s_prev_raw * s_next_raw
                score_start_raw = s_next_raw * s_next_raw
                score_end_raw   = s_prev_raw * s_prev_raw
                
                total_scores_raw = (score_mid_raw * is_mid) + \
                                   (score_start_raw * is_start) + \
                                   (score_end_raw * is_end)
            else:
                # 如果没开 MaxPool，原始分数就是当前分数，直接复用以节省算力
                total_scores_raw = total_scores
            # 计算标准差 (N, 1)
            # 作用: 只有当某个方向的分数显著高于其他方向(std大)时，才认为是有效轨迹
            score_std = torch.std(total_scores_raw, dim=1, keepdim=True)
            
            # 最终权重 = 最大分数 * 标准差
            max_scores = max_scores * score_std
            
            # Cosine 输出:
            # 基础特征为 features (原始输入)，增强量为 features * final_weights
            enhanced_features = features + (features * max_scores * self.alpha) #是“放大器”。它不搬运邻居特征，而是判断：“如果我发现自己处于一条直线上（置信度高），我就把我自己的特征值放大”。

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
    model = SparseSymmetricCosineAttention(in_channels=3, kernel_size=7, use_qkv=False, use_maxpool=False, alpha=1).to(device)
    
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