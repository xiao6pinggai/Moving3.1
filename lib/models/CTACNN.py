import torch
import torch.nn.functional as F

def memory_efficient_motion_attention(frame_prev, frame_curr, frame_next, kernel_size=15):
    """
    Args:
        frame_prev, frame_curr, frame_next: (B, C, H, W)
    Returns:
        weighted_features: (B, C, H, W) - 被注意力加权后的 T 帧特征
    """
    B, C, H, W = frame_curr.shape
    radius = kernel_size // 2
    
    # 1. 预处理：归一化 (关键，否则数值范围不可控)
    # 这样点乘就变成了余弦相似度
    f_p = F.normalize(frame_prev, dim=1)
    f_c = F.normalize(frame_curr, dim=1)
    f_n = F.normalize(frame_next, dim=1)
    
    # 用于存储每个像素在所有可能速度下的最大响应值
    # 初始化为 0 或一个很小的数
    max_response_map = torch.zeros(B, 1, H, W, device=frame_curr.device)
    
    # 2. 遍历窗口内的所有可能的位移 (dy, dx)
    # 相当于遍历所有可能的“速度”
    # 循环次数 = 15*15 = 225 次，GPU上极快，显存占用极低
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # --- 核心逻辑 ---
            # 假设目标匀速运动，速度为 V=(dy, dx)
            # 那么在 T-1 帧，目标应该在 (y-dy, x-dx)
            # 那么在 T+1 帧，目标应该在 (y+dy, x+dx)
            
            # 使用 torch.roll 进行位移 (注意：边缘部分可能是乱的，但小目标检测通常忽略边缘)
            # 或者使用切片，这里为了代码简洁用 roll，工程上可用切片优化边缘
            
            # T-1 帧向右下移 (模拟去原来的位置找目标)
            feat_prev_shifted = torch.roll(f_p, shifts=(dy, dx), dims=(2, 3))
            
            # T+1 帧向左上移 (对称位置)
            # 注意这里是 (-dy, -dx)，这就是你说的“Flip中心对称”在位移上的体现
            feat_next_shifted = torch.roll(f_n, shifts=(-dy, -dx), dims=(2, 3))
            
            # 计算相似度 (B, 1, H, W)
            # 这里的 sum(dim=1) 就是 Channel 维度的点乘
            score_prev = (f_c * feat_prev_shifted).sum(dim=1, keepdim=True)
            score_next = (f_c * feat_next_shifted).sum(dim=1, keepdim=True)
            
            # 只有当两边都有高响应时，乘积才大 (ReLU滤除负相关)
            score = F.relu(score_prev) * F.relu(score_next)
            
            # 更新最大响应图
            # 意思：如果在任意一个速度方向上检测到了匀速运动，就保留这个最强的信号
            max_response_map = torch.maximum(max_response_map, score)

    # 3. 后处理：将注意力得分恢复成2D与T帧特征直接相乘
    # max_response_map 形状为 (B, 1, H, W)，作为权重 Mask
    # 这一步就是 Soft Attention 机制
    enhanced_features = frame_curr * max_response_map
    
    return enhanced_features, max_response_map

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟数据 (Batch=1, Channel=64, 512x512)
    # 这种尺寸用 Unfold 必炸，用这个方法显存几乎不涨
    x1 = torch.randn(1, 64, 512, 512).cuda()
    x2 = torch.randn(1, 64, 512, 512).cuda()
    x3 = torch.randn(1, 64, 512, 512).cuda()
    
    out_feat, attn_map = memory_efficient_motion_attention(x1, x2, x3)
    print("输出特征图尺寸:", out_feat.shape) # (1, 64, 512, 512)
    print("显存占用极低，计算完成")