import torch

# def bboxes_to_binary_mask(bboxes, H, W):
#     """
#     将 bounding boxes 转换为二值掩码 tensor。
#     修正后输出形状严格为 (B, 1, T, H, W)
#     """
#     # 1. 获取基础信息和设备
#     B, T, N, _ = bboxes.shape
#     device = bboxes.device
    
#     # 2. 生成网格坐标
#     y_range = torch.arange(H, device=device, dtype=torch.float32)
#     x_range = torch.arange(W, device=device, dtype=torch.float32)
#     y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
#     # 3. 维度扩展以支持广播
#     # Grid Shape: (1, 1, 1, H, W)
#     y_grid = y_grid[None, None, None, :, :] 
#     x_grid = x_grid[None, None, None, :, :]
    
#     # 4. 【关键修改】提取坐标并调整维度
#     # 使用整数索引 '0' 而不是切片 '0:1'，以去掉最后一维
#     # bboxes[..., 0] shape: (B, T, N)
#     # 增加 None 后 shape: (B, T, N, 1, 1)
#     x1 = bboxes[..., 0, None, None]
#     y1 = bboxes[..., 1, None, None]
#     x2 = bboxes[..., 2, None, None]
#     y2 = bboxes[..., 3, None, None]
    
#     # 5. 生成掩码
#     # (1, 1, 1, H, W) vs (B, T, N, 1, 1) -> 广播结果: (B, T, N, H, W)
#     mask_per_box = (x_grid >= x1) & (x_grid <= x2) & \
#                    (y_grid >= y1) & (y_grid <= y2)
    
#     # 6. 过滤无效框
#     # valid_box shape: (B, T, N, 1, 1)
#     valid_box = (x2 > x1) & (y2 > y1) 
#     mask_per_box = mask_per_box & valid_box

#     # 7. 聚合 N 维度 (Union)
#     # (B, T, N, H, W) -> (B, T, H, W)
#     final_mask = mask_per_box.any(dim=2)
    
#     # 8. 调整最终形状
#     # (B, T, H, W) -> (B, 1, T, H, W)
#     return final_mask.unsqueeze(1).float()
def bboxes_to_binary_mask(bboxes, H, W):
    """
    针对稀疏框优化的掩码生成函数。
    当检测到 bbox 为全 0 填充时，跳过计算。
    """
    B, T, N, _ = bboxes.shape
    device = bboxes.device
    
    # 1. 初始化最终掩码 (B, T, H, W), 使用 bool 节省显存
    final_mask = torch.zeros((B, T, H, W), device=device, dtype=torch.bool)
    
    # 2. 生成网格 (预先生成，避免循环内重复开销)
    y_range = torch.arange(H, device=device, dtype=torch.float32)
    x_range = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # (H, W) -> (1, 1, H, W) 用于广播
    grid_y = grid_y[None, None, :, :]
    grid_x = grid_x[None, None, :, :]
    
    # 3. 循环 N，逐个处理，但跳过空框
    for i in range(N):
        # 取出当前第 i 个框的数据: shape (B, T, 4)
        bbox_i = bboxes[:, :, i, :] 

        # =======================================================
        # 【优化核心】：如果该索引下，所有 Batch 和 Time 的框都是 0，直接跳过
        # 判断依据：x2 (index 2) 是否全为 0 (通常 x2=0 意味着无效框)
        # 或者更严格： sum() == 0
        # =======================================================
        if (bbox_i == 0).all(): 
            continue
            
        # 如果你想更严谨一点（防止有框是 [0,0,0,0] 但实际上是噪点），
        # 可以用这个判断：所有框的宽度都 <= 0
        # valid_check = (bbox_i[..., 2] > bbox_i[..., 0]) & (bbox_i[..., 3] > bbox_i[..., 1])
        # if not valid_check.any():
        #     continue

        # 4. 只有当存在有效框时，才执行下面的矩阵运算
        # 扩展维度: (B, T) -> (B, T, 1, 1)
        x1 = bbox_i[..., 0, None, None]
        y1 = bbox_i[..., 1, None, None]
        x2 = bbox_i[..., 2, None, None]
        y2 = bbox_i[..., 3, None, None]
        
        # 再次生成有效性掩码 (B, T, 1, 1)
        # 这一步是为了处理：Batch中部分样本有框，部分样本是0填充的情况
        valid_box = (x2 > x1) & (y2 > y1) 
        
        # 只有在有效时才进行 Grid 比较
        # (B, T, H, W)
        mask_i = (grid_x >= x1) & (grid_x <= x2) & \
                 (grid_y >= y1) & (grid_y <= y2)
                 
        # 过滤掉 0 填充产生的错误 mask (0>=0 是 True, 会导致全图变白，必须用 valid_box 过滤)
        mask_i = mask_i & valid_box
        
        # 累积结果
        final_mask.logical_or_(mask_i)

    # 5. 转换回 float 并增加通道维度
    return final_mask.unsqueeze(1).float()
# --- 使用示例 ---
# 假设 bboxes 是之前生成的 (Batch=2, Time=5, N=512, 6)
# bboxes = torch.randn(2, 5, 512, 6).to('cuda') # 模拟数据
# H, W = 256, 256

# # 调用
# binary_mask = bboxes_to_binary_mask(bboxes, H, W)

# print(binary_mask.shape)  # torch.Size([2, 1, 5, 256, 256])
# print(binary_mask.device) # 与输入 bboxes 相同