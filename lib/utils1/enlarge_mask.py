import torch
import torch.nn.functional as F

def dilate_mask_fast(binary_mask):
    """
    使用 MaxPool3d 快速实现二值掩码的空间膨胀。
    输入形状: (B, 1, T, H, W)
    """
    # 1. 设置 Kernel Size
    # 想要扩大两次 (相当于 radius=2)，对应 5x5 的卷积核
    # 格式为 (D, H, W)。这里 D=1 表示不在时间维度上膨胀
    kernel_size = (1, 3, 3)
    
    # 2. 设置 Padding
    # padding = kernel_size // 2 以保持原图大小
    padding = (0, 1, 1)
    
    # 3. 执行 MaxPool
    # stride=1 保证步长为1，不降采样
    dilated_mask = F.max_pool3d(
        binary_mask, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding
    )
    
    return dilated_mask
if __name__ == '__main__':
    # --- 测试 ---
    B, T, H, W = 1, 5, 64, 64
    mask = torch.zeros(B, 1, T, H, W).cuda()
    # 模拟中心有一个点是 1
    mask[:, :, :, 30, 30] = 1.0 

    # 执行操作
    result = dilate_mask_fast(mask)

    # 验证
    print(f"原值 1.0 的数量: {mask.sum().item()}")
    print(f"膨胀后 1.0 的数量: {result.sum().item()}") # 应该变多
    # 检查中心点周围是否变成了 1 (5x5 区域)
    print(f"中心切片截取:\n{result[0, 0, 0, 28:33, 28:33]}")
    print(f"result.shape={result.shape}")