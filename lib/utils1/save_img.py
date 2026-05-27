import torch
import numpy as np
import cv2
import os

def save_net1_output(img, path, mode='255'):
    """
    将 (1, H, W) 或 (H, W) 的 0-1 范围数据保存为图片。

    Args:
        img: torch.Tensor 或 np.ndarray, 值域在 [0, 1] 之间
        path: 保存路径，例如 'results/mask.png'
        mode: 
            '255' (默认): 线性拉伸到 0-255。用于人眼观察（可视化）。
            '0-1': 仅进行取整，保留 0 和 1。用于制作数据集标签（看起来是全黑的）。
    """
    # 1. Tensor -> Numpy (自动处理 GPU/CPU)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # 2. 降维 (1, H, W) -> (H, W)
    if img.ndim == 3 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    
    # 3. 根据模式处理数值
    # 假设输入原本就是 0.0 - 1.0 的浮点数
    
    if mode == '255':
        # 可视化模式：拉伸到 0-255
        img = img * 255
        # 截断防止越界
        img = np.clip(img, 0, 255)
        
    elif mode == '0-1':
        # 标签模式：通常用于保存二值 Mask
        # 四舍五入：大于0.5变为1，小于0.5变为0
        img = np.round(img)
        # 强制截断到 0-1
        img = np.clip(img, 0, 1)
        
        # 警告：保存为 JPG 可能会因为压缩导致值不精确，建议 mode='0-1' 时保存为 png
        if path.endswith('.jpg') or path.endswith('.jpeg'):
            print(f"[Warning] Saving 0-1 mask as JPG ({path}) usually corrupts values due to compression. Use .png instead.")

    # 4. 转为整数 uint8
    img = img.astype(np.uint8)
    
    # 5. 自动创建目录
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 6. 保存
    # cv2 保存时，如果数据是 0和1，图片打开看就是全黑的，这是正常的
    cv2.imwrite(path, img)
    print(f"Saved {mode}-mode image to {path}")
if __name__ == '__mian__':
    # ================= 使用示例 =================

    # 模拟网络输出 (Sigmoid后，0~1之间)
    dummy_output = torch.rand(1, 256, 256).cuda() 

    # 场景 1：保存给人看 (黑白分明)
    save_net1_output(dummy_output, "vis_result.png", mode='255')

    # 场景 2：保存为训练用的标签 (像素值只有 0 和 1)
    save_net1_output(dummy_output, "label_mask.png", mode='0-1')