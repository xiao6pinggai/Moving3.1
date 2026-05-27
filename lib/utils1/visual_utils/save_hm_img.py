import torch
import numpy as np
import os


import cv2

import matplotlib
import platform
system_name = platform.system()
if system_name == 'Windows':
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 注意：需根据实际数据集配置均值和标准差
mean = np.array([0.485, 0.456, 0.406])  # 示例值，需替换为你的数据集均值
std = np.array([0.229, 0.224, 0.225])   # 示例值，需替换为你的数据集标准差

def save_hm_img(batch, output, batch_idx=0, time_idx=2):
    """
    可视化并保存大高斯热图与网络输出图，调整为2x2布局：
    - (0,0)：原图+GT大高斯热图（不变）
    - (0,1)：原图+网络输出概率图（不变）
    - (1,0)：纯网络输出热图 + 色彩刻度bar
    - (1,1)：网络输出二值掩码图（原第三幅图）

    Args:
        batch (dict): 包含 'input' (B, C, T, H, W) 和 'file_name' (列表) 的字典。
        output (list/dict): 包含网络输出的列表/字典。
            output[0]['hm_large_heatmap'] (B, 1, T, H, W): GT 大高斯热图。
            output[1] (B, C, T, H, W): 网络最终输出概率图 (假设是 Sigmoid 后的 logits，与 GT 结构类似)。
        batch_idx (int): 批次索引。
        time_idx (int): 帧索引。
    """
    # --- 1. 提取数据 ---
    input_tensor = batch['input'].cpu().numpy()
    file_names = batch['file_name']
    
    # 提取 GT 热图 (B, 1, T, H, W)
    hm_large_tensor = batch['hm_large_heatmap'].cpu().numpy()
    obj_num = batch['ind'].cpu().numpy()
        
    # **关键：提取网络最终输出图** (B, 1, T, H, W)
    net_output_tensor = output['hm_large_heatmap']
    if isinstance(net_output_tensor, torch.Tensor):
        net_output_tensor = net_output_tensor.detach().cpu().numpy()
    
    if input_tensor.shape[0] <= batch_idx or input_tensor.shape[2] <= time_idx:
        print("Error: Batch or time index out of bounds.")
        return

    # --- 2. 反标准化原始图像 ---
    # 提取原始图像 (C, H, W) -> 转换为 (H, W, C)
    img_norm = input_tensor[batch_idx, :, time_idx].transpose(1, 2, 0)
    
    # 反标准化操作： img = img * std + mean
    img_denorm = (img_norm * std) + mean
    
    # 确保图像在 [0, 1] 范围内，并转换为 8位整数 [0, 255]
    img_denorm = np.clip(img_denorm, 0, 1)
    img_vis = (img_denorm * 255).astype(np.uint8)
    
    # 提取 GT 热图帧 (H, W)
    hm_gt_frame = hm_large_tensor[batch_idx, 0, time_idx] 
    
    # 提取网络输出帧 (H, W)
    hm_net_frame = net_output_tensor[batch_idx, 0, time_idx]
    # 裁剪网络输出范围（确保可视化稳定性）
    hm_net_frame_clip = np.clip(hm_net_frame, 0, 1)
    
    # --- 3. 路径处理与格式修改 ---
    full_path = file_names[batch_idx][0]
    path_parts = full_path.split('/')
    
    # 提取文件夹名 (例如 'cityA_1159_1_2_5_5')
    dir_name = path_parts[2] if len(path_parts) > 2 else 'unknown'  # 增加边界判断
    
    # 提取文件名 (不包含扩展名)
    base_name = os.path.basename(full_path)
    file_name = os.path.splitext(base_name)[0]
    
    save_dir = os.path.join("./vis", dir_name)
    os.makedirs(save_dir, exist_ok=True)
    # 格式修改为 PNG
    obj_num = np.sum(obj_num[batch_idx][time_idx] > 0)
    save_path = os.path.join(save_dir, f"{file_name}_f{time_idx+1}_{obj_num}.png")
    
    # --- 4. 可视化处理：GT 叠加图 ---
    # A. 将 GT 热图转换为彩色 heatmap (Jet 模式)
    hm_gt_color = cv2.applyColorMap((hm_gt_frame * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_gt_color = cv2.cvtColor(hm_gt_color, cv2.COLOR_BGR2RGB) 
    
    # B. 叠加热图到原始图像上 (alpha 混合)
    alpha = 0.5 
    # 确保尺寸一致
    if img_vis.shape[:2] != hm_gt_color.shape[:2]:
        hm_gt_color = cv2.resize(hm_gt_color, (img_vis.shape[1], img_vis.shape[0]))
        
    overlay_gt = cv2.addWeighted(img_vis, 1 - alpha, hm_gt_color, alpha, 0)
    
    # --- 5. 可视化处理：网络输出叠加图 ---
    # 将网络输出热图转换为彩色 heatmap (Jet 模式)
    hm_net_color = cv2.applyColorMap((hm_net_frame_clip * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_net_color = cv2.cvtColor(hm_net_color, cv2.COLOR_BGR2RGB)
    
    # 叠加热图到原始图像上
    if img_vis.shape[:2] != hm_net_color.shape[:2]:
        hm_net_color = cv2.resize(hm_net_color, (img_vis.shape[1], img_vis.shape[0]))
        
    overlay_net = cv2.addWeighted(img_vis, 1 - alpha, hm_net_color, alpha, 0)
    
    # --- 6. 可视化处理：二值掩码图 (网络输出>0.05) ---
    # 生成二值掩码 (大于0.05为1，否则为0)
    hm_net_binary = (hm_net_frame > 0.05).astype(np.uint8)
    # 创建二值掩码的彩色可视化（白色标记掩码区域）
    hm_binary_vis = (hm_net_binary * 255).astype(np.uint8)
    overlay_binary = cv2.cvtColor(hm_binary_vis, cv2.COLOR_GRAY2RGB)  # 转为RGB适配显示
    
    # --- 7. Matplotlib 绘制：调整为2x2布局 ---
    # 调整画布尺寸适配2x2布局，预留colorbar空间
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2行2列，尺寸适配
    fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.1)  # 右侧留空间给colorbar
    
    # ===== 第一幅图 (0,0)：原图 + GT 大高斯热图（不变）=====
    axes[0,0].imshow(overlay_gt)
    axes[0,0].set_title(f"GT Large Gaussian (Frame {time_idx+1})", fontsize=10)
    axes[0,0].axis('off')

    # ===== 第二幅图 (0,1)：原图 + 网络输出概率图（不变）=====
    axes[0,1].imshow(overlay_net)
    axes[0,1].set_title(f"Network Prediction (Frame {time_idx+1})", fontsize=10)
    axes[0,1].axis('off')

    # ===== 第三幅图 (1,0)：纯网络输出热图 + 色彩bar =====
    im = axes[1,0].imshow(hm_net_frame_clip, cmap='jet', vmin=0, vmax=1)  # 纯输出热图，固定值域0-1
    axes[1,0].set_title(f"Pure Network Output (Frame {time_idx+1})", fontsize=10)
    axes[1,0].axis('off')
    # 添加色彩刻度bar（放在子图右侧）
    cbar_ax = fig.add_axes([0.47, 0.05, 0.02, 0.4])  # [left, bottom, width, height] 适配2x2布局
    fig.colorbar(im, cax=cbar_ax, label='Probability Value')

    # ===== 第四幅图 (1,1)：网络输出二值掩码图 =====
    axes[1,1].imshow(overlay_binary)
    axes[1,1].set_title(f"Network Binary Mask (>0.05) (Frame {time_idx+1})", fontsize=10)  # 修正阈值标注
    axes[1,1].axis('off')

    # 保存图片（防止标题/colorbar被裁剪）
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"Successfully saved 2x2 visualization to: {save_path}")

# --- 示例用法 (需要导入必要的函数和设置 max_objs) ---
if __name__ == '__main__':
    
    # 确保 plt.switch_backend('Agg') 在开头运行
    
    # 模拟数据
    B, C, T, H, W = 1, 3, 5, 512, 512
    
    # 模拟输入图像 (标准化后的值)
    # 确保 mock_input 在标准化后是有效的，这里直接模拟标准化后的随机值
    mock_input_norm = (torch.randn(B, C, T, H, W) * 0.08255) + 0.49965
    mock_input_norm = torch.clip(mock_input_norm, 0, 1) # 保证在 [0, 1] 范围内
    
    # 模拟大高斯热图 GT (值在 0-1 之间)
    mock_hm_large = torch.rand(B, 1, T, H, W)
    
    # 模拟网络最终输出 (值在 0-1 之间)
    mock_net_output = torch.rand(B, 1, T, H, W)
    
    mock_batch = {
        'input': mock_input_norm,
        'file_name': ["images/train/cityA_1159_1_2_5_5/img1/000001.png"]
    }
    
    # 模拟 output 列表结构：[GT_hm, net_output]
    mock_output = [{'hm_large_heatmap': mock_hm_large}, mock_net_output]
    
    print("Running visualization simulation...")
    save_hm_img(mock_batch, mock_output)
    print("Simulation finished. Check ./vis directory for output.")