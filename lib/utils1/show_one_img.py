import matplotlib
import platform
system_name = platform.system()
if system_name == 'Windows':
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def show_one_img(img, save_dir='debug_imgs', img_name='debug.png'):
    system_name = platform.system()
    if system_name == 'Windows':
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    # 1. 数据处理：确保从 GPU 拿下来并转为 numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # 显示ndarray图像
    h, w = img.shape[:2]
    plt.figure(figsize=(h / 100, w / 100))  # 设置显示窗口大小
    if len(img.shape) == 3:
        plt.imshow(img)  # 灰度图需指定cmap='gray'
        print("rgb")
    else:
        plt.imshow(img, cmap='gray')
        print('gray')
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像
# if __name__ == '__main__':
#     img = np.random.random([1024,1024])
#     show_one_img(img)
# # 使用示例
# # show_one_img(binary_mask[0][0][0])
if __name__ == '__main__':
    print("=== 开始测试 show_one_img 函数 ===")

    # 1. 基础测试：普通 Numpy 数组 (灰度)
    # 修正：np.random 模块下没有直接调用的 random 函数接受 list 作为参数生成数组，
    # 通常用 np.random.rand(H, W) 或 np.random.random((H, W))
    print(">> Case 1: Testing Numpy Array (Gray)")
    img_np = np.random.rand(512, 512) 
    show_one_img(img_np, img_name='test_numpy_gray.png')

    # 2. 进阶测试：PyTorch Tensor (模拟真实训练中的最复杂情况)
    # 测试点：
    #   a. 位于 GPU 上 (cuda) -> 测试 .cpu()
    #   b. 带有梯度 (requires_grad) -> 测试 .detach()
    #   c. 带有 Batch 维度 (1, 1, H, W) -> 测试 .squeeze()
    print("\n>> Case 2: Testing PyTorch Tensor (GPU + Grad + Batch)")
    if torch.cuda.is_available():
        # 创建一个模拟的 GPU Tensor
        img_tensor = torch.randn(1, 1, 256, 256).cuda().requires_grad_(True)
        # 如果函数足够健壮，这里不会报错
        show_one_img(img_tensor, img_name='test_tensor_gpu.png')
    else:
        # CPU 环境测试
        img_tensor = torch.randn(1, 1, 256, 256).requires_grad_(True)
        show_one_img(img_tensor, img_name='test_tensor_cpu.png')

    # 3. 形状测试：彩色图像
    print("\n>> Case 3: Testing Color Image")
    img_color = np.random.rand(512, 512, 3)
    show_one_img(img_color, img_name='test_numpy_color.png')

    print("\n=== 测试结束，如果没有报错且看到了图片（或生成了文件），说明函数正常 ===")