import time
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import math
import sys
ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# 假设 deform_conv_func.py 和 deform_conv.py 在 dcn 目录下
# 根据你的文件结构调整 import

from lib.dcn.functions.deform_conv_func import DeformConvFunction

class TSConv_CUDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t=3, kernel_size_s=3, 
                 stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, 
                 im2col_step=1, bias=True, extend_scope=3):
        """
        基于 D3Dnet CUDA 算子的时域蛇形卷积
        
        Args:
            kernel_size_t: 时域卷积核大小 (Kt)
            kernel_size_s: 空域卷积核大小 (Ks), 假设 Kh=Kw=Ks
            extend_scope: 偏移量缩放因子
        """
        super(TSConv_CUDA, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 3D 卷积核尺寸: (Kt, Ks, Ks)
        # 注意: D3Dnet 期望 kernel_size 是 tuple (Kt, Kh, Kw)
        if isinstance(kernel_size_s, int):
            self.kernel_size = (kernel_size_t, kernel_size_s, kernel_size_s)
        else:
            self.kernel_size = (kernel_size_t, kernel_size_s[0], kernel_size_s[1])
            
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        self.extend_scope = extend_scope
        
        # 1. 偏移量生成网络 (只预测 Y, X)
        # 输出通道: 2 * Kt * DeformGroups
        # 我们假设每个时间步 (Kt) 有独立的偏移，且支持 DeformGroups 分组
        self.offset_conv = nn.Conv3d(
            in_channels, 
            2 * kernel_size_t * deformable_groups, 
            kernel_size=3, 
            padding=1, 
            bias=True
        )
        self.offset_bn = nn.BatchNorm3d(2 * kernel_size_t * deformable_groups)
        
        # 2. 卷积权重
        # Shape: (Out, In/Groups, Kt, Ks, Ks)
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        # 初始化偏移生成器 (接近 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def _get_snake_offsets(self, x):
        """
        生成并处理蛇形偏移量
        Returns: 
            snake_offset: [B, 2*Kt*DG, T, H, W]
        """
        # 1. 预测
        offset = self.offset_conv(x)
        offset = self.offset_bn(offset)
        offset = torch.tanh(offset) * self.extend_scope # [-1, 1] * scope
        
        # 2. 蛇形累加约束
        # 维度: [B, 2*Kt*DG, T, H, W]
        # Reshape 方便处理: [B, DG, Kt, 2, T, H, W]
        B, _, T, H, W = offset.shape
        Kt = self.kernel_size[0]
        DG = self.deformable_groups
        
        offset = offset.view(B, DG, Kt, 2, T, H, W)
        
        # 克隆用于累加
        final_offset = offset.clone()
        center = Kt // 2
        
        # 锚定中心为 0
        final_offset[:, :, center, :, ...] = 0
        
        # 前向累加 (t+1, ...)
        for i in range(center + 1, Kt):
            final_offset[:, :, i, ...] = final_offset[:, :, i-1, ...] + offset[:, :, i, ...]
            
        # 后向累加 (t-1, ...)
        for i in range(center - 1, -1, -1):
            final_offset[:, :, i, ...] = final_offset[:, :, i+1, ...] + offset[:, :, i, ...]
            
        # 还原形状: [B, 2*Kt*DG, T, H, W]
        return final_offset.view(B, 2 * Kt * DG, T, H, W)

    def forward(self, input):
        # input: [B, C, T, H, W]
        B, C, T, H, W = input.shape
        Kt, Kh, Kw = self.kernel_size
        DG = self.deformable_groups
        
        # 1. 获取 Snake Offsets (Y, X)
        # shape: [B, 2*Kt*DG, T, H, W]
        snake_offset = self._get_snake_offsets(input) # dim=5 
        
        # 2. 构造 D3D CUDA 算子所需的 Full Offset
        # D3D 要求 Offset Shape: [B, 3 * DG * Kt * Kh * Kw, T, H, W]
        # 通道顺序: 每个采样点 (Z, Y, X) 三元组
        # 我们的需求:
        #   Z (Time): 0 (不偏移)
        #   Y (Height): 来自 snake_offset[:, 0::2]
        #   X (Width):  来自 snake_offset[:, 1::2]
        # 且对于同一个时间步 t，所有的 Spatial Kernel Points (Kh * Kw) 共享同一个 (dy, dx)
        
        # 初始化全 0 offset
        # 通道数 = 3 * DG * Kt * Kh * Kw
        full_offset = torch.zeros(B, 3 * DG * Kt * Kh * Kw, T, H, W, device=input.device, dtype=input.dtype) # dim=5 备用传入dcn
        
        # 为了高效赋值，我们 reshape
        # snake_offset: [B, DG, Kt, 2, T, H, W]
        snake_offset_reshaped = snake_offset.view(B, DG, Kt, 2, T, H, W) # 只考虑时域dim=6， 2表示xy # 里面存放了得到的偏移量
        
        # full_offset: [B, DG, Kt, Kh, Kw, 3, T, H, W]
        full_offset_view = full_offset.view(B, DG, Kt, Kh, Kw, 3, T, H, W)
        
        # 赋值逻辑：
        # dimension 0 (Time Z): 保持 0
        # dimension 1 (Height Y): 填入 snake y
        # dimension 2 (Width X): 填入 snake x
        
        # 提取 y, x
        y_off = snake_offset_reshaped[:, :, :, 0:1, ...] # [B, DG, Kt, 1, T, H, W] # 保持为dim=6
        x_off = snake_offset_reshaped[:, :, :, 1:2, ...] # [B, DG, Kt, 1, T, H, W]
        
        # 广播填充到所有 Kh, Kw
        # 我们利用 Python 的广播机制，直接赋值
        # full_offset_view[..., 1, :, :, :] 对应 Y 通道
        # full_offset_view[..., 2, :, :, :] 对应 X 通道
        
        # 注意: 这里的 broadcasting 需要 dimension matching
        # y_off 是 [B, DG, Kt, 1, T, H, W] -> 广播到 [B, DG, Kt, Kh, Kw, T, H, W]
        full_offset_view[:, :, :, :, :, 1, ...] = y_off.unsqueeze(3).expand(-1, -1, -1, Kh, Kw, -1, -1, -1)
        full_offset_view[:, :, :, :, :, 2, ...] = x_off.unsqueeze(3).expand(-1, -1, -1, Kh, Kw, -1, -1, -1)
        
        # 展平回 [B, Channels, T, H, W]
        full_offset = full_offset.view(B, -1, T, H, W)
        
        # 3. 调用 CUDA 算子
        # 确保输入连续，防止 CUDA 报错
        if not input.is_contiguous():
            input = input.contiguous()
        if not full_offset.is_contiguous():
            full_offset = full_offset.contiguous()
            
        # Ensure bias is a tensor (C++ binding does not accept None)
        bias = self.bias
        if bias is None:
            bias = self.weight.new_zeros((self.out_channels,), dtype=self.weight.dtype, device=self.weight.device)

        # Ensure integer/simple python types for scalar args
        group = int(self.groups)
        deformable_group = int(self.deformable_groups)
        im2col_step = int(self.im2col_step)

        output = DeformConvFunction.apply(
            input, 
            full_offset,
            self.weight, 
            bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            group,
            deformable_group,
            im2col_step
        )
        
        return output

# ================= 单元测试 =================
if __name__ == "__main__":
    # 模拟环境：假设已有 cuda 环境
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Testing TSConv_CUDA on GPU...")
        
        # B, C, T, H, W
        x = torch.randn(2, 32, 8, 64, 64).to(device)
        
        # 初始化
        ts_conv = TSConv_CUDA(
            in_channels=32, 
            out_channels=64, 
            kernel_size_t=5, 
            kernel_size_s=3, 
            padding=(2,1,1),
            im2col_step=8,
        ).to(device)
        
        # 前向
        time1=time.time()
        out = ts_conv(x)
        time2=time.time()
        print(f"Forward Time: {time2-time1:.4f} seconds")
        
        print(f"Input: {x.shape}")
        print(f"Output: {out.shape}")
        
        # 检查梯度 (确保 offset 参与计算)
        loss = out.sum()
        loss.backward()
        print("Backward pass successful.")
        
        # 验证 Offset 逻辑
        # 我们可以 hook 内部的 _get_snake_offsets 看看形状
        snake_offsets = ts_conv._get_snake_offsets(x)
        print(f"Snake Offset Shape: {snake_offsets.shape} (Expected: [2, 2*3*1, 8, 64, 64])")
    else:
        print("CUDA not available, skipping test.")