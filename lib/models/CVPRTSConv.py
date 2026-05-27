import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSnakeConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size_t=3, kernel_size_s=3, padding=1, bias=False, 
                 extend_scope=10, if_offset=True, device='cuda'):
        """
        针对运动小目标的时域蛇形卷积
        输入: [B, C, T, H, W]
        输出: [B, Out_C, T, H, W]
        """
        super(TemporalSnakeConv3d, self).__init__()
        self.device = device
        self.kernel_size_t = kernel_size_t
        self.extend_scope = extend_scope
        self.if_offset = if_offset

        # 1. 偏移量生成 (Offset Generation)
        # 输入: [B, C, T, H, W]
        # 输出: [B, 2*Kt, T, H, W] -> 只预测 Y(Height) 和 X(Width) 的偏移
        # 我们不预测时间轴偏移，时间轴按规则滑动
        self.offset_conv = nn.Conv3d(in_ch, 2 * kernel_size_t, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(2 * kernel_size_t)
        
        # 2. 聚合卷积 (Aggregated Convolution)
        # 作用：将变形后的特征图重新聚合，同时提取空域特征
        # 输入 Depth: T * Kt
        # Kernel: (Kt, 3, 3) -> 时间上一次卷 Kt 个点(即一个蛇形轨迹)，空间上 3x3
        # Stride: (Kt, 1, 1) -> 时间上步长为 Kt，保证输出 T 维度恢复为原大小
        self.aggregated_conv = nn.Conv3d(
            in_ch, 
            out_ch, 
            kernel_size=(kernel_size_t, kernel_size_s, kernel_size_s), 
            stride=(kernel_size_t, 1, 1), 
            padding=(0, padding, padding),
            bias=bias
        )
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f):
        # f: [B, C, T, H, W]
        
        # 1. 生成偏移量
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset) # [-1, 1]
        # store learned offsets (scaled by extend_scope) for visualization/debugging
        try:
            self.last_snake_offset = (offset * self.extend_scope).detach().cpu()
        except Exception:
            self.last_snake_offset = None

        # 2. 核心蛇形采样
        input_shape = f.shape
        dsc_core = TemporalSnakeCore(input_shape, self.kernel_size_t, self.extend_scope, self.device)
        
        # 输出: [B, C, T*Kt, H, W]
        # 这里的时间维度被放大了 Kt 倍，包含了所有采样点
        deformed_feature = dsc_core.deform_conv(f, offset, self.if_offset)
        
        # 3. 聚合与标准空域卷积
        # [B, C, T*Kt, H, W] -> [B, Out_C, T, H, W]
        x = self.aggregated_conv(deformed_feature)
        x = self.gn(x)
        x = self.relu(x)
        
        return x


class TemporalSnakeCore(object):
    def __init__(self, input_shape, kernel_size, extend_scope, device):
        # input_shape: [B, C, T, H, W]
        self.num_points = kernel_size # Kt
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]
        self.time = input_shape[2]    # T
        self.height = input_shape[3]  # H (Y轴)
        self.width = input_shape[4]   # W (X轴)
        self.device = device
        self.extend_scope = extend_scope

    def _coordinate_map_temporal(self, offset, if_offset):
        """
        构建时域蛇形网格
        Args:
            offset: [B, 2*Kt, T, H, W]
        Returns:
            t_new, y_new, x_new: [B, T*Kt, H, W] (Flattened for aggregation)
        """
        # offset split: 前 Kt 个是 Y (H) 偏移, 后 Kt 个是 X (W) 偏移
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        # === 1. 构建基础坐标网格 (MeshGrid) ===
        # T (Time): [0, 1, ..., T-1]
        t_center = torch.arange(0, self.time).reshape(1, self.time, 1, 1)
        t_center = t_center.repeat([self.num_points, 1, self.height, self.width]).float()
        t_center = t_center.unsqueeze(0) # [1, Kt, T, H, W]

        # H (Y-axis): [0, 1, ..., H-1]
        y_center = torch.arange(0, self.height).reshape(1, 1, self.height, 1)
        y_center = y_center.repeat([self.num_points, self.time, 1, self.width]).float()
        y_center = y_center.unsqueeze(0)

        # W (X-axis): [0, 1, ..., W-1]
        x_center = torch.arange(0, self.width).reshape(1, 1, 1, self.width)
        x_center = x_center.repeat([self.num_points, self.time, self.height, 1]).float()
        x_center = x_center.unsqueeze(0)

        # === 2. 构建卷积核偏移 (Kernel Spread) ===
        # 时间轴: [-k//2, ..., 0, ..., k//2]
        t_kernel = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
        # 空间轴: 0 (空间本身不扩展，全靠 offset 移动)
        
        # t_grid: [1, Kt, T, H, W]
        t_grid = t_kernel.reshape(1, self.num_points, 1, 1, 1)
        t_grid = t_grid.repeat([1, 1, self.time, self.height, self.width])

        # === 3. 生成绝对坐标 ===
        t_new = t_center + t_grid # Time: t-1, t, t+1
        y_new = y_center          # Height: y
        x_new = x_center          # Width: x

        # 扩展 Batch
        t_new = t_new.repeat(self.num_batch, 1, 1, 1, 1).to(self.device)
        y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1).to(self.device)
        x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1).to(self.device)

        # === 4. 应用 Snake Offsets ===
        if if_offset:
            y_offset_new = y_offset.detach().clone()
            x_offset_new = x_offset.detach().clone()
            
            # Snake 迭代累加约束
            center = int(self.num_points // 2)
            
            # 锚定中心：中心帧无偏移
            y_offset_new[:, center, ...] = 0
            x_offset_new[:, center, ...] = 0
            
            # 向未来累加 (Forward)
            for index in range(1, center + 1):
                y_offset_new[:, center + index] = y_offset_new[:, center + index - 1] + y_offset[:, center + index]
                x_offset_new[:, center + index] = x_offset_new[:, center + index - 1] + x_offset[:, center + index]
                
                # 向过去累加 (Backward)
                y_offset_new[:, center - index] = y_offset_new[:, center - index + 1] + y_offset[:, center - index]
                x_offset_new[:, center - index] = x_offset_new[:, center - index + 1] + x_offset[:, center - index]

            # 加上偏移量
            y_new = y_new + y_offset_new * self.extend_scope
            x_new = x_new + x_offset_new * self.extend_scope

        # === 5. Reshape 适配聚合卷积 ===
        # 原始: [B, Kt, T, H, W]
        # 目标: [B, T*Kt, H, W] (为了 Conv3d 的 stride=Kt 能正确分组)
        # 我们希望的顺序是: Frame0_K0, Frame0_K1... Frame0_Kn, Frame1_K0...
        # 所以先要把 Kt 放到 T 后面，然后合并
        
        # Permute: [B, T, Kt, H, W]
        t_new = t_new.permute(0, 2, 1, 3, 4).reshape(self.num_batch, self.time * self.num_points, self.height, self.width)
        y_new = y_new.permute(0, 2, 1, 3, 4).reshape(self.num_batch, self.time * self.num_points, self.height, self.width)
        x_new = x_new.permute(0, 2, 1, 3, 4).reshape(self.num_batch, self.time * self.num_points, self.height, self.width)

        return t_new, y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, t, y, x):
        """
        在 [B, C, T, H, W] 特征图上进行插值
        Coords Shape: [B, T*Kt, H, W]
        Output Shape: [B, C, T*Kt, H, W]
        """
        # Flatten Coords to [-1]
        t = t.reshape([-1]).float()
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_t = self.time - 1
        max_y = self.height - 1
        max_x = self.width - 1

        # 寻找 8 邻域 (Trilinear)
        t0 = torch.floor(t).int()
        t1 = t0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # Clip (Border Mode: Clamp)
        t0 = torch.clamp(t0, zero, max_t)
        t1 = torch.clamp(t1, zero, max_t)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        # 展平 Input Feature
        # [B, C, T, H, W] -> [B, T, H, W, C] -> [-1, C]
        input_flat = input_feature.permute(0, 2, 3, 4, 1).reshape(-1, self.num_channels)

        # 计算 Base Index (Batch Offset)
        # 每个 Batch 的大小
        dim_batch = self.time * self.height * self.width
        
        # 生成 Batch 索引: [0, dim, 2*dim ...]
        base = torch.arange(self.num_batch) * dim_batch
        base = base.reshape([-1, 1]).float().to(self.device)
        
        # 这一步是关键：coords 是 [B, T*Kt, H, W]
        # 每个 Batch 需要采样的点数是 T*Kt * H * W
        num_samples_per_batch = self.num_points * self.time * self.height * self.width
        repeat = torch.ones([num_samples_per_batch]).unsqueeze(0).float().to(self.device)
        
        # [B, Num_Samples] -> Flatten -> [-1]
        base = torch.matmul(base, repeat).reshape([-1])

        # 计算 8 个角点的 1D 索引
        # Index = Base + t*(H*W) + y*W + x
        stride_t = self.height * self.width
        stride_y = self.width
        
        base_t0 = base + t0 * stride_t
        base_t1 = base + t1 * stride_t
        
        # 8 corners indices
        idx_a0 = base_t0 + y0 * stride_y + x0
        idx_b0 = base_t1 + y0 * stride_y + x0
        idx_c0 = base_t0 + y1 * stride_y + x0
        idx_d0 = base_t1 + y1 * stride_y + x0
        idx_a1 = base_t0 + y0 * stride_y + x1
        idx_b1 = base_t1 + y0 * stride_y + x1
        idx_c1 = base_t0 + y1 * stride_y + x1
        idx_d1 = base_t1 + y1 * stride_y + x1

        # Gather Features
        v_a0 = input_flat[idx_a0.long()]
        v_b0 = input_flat[idx_b0.long()]
        v_c0 = input_flat[idx_c0.long()]
        v_d0 = input_flat[idx_d0.long()]
        v_a1 = input_flat[idx_a1.long()]
        v_b1 = input_flat[idx_b1.long()]
        v_c1 = input_flat[idx_c1.long()]
        v_d1 = input_flat[idx_d1.long()]

        # Trilinear Interpolation Weights
        t0_f, t1_f = t0.float(), t1.float()
        y0_f, y1_f = y0.float(), y1.float()
        x0_f, x1_f = x0.float(), x1.float()

        # Weights
        w_a0 = ((t1_f - t) * (y1_f - y) * (x1_f - x)).unsqueeze(-1)
        w_b0 = ((t - t0_f) * (y1_f - y) * (x1_f - x)).unsqueeze(-1)
        w_c0 = ((t1_f - t) * (y - y0_f) * (x1_f - x)).unsqueeze(-1)
        w_d0 = ((t - t0_f) * (y - y0_f) * (x1_f - x)).unsqueeze(-1)
        w_a1 = ((t1_f - t) * (y1_f - y) * (x - x0_f)).unsqueeze(-1)
        w_b1 = ((t - t0_f) * (y1_f - y) * (x - x0_f)).unsqueeze(-1)
        w_c1 = ((t1_f - t) * (y - y0_f) * (x - x0_f)).unsqueeze(-1)
        w_d1 = ((t - t0_f) * (y - y0_f) * (x - x0_f)).unsqueeze(-1)

        output = v_a0 * w_a0 + v_b0 * w_b0 + v_c0 * w_c0 + v_d0 * w_d0 + \
                 v_a1 * w_a1 + v_b1 * w_b1 + v_c1 * w_c1 + v_d1 * w_d1

        # Restore Shape
        # Flatten -> [B, T*Kt, H, W, C] -> [B, C, T*Kt, H, W]
        output = output.reshape(self.num_batch, self.time * self.num_points, self.height, self.width, self.num_channels)
        output = output.permute(0, 4, 1, 2, 3)
        
        return output

    def deform_conv(self, input, offset, if_offset):
        t, y, x = self._coordinate_map_temporal(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, t, y, x)
        return deformed_feature

# ================= 验证代码 =================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 严格检查 BCTHW 顺序
    # Batch=2, Channel=16, Time=8, Height=32, Width=32
    B, C, T, H, W = 2, 16, 8, 32, 32
    x = torch.randn(B, C, T, H, W).to(device)
    
    print(f"Input Shape: {x.shape} (B, C, T, H, W)")

    # 实例化
    # kernel_size_t=3: 意味着每个时间点会融合 t-1, t, t+1
    model = TemporalSnakeConv3d(in_ch=16, out_ch=32, kernel_size_t=3, device=device).to(device)
    
    # 前向传播
    out = model(x)
    
    print(f"Output Shape: {out.shape} (B, Out_C, T, H, W)")
    
    # 维度检查 assert
    assert out.shape[0] == B
    assert out.shape[1] == 32
    assert out.shape[2] == T  # 时间维度必须保持不变
    assert out.shape[3] == H  # 高度保持不变 (padding=1)
    assert out.shape[4] == W  # 宽度保持不变 (padding=1)
    print("Verification Passed: Dimensions are strictly BCTHW preserved.")