import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class TemporalSnakeConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_t=3, kernel_size_s=3, stride=1, padding=1, bias=False, extend_scope=3.0, anchor_center=True, center_bias_learnable=False, center_bias_init=0.0):
        """
        时域蛇形卷积 (Temporal Snake Convolution)

        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_size_t: 时域卷积核长度 (必须为奇数, e.g., 3, 5)
            kernel_size_s: 空域卷积核大小 (e.g., 3)
            stride: 空间步长
            padding: 空间填充
            extend_scope: tanh 输出缩放因子（单位像素），控制偏移量范围
            anchor_center: 是否将 kernel 中心时间步固定为 0（锚点）
            center_bias_learnable: 当 anchor_center=False 时是否使用可学习的小偏置作为中心
            center_bias_init: 可学习中心偏置的初始值
        """
        super(TemporalSnakeConv3d, self).__init__()

        assert kernel_size_t % 2 == 1, "时域卷积核长度必须为奇数"
        self.kt = kernel_size_t
        self.kernel_size_t = kernel_size_t
        self.ks = kernel_size_s
        self.stride = stride
        # normalize padding: accept int or tuple of 3 (pad_t, pad_h, pad_w)
        # if isinstance(padding, int):
        #     self.pad_t = padding
        #     self.pad_h = padding
        #     self.pad_w = padding
        # elif isinstance(padding, (tuple, list)) and len(padding) == 3:
        #     self.pad_t, self.pad_h, self.pad_w = padding
        # else:
        #     raise ValueError('padding must be int or tuple/list of length 3')
        # convenience: keep a tuple too
        self.pad_t, self.pad_h, self.pad_w = self.kt // 2, self.ks//2, self.ks//2
        self.padding = (self.pad_t, self.pad_h, self.pad_w)
        self.center_idx = kernel_size_t // 2
        # extend_scope 控制 tanh 的放缩（以像素为单位），建议按特征尺度和数据位移设置
        self.extend_scope = float(extend_scope)
        # anchor center behavior
        self.anchor_center = bool(anchor_center)
        self.center_bias_learnable = bool(center_bias_learnable)
        if self.center_bias_learnable:
            self.center_bias = nn.Parameter(torch.full((1, 2, 1, 1, 1), float(center_bias_init)))
        else:
            self.register_parameter('center_bias', None)
        # last offsets storage
        self.last_snake_offset = None

        # 1. 偏移量生成网络 (Offset Generator)
        # 输入: B, C, T, H, W
        # 输出: B, 2*Kt, T, H, W (预测的是 frame-to-frame 的相对位移 delta)
        # 我们使用一个轻量级的 3D 卷积来感知时空运动
        self.offset_conv = nn.Conv3d(
            in_channels, 
            2 * kernel_size_t, 
            kernel_size=(3, 3, 3), 
            padding=(1, 1, 1), 
            bias=True
        )
        # 初始化使其接近 0，保证初始训练稳定性
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # 2. 卷积权重 (Convolution Weights)
        # 形状: (Out, In, Kt, Ks, Ks)
        # 我们将其存储为 Parameter，但在 forward 时会拆解给 2D DCN 使用
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size_t, kernel_size_s, kernel_size_s)
        )
        self.tsc_bias = nn.Parameter(torch.Tensor(out_channels))
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.tsc_bias, -bound, bound)

    def _generate_snake_offsets(self, x):
        """
        生成符合蛇形约束的连续偏移量
        """
        B, C, T, H, W = x.shape
        
        # 1. 预测相对偏移 (Velocity/Delta)
        # Shape: (B, 2*Kt, T, H, W)
        raw_deltas = self.offset_conv(x)

        # 限制幅度，防止训练初期飞出边界：使用 extend_scope 来控制最大偏移
        raw_deltas = torch.tanh(raw_deltas) * self.extend_scope

        # 将通道重排方便处理: (B, Kt, 2, T, H, W)
        deltas = raw_deltas.view(B, self.kt, 2, T, H, W)

        # 初始化最终的偏移量容器
        final_offsets = torch.zeros_like(deltas)

        # === 核心：Snake 累加机制 ===
        # 中心帧 (t) 的偏移量处理：
        if self.anchor_center:
            # 保持锚点为 0（默认做法）
            final_offsets[:, self.center_idx, ...] = 0
        else:
            # 允许中心学习：如果启用了可学习中心偏置则使用之，否则使用网络预测的 delta
            if self.center_bias_learnable and self.center_bias is not None:
                # broadcast to (1, 2, 1, 1, 1) -> (B, 2, T, H, W)
                cb = self.center_bias.view(1, 2, 1, 1, 1)
                final_offsets[:, self.center_idx, ...] = cb
            else:
                final_offsets[:, self.center_idx, ...] = deltas[:, self.center_idx, ...]
        
        # 向未来方向累加 (Forward Accumulation)
        # Offset[t+1] = Offset[t] + Delta[t->t+1]
        for i in range(self.center_idx + 1, self.kt):
            final_offsets[:, i] = final_offsets[:, i-1] + deltas[:, i]
            
        # 向过去方向累加 (Backward Accumulation)
        # Offset[t-1] = Offset[t] + Delta[t->t-1]
        for i in range(self.center_idx - 1, -1, -1):
            final_offsets[:, i] = final_offsets[:, i+1] + deltas[:, i]
            
        # 返回形状: (B, Kt, 2, T, H, W)
        return final_offsets, deltas

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            out: (B, Out_C, T, H_out, W_out)
        """
        B, C, T, H, W = x.shape
        
        # 1. 获取蛇形偏移量
        # snake_offsets: (B, Kt, 2, T, H, W)
        # raw_deltas: 用于计算平滑 Loss
        snake_offsets, raw_deltas = self._generate_snake_offsets(x)

        # 保存用于 Loss 计算 (如果需要)
        self.last_deltas = raw_deltas
        # 保存经过累加后的最终偏移量（供可视化/调试）, detach 到 cpu
        try:
            self.last_snake_offset = snake_offsets.detach().cpu()
        except Exception:
            self.last_snake_offset = None

        # 准备输出容器
        # 注意：这里我们假设 stride, padding 使得 T 维度不变，空间维度根据 conv 配置变化
        # 为了简化演示，这里对 H, W 做标准卷积尺寸计算
        h_out = (H + 2 * self.pad_h - self.ks) // self.stride + 1
        w_out = (W + 2 * self.pad_w - self.ks) // self.stride + 1
        
        output = torch.zeros(B, self.weight.shape[0], T, h_out, w_out, device=x.device)

        # 2. 逐时间步执行 DCN (Time-wise Loop)
        # 我们要计算输出特征图在时刻 t 的值 Output[:, :, t, :, :]
        # 这需要聚合 Input 在 [t - center, ..., t + center] 范围内的特征
        
        # 为了通过 DCN 处理，我们将 T 维度折叠进 Batch 维度，或者在循环中处理
        # 循环处理更直观，也更省显存
        
        for k in range(self.kt): # 5
            # k 是卷积核的时间索引 (0 ~ Kt-1)
            # relative_t 是相对于中心帧的时间差 (e.g., -1, 0, 1)
            relative_t = k - self.center_idx # -2, -1, 0, 1, 2
            
            # === A. 准备输入特征 (Input Feature Shifting) ===
            # 我们需要 Input 在 t + relative_t 时刻的特征
            # 为了矩阵并行，我们对 input 做简单的 roll 或者 slice
            # 但要注意边界问题（Padding T）。这里采用 Zero Padding 策略。
            
            # 计算当前 step 对应的 input indices
            # output t 对应 input t + relative_t
            # 如果 input index < 0 或 >= T，则为 padding
            
            # 构造切片逻辑：
            # 这是一个高效的 trick，不需要显式 pad 整个 5D tensor
            # 它通过调整 slice 的起点和终点，实现了“忽略越界部分”的效果，这等价于 Zero Padding
            valid_t_start = max(0, -relative_t) # 2 1 0 0 0
            valid_t_end = min(T, T - relative_t) # 5 4 5 5 5
            
            # 对应的 output 时间索引
            out_t_start = max(0, relative_t) # 0 0 0 1 2
            out_t_end = min(T, T + relative_t) # 3 4 5 5 5
            
            # 如果完全没有重叠（比如 kernel 很大 T 很小），跳过
            if valid_t_start >= valid_t_end:
                continue

            # 取出有效的输入切片: (B, C, T_valid, H, W)
            input_slice = x[:, :, valid_t_start:valid_t_end, :, :]  
            
            # 为了喂给 2D DCN，合并 B 和 T: (B*T_valid, C, H, W)
            input_reshape = input_slice.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            
            # === B. 准备偏移量 (Prepare Offsets) ===
            # 取出对应时间核 k 的偏移量
            # snake_offsets: (B, Kt, 2, T, H, W) -> 取第 k 个核 -> (B, 2, T, H, W)
            # 注意：Offset 是基于 Output 的时间网格定义的，所以我们取 out_t_start:out_t_end
            offset_slice = snake_offsets[:, k, :, out_t_start:out_t_end, :, :]
            
            # 同样合并 B 和 T: (B*T_valid, 2, H, W)
            offset_reshape = offset_slice.permute(0, 2, 1, 3, 4).reshape(-1, 2, H, W)
            
            # === C. 构造 DCN 所需的偏移格式 (Key Step for DCN) ===
            # DCN 要求 offset 形状: (Batch, 2 * Ks * Ks, H, W)
            # 这意味着每个 kernel 里的点可以有独立偏移。
            # 但你的需求是："以此为中心 ks*ks 卷积进行空域位置采样"
            # 这意味着：Kernel 内的相对位置不变（标准网格），只整体移动中心。
            # 所以我们要把同一个 (dx, dy) 复制 Ks*Ks 次。
            
            # (B*T, 2, H, W) -> (B*T, 2 * Ks*Ks, H, W)
            offset_final = offset_reshape.repeat(1, self.ks * self.ks, 1, 1)
            
            # === D. 执行 DCN ===
            # 取出当前时间步的权重: (Out, In, Ks, Ks)
            weight_k = self.weight[:, :, k, :, :]
            
            # 调用 torchvision DCN
            # torchvision.ops.deform_conv2d expects integer stride/padding values (not nested tuples)
            dcn_out = deform_conv2d(
                input_reshape,
                offset_final,
                weight_k,
                bias=None,
                stride=self.stride,
                padding=self.pad_h,  # pass spatial padding as integer
            )
            
            # === E. 累加回 Output Tensor ===
            # dcn_out: (B*T_valid, Out, H_out, W_out) -> (B, T_valid, Out, ...)
            dcn_out = dcn_out.view(B, -1, self.weight.shape[0], h_out, w_out).permute(0, 2, 1, 3, 4)
            
            # 加到对应的输出时间位置
            output[:, :, out_t_start:out_t_end, :, :] += dcn_out

        # 最后加上 Bias
        if self.bias:
            output += self.tsc_bias.view(1, -1, 1, 1, 1)
        
        return output

    def get_smoothness_loss(self):
        """
        计算轨迹平滑性 Loss (加速度最小化约束)
        逻辑：
           Delta_t 是速度 v_t
           Acceleration a_t = v_t - v_{t-1}
           我们希望 a_t -> 0
        """
        if not hasattr(self, 'last_deltas'):
            return torch.tensor(0.0)
            
        # last_deltas shape: (B, Kt, 2, T, H, W)
        # 这里的 Kt 维度实际上代表了相对于中心的时间偏移
        # 对于时域轨迹，我们需要计算的是“同一组轨迹”在不同时间步之间的平滑性
        # 但这里的实现其实是针对卷积核参数的。
        
        # 更合理的约束：
        # 对于同一个 spatial location (h, w) 和 time (t), 
        # 它的前向偏移 delta_forward (k > center) 和 后向偏移 delta_backward (k < center)
        # 应该呈现某种对称性或惯性。
        
        # 简化版惯性约束：相邻核索引的 Delta 应该相似
        # (B, Kt-1, 2, T, H, W)
        diff = self.last_deltas[:, 1:] - self.last_deltas[:, :-1]
        loss = torch.mean(diff ** 2)
        return loss

# ================= 调用示例 =================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # B=2, C=32, T=5, H=64, W=64
    x = torch.randn(2, 32, 10, 64, 64).to(device)
    
    # 定义时域蛇形卷积: 3D Snake, Temporal Kernel=3, Spatial Kernel=3
    tsc = TemporalSnakeConv3d(
        in_channels=32, 
        out_channels=64, 
        kernel_size_t=5, 
        kernel_size_s=1, 
        padding=0
    ).to(device)
    
    out = tsc(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    # 获取平滑 Loss
    loss_smooth = tsc.get_smoothness_loss()
    print(f"Smoothness Loss: {loss_smooth.item()}")