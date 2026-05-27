
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class STCA_Block(nn.Module):
    def __init__(self, channels, kernel_size=12, use_ffn=False, norm_layer=nn.Identity):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.use_ffn = use_ffn
        
        # 1. 投影层
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        
        # 2. 相对位置编码
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        nn.init.trunc_normal_(self.pos_bias, std=0.02) # 使用截断正态分布初始化更稳定

        # 3. 输出投影
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 4. 门控缩放因子 (关键：防止 mean 导致数值过小)
        self.gate_scale = nn.Parameter(torch.ones(1) * 2.0) 

        # 5. FFN & Norm
        self.norm1 = norm_layer(channels)
        if use_ffn:
            self.norm2 = norm_layer(channels)
            self.ffn = nn.Sequential(
                nn.Conv2d(channels, channels*4, 1),
                nn.GELU(),
                nn.Conv2d(channels*4, channels, 1)
            )

    def extract_windows(self, x): # B C H W
        """利用 Unfold 提取局部窗口 K (Same Padding)"""
        pad = self.kernel_size // 2
        pad_l = pad_r = pad_t = pad_b = pad
        if self.kernel_size % 2 == 0:
            pad_r -= 1
            pad_b -= 1
            
        x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        # [B, C, H, W] -> [B, C*K*K, HW]
        windows = F.unfold(x_pad, kernel_size=self.kernel_size)
        B, _, HW = windows.shape
        
        # Reshape -> [B, HW, K, K, C]
        # 注意 view 的顺序必须匹配 unfold 的内存布局
        windows = windows.view(B, self.channels, self.kernel_size**2, HW)
        windows = windows.permute(0, 3, 2, 1) # [B, HW, K*K, C]
        windows = windows.view(B, HW, self.kernel_size, self.kernel_size, self.channels)
        return windows

    def compute_gating_map(self, curr, prev=None, next=None, is_edge=False):
        B, C, H, W = curr.shape
        
        # Q: [B, HW, 1, 1, C]
        q = self.to_q(curr).view(B, C, H*W).permute(0, 2, 1).view(B, H*W, 1, 1, C)
        
        if is_edge:
            # === 边缘帧：Self Attention ===
            k = self.extract_windows(self.to_k(curr))
            score = torch.sum(q * k, dim=-1) + self.pos_bias
            attn = torch.sigmoid(score)
            
            # 聚合
            gate_raw = torch.mean(attn, dim=[-2, -1])
            
        else:
            # === 中间帧：Symmetric Cross Attention ===
            k_prev = self.extract_windows(self.to_k(prev))
            k_next = self.extract_windows(self.to_k(next))
            
            score_prev = torch.sum(q * k_prev, dim=-1) + self.pos_bias
            score_next = torch.sum(q * k_next, dim=-1) + self.pos_bias
            
            attn_prev = torch.sigmoid(score_prev)
            attn_next = torch.sigmoid(score_next)
            
            # 核心：中心对称翻转
            attn_next_flipped = torch.flip(attn_next, dims=[-2, -1])
            
            # 核心：一致性乘积
            # 【修正点】使用 sqrt (几何平均) 来保持与边缘帧相似的量级
            consistency = torch.sqrt(attn_prev * attn_next_flipped + 1e-8)
            
            gate_raw = torch.mean(consistency, dim=[-2, -1])

        # Reshape 并应用可学习缩放
        gate = gate_raw.view(B, 1, H, W) * self.gate_scale
        return gate

    def forward(self, x, t_idx, x_prev=None, x_next=None):
        is_edge = (x_prev is None) or (x_next is None)
        
        # 1. 计算 Attention Gate
        gate_map = self.compute_gating_map(x, x_prev, x_next, is_edge)
        
        # 2. 施加门控 (Residual Gating)
        # x_enhanced = proj(x) * (1 + gate) 
        x_enhanced = self.proj(x) * (1 + gate_map)
        
        # Add & Norm
        out = self.norm1(x + x_enhanced)
        
        # 3. FFN
        if self.use_ffn:
            out = out + self.ffn(self.norm2(out))
            
        return out
    
class VideoSTCAModule(nn.Module):
    def __init__(self, channels, kernel_size=12):
        super().__init__()
        # 这里启用了 GroupNorm 替代 LayerNorm (更适合小 Batch 视觉任务)
        self.block = STCA_Block(
            channels, 
            kernel_size, 
            use_ffn=True, # 加上 FFN 增强拟合能力
            norm_layer=lambda c: nn.GroupNorm(num_groups=8, num_channels=c)
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        outputs = []
        
        for t in range(T):
            x_t = x[:, :, t]
            
            if t == 0:
                out = self.block(x_t, t, x_prev=None, x_next=None)
            elif t == T - 1:
                out = self.block(x_t, t, x_prev=None, x_next=None)
            else:
                x_prev = x[:, :, t-1]
                x_next = x[:, :, t+1]
                out = self.block(x_t, t, x_prev=x_prev, x_next=x_next)
                
            outputs.append(out)
            
        return torch.stack(outputs, dim=2)


if __name__ == "__main__":
    # 模拟输入: Batch=2, Channels=64, Time=5, H=32, W=32
    input_video = torch.randn(2, 32, 10, 256, 256)
    
    # 实例化模块
    model = VideoSTCAModule(channels=32, kernel_size=5)
    
    # 前向传播
    output_video = model(input_video)
    
    print(f"Input shape: {input_video.shape}")
    print(f"Output shape: {output_video.shape}")
    
    # 检查数值是否正常 (不应有 NaNs)
    if torch.isnan(output_video).any():
        print("Error: Output contains NaNs!")
    else:
        print("Success: Forward pass complete.")