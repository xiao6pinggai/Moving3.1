import math
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, ReLU, Sigmoid

# ==========================================
# 1. 支持 Dilation 的 基础组件
# ==========================================

class SpatialDilatedCDC3d(nn.Module):
    """
    支持 Dilation 的 3D 空间中心差分卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, theta=0.7):
        super().__init__()
        # 计算 padding: (kernel_size - 1) * dilation // 2
        p_s = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=(1, kernel_size, kernel_size), 
                              stride=stride, 
                              padding=(0, p_s, p_s), 
                              dilation=(1, dilation, dilation),
                              bias=bias)
        self.theta = theta

    def forward(self, x):
        weights = self.conv.weight
        kernel_sum = weights.sum(dim=(3, 4), keepdim=True)
        
        cdc_weights = weights.clone()
        
        center_idx_h = self.conv.kernel_size[1] // 2
        center_idx_w = self.conv.kernel_size[2] // 2
        
        # 修正广播维度的 view 操作
        cdc_weights[:, :, 0, center_idx_h, center_idx_w] -= kernel_sum.view(weights.shape[0], weights.shape[1])
        
        final_weights = self.theta * weights + (1 - self.theta) * cdc_weights
        
        return F.conv3d(x, final_weights, self.conv.bias, 
                        self.conv.stride, self.conv.padding, self.conv.dilation)
class TemporalDilatedZeroSumConv3d(nn.Conv3d):
    """
    支持 Dilation 的 时域零和卷积 + 动态镜像 Padding
    初始化：中间帧权重为 n (kernel_size//2)，其余帧权重均匀分配负值使总和为 0
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        # 初始化时不设 padding，forward 里手动 pad
        super().__init__(in_channels, out_channels, kernel_size, stride, 
                         padding=0, dilation=dilation, bias=bias) 
        
        # 1. 计算 Padding
        self.pad_t = (kernel_size[0] - 1) * dilation[0] // 2
        self.pad_h = (kernel_size[1] - 1) * dilation[1] // 2
        self.pad_w = (kernel_size[2] - 1) * dilation[2] // 2
        
        # 2. 特殊时域权重初始化
        self._initialize_weights_zero_sum()

    def _initialize_weights_zero_sum(self):
        with torch.no_grad():
            kt, kh, kw = self.kernel_size
            n = kt // 2  # 中间权重值
            
            # 创建时域权重模板: [kt]
            # 例如 kt=3: [-0.5, 1.0, -0.5]
            # 例如 kt=5: [-0.5, -0.5, 2.0, -0.5, -0.5]
            t_weight = torch.full((kt,), -n / (kt - 1))
            t_weight[n] = float(n)
            
            # 将该模板扩展到整个卷积核形状 (Out, In, Kt, Kh, Kw)
            # 我们希望空间位置上保持一致，所以直接广播
            # 先重塑为 (1, 1, Kt, 1, 1)
            new_weight = t_weight.view(1, 1, kt, 1, 1).expand_as(self.weight)
            
            # 赋值并保持张量连续
            self.weight.copy_(new_weight)

    def forward(self, x):
        # 1. 权重零和约束 (训练过程中强制保持均值为0)
        # 虽然初始化已经是零和，但梯度更新会破坏它，所以 forward 仍需中心化
        w_mean = self.weight.mean(dim=2, keepdim=True)
        zero_sum_w = self.weight - w_mean
        
        # 2. 镜像 Padding
        if self.pad_t > 0:
            T_dim = x.shape[2]
            # 安全检查：如果 T 维度不够 reflect，降级为 replicate
            pad_mode = 'reflect' if self.pad_t < T_dim else 'replicate'
            x = F.pad(x, (0, 0, 0, 0, self.pad_t, self.pad_t), mode=pad_mode)
            
        # 3. 空间 Padding 并执行卷积
        return F.conv3d(x, zero_sum_w, self.bias, self.stride, 
                        (0, self.pad_h, self.pad_w), self.dilation, self.groups)
# ==========================================
# 2. 分组多分支模块 (Group Multi-Branch Block)
# ==========================================
# 定义一个封装了 abs 函数的 Module
class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
class MultiBranch_Grouped_Block(nn.Module):
    """
    新架构：Group Split -> Multi-Dilation Branch -> Concat -> Fusion
    """
    def __init__(self, in_feat, out_feat, kernel_dict=None, 
                 spatial_dilations=[1, 2], temporal_dilations=[1, 3], 
                 stride=1, padding=1, use_cdc=True, residual=None):
        super().__init__()
        
        if kernel_dict is None:
            kernel_dict = {'std': (3,3,3), 'spatial': 3, 'temporal': (3,1,1)} # T分支通常空间为1x1
            
        self.use_cdc = use_cdc
        
        # === Part 1: 特征提取 (升维/变换) ===
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=kernel_dict['std'], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )

        # === Part 2: 分支构建函数 ===
        
        def _get_splits(channels, num_branches):
            """使用 divmod 计算每个分支的通道数"""
            base, remainder = divmod(channels, num_branches)
            # 第一组拿走余数
            splits = [base + remainder] + [base] * (num_branches - 1)
            return splits

        # # --- Branch 1: 常规 3D (保持单路，作为基准特征) ---
        # self.b1_conv = nn.Sequential(
        #     nn.Conv3d(out_feat, out_feat, kernel_size=kernel_dict['std'], stride=1, padding=padding, bias=False),
        #     nn.BatchNorm3d(out_feat),
        #     nn.ReLU(inplace=True)
        # )
        # # self.b1_proj = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

        # --- Branch 2: 空间多分支 (Spatial Group) ---
        self.s_dilations = spatial_dilations
        self.s_splits = _get_splits(out_feat, len(spatial_dilations))
        self.b2_branches = nn.ModuleList()
        
        ks_s = kernel_dict['spatial']
        
        for idx, (ch, d) in enumerate(zip(self.s_splits, self.s_dilations)):
            if self.use_cdc:
                # CDC 模块内已计算 padding
                op = SpatialDilatedCDC3d(ch, ch, kernel_size=ks_s, stride=1, dilation=d, bias=False)
            else:
                p = (ks_s - 1) * d // 2
                op = nn.Conv3d(ch, ch, kernel_size=(1, ks_s, ks_s), stride=1, 
                               padding=(0, p, p), dilation=(1, d, d), bias=False)
            
            # 每个分支独立的 BN/ReLU
            branch = nn.Sequential(
                op,
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True)
            )
            self.b2_branches.append(branch)
            
        self.b2_proj = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

        # --- Branch 3: 时域多分支 (Temporal Group) ---
        self.t_dilations = temporal_dilations
        self.t_splits = _get_splits(out_feat, len(temporal_dilations))
        self.b3_branches = nn.ModuleList()
        
        ks_t = kernel_dict['temporal'] # e.g. (3, 1, 1)
        
        for idx, (ch, d) in enumerate(zip(self.t_splits, self.t_dilations)):
            # 构造 3D dilation tuple: (d_time, 1, 1)
            d_tuple = (d, 1, 1)
            # op = TemporalDilatedZeroSumConv3d(ch, ch, kernel_size=ks_t, stride=1, dilation=d_tuple, bias=False)
            op = nn.Conv3d(ch, ch, kernel_size=(3,3,3), stride=1, padding=1, bias=False)  # 接一个3x3x3卷积
            branch = nn.Sequential(
                op,
                nn.BatchNorm3d(ch),
                nn.ReLU(inplace=True)
                # Abs()  # 使用绝对值激活函数
            )
            self.b3_branches.append(branch)
            
        self.b3_proj = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=1, padding=1, bias=False),
            # nn.BatchNorm3d(out_feat),
            # nn.ReLU(inplace=True)
        )
        # === Part 3: 融合 ===
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(2*out_feat, out_feat, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )

        # 残差
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = nn.Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 1. Residual
        if self.residual == "conv":
            res = self.residual_conv(x)
        elif self.residual is not None:
            res = x
        else:
            res = None

        # 2. Conv1
        x = self.conv1(x) # [B, C, T, H, W]

        # 3. 多分支处理
        
        # --- B1: Standard 3D ---
        # 完整通道进去，完整通道出来
        # feat_b1 = self.b1_conv(x)
        # feat_b1 = self.b1_proj(feat_b1)
        
        # --- B2: Spatial Grouped Dilation ---
        # Split -> Process -> Concat
        x_splits_s = torch.split(x, self.s_splits, dim=1)
        b2_outs = []
        for chunk, branch in zip(x_splits_s, self.b2_branches):
            b2_outs.append(branch(chunk))
        feat_b2 = torch.cat(b2_outs, dim=1)
        # feat_b2 = self.b2_proj(feat_b2)
        
        # --- B3: Temporal Grouped Dilation ---
        x_splits_t = torch.split(x, self.t_splits, dim=1)
        b3_outs = []
        for chunk, branch in zip(x_splits_t, self.b3_branches):
            b3_outs.append(branch(chunk))
        feat_b3 = torch.cat(b3_outs, dim=1)
        # feat_b3 = self.b3_proj(feat_b3)
        
        # 4. Add & Fusion
        # buffer = b1 + b2 + b3
        # buffer = feat_b1 + feat_b2 + feat_b3
        buffer = torch.cat([feat_b2, feat_b3], dim=1)
        out = self.fusion_conv(buffer)
        
        if res is not None:
            out = out + res
            
        return out


# ==========================================
# 3. UNet 主体更新
# ==========================================

class Normal_Conv3D_Block(Module):
    """保留用于 Decoder 或 bottleneck"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        return self.conv2(self.conv1(x))

class Upsample3D_Block(Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride=2, padding=1, mode="trilinear", T_upsample=True):
        super().__init__()
        scale = (2, 2, 2) if T_upsample else (1, 2, 2)
        self.upsample = Sequential(
            nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=True),
            Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
            ReLU(inplace=True)
        )
    def forward(self, x):
        return self.upsample(x)

class UNet3DWithGroupedMultiBranch(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None, 
                 upsample_mode="trilinear", activation='sigmoid', T_pooling=True, downsample_mode="stride"):
        super().__init__()
        self.feat_channels = feat_channels
        self.activation = activation
        
        # 下采样构建函数
        def build_downsample_layer(channels):
            if downsample_mode == "stride":
                k, s, p = ((3,3,3), (2,2,2), (1,1,1)) if T_pooling else ((1,3,3), (1,2,2), (0,1,1))
                return Conv3d(channels, channels, kernel_size=k, stride=s, padding=p)
            elif downsample_mode == "maxpool":
                k, s = ((2,2,2), (2,2,2)) if T_pooling else ((1,2,2), (1,2,2))
                return MaxPool3d(kernel_size=k, stride=s)

        self.down1 = build_downsample_layer(feat_channels[0])
        self.down2 = build_downsample_layer(feat_channels[1])
        self.down3 = build_downsample_layer(feat_channels[2])

        # === 配置 Dilation 策略 ===
        # 越深层，感受野需求越大，可以适当增加 dilation
        k_dic = {'std': (3,3,3), 'spatial': 3, 'temporal': (3,1,1)}
        
        # Layer 1: 浅层关注细节，dilation 较小
        s_d1 = [1,2]
        t_d1 = [1]
        
        # Layer 2: 中层
        s_d2 = [1,2]
        t_d2 = [1] # 时域拉长
        
        # Layer 3: 深层，需要大感受野
        s_d3 = [1,2]
        t_d3 = [1]

        # Encoder Layers (使用新的分组多分支模块)
        self.enc_conv1 = MultiBranch_Grouped_Block(num_channels, feat_channels[0], kernel_dict=k_dic, 
                                                   spatial_dilations=s_d1, temporal_dilations=t_d1,
                                                   stride=1, padding=1, use_cdc=True)
        
        self.enc_conv2 = MultiBranch_Grouped_Block(feat_channels[0], feat_channels[1], kernel_dict=k_dic, 
                                                   spatial_dilations=s_d2, temporal_dilations=t_d2,
                                                   stride=1, padding=1, use_cdc=True)
        
        self.enc_conv3 = MultiBranch_Grouped_Block(feat_channels[1], feat_channels[2], kernel_dict=k_dic, 
                                                   spatial_dilations=s_d3, temporal_dilations=t_d3,
                                                   stride=1, padding=1, use_cdc=True)

        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(feat_channels[2], feat_channels[3])

        # Decoder Layers (保持普通卷积以减轻计算量)
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(2 * feat_channels[3], feat_channels[3])
            self.upsample3 = Upsample3D_Block(feat_channels[3], feat_channels[2], T_upsample=T_pooling)

        self.dec_conv3 = Normal_Conv3D_Block(2 * feat_channels[2], feat_channels[2])
        self.dec_conv2 = Normal_Conv3D_Block(2 * feat_channels[1], feat_channels[1])
        self.dec_conv1 = Normal_Conv3D_Block(2 * feat_channels[0], feat_channels[0])

        self.upsample2 = Upsample3D_Block(feat_channels[2], feat_channels[1], T_upsample=T_pooling)
        self.upsample1 = Upsample3D_Block(feat_channels[1], feat_channels[0], T_upsample=T_pooling)

        # Final Head
        self.final_conv = Conv3d(feat_channels[0], num_classes, kernel_size=1, bias=True)
        # 初始化 Bias 优化
        self.final_conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
        self.final_conv.weight.data.normal_(0, 0.01)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # ---------------------------------------------------
        # 情况 A: 4层特征 (即 3次下采样 + 1个瓶颈层)
        # ---------------------------------------------------
        if len(self.feat_channels) - 1 == 3:
            # === Encoder ===
            enc1 = self.enc_conv1(x)                # [B, C0, D, H, W]
            # enc1 = enc1 + self.atdc1(enc1)
            down1 = self.down1(enc1)                # Downsample

            enc2 = self.enc_conv2(down1)            # [B, C1, D/s, H/2, W/2]
            # enc2 = enc2 + self.atdc(enc2)
            # del down1
            down2 = self.down2(enc2)                # Downsample

            enc3 = self.enc_conv3(down2)            # [B, C2, D/s2, H/4, W/4]
            # del down2
            down3 = self.down3(enc3)                # Downsample

            # === Bottleneck ===
            bottleneck = self.enc_conv4(down3)      # [B, C3, D/s3, H/8, W/8]
            # del down3
            # === Decoder ===
            tmp = self.upsample3(bottleneck)
            # tmp = self._align_feature_size(tmp, enc3) 
            tmp = torch.cat([tmp, enc3], dim=1)
            # del enc3
            tmp = self.dec_conv3(tmp)

            tmp = self.upsample2(tmp)
            # tmp = self._align_feature_size(tmp, enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            # del enc2
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            # tmp = self._align_feature_size(tmp, enc1)
            tmp = torch.cat([tmp, enc1], dim=1)
            # del enc1
            tmp = self.dec_conv1(tmp)

            tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            return tmp

        # ---------------------------------------------------
        # 情况 B: 3层特征 (即 2次下采样 + 1个瓶颈层)
        # ---------------------------------------------------
        elif len(self.feat_channels) - 1 == 2:
            enc1 = self.enc_conv1(x)
            down1 = self.down1(enc1)

            enc2 = self.enc_conv2(down1)
            # del down1
            down2 = self.down2(enc2)

            bottleneck = self.enc_conv3(down2)
            # del down2

            tmp = self.upsample2(bottleneck)
            # tmp = self._align_feature_size(tmp, enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            # del enc2
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            # tmp = self._align_feature_size(tmp, enc1)
            tmp = torch.cat([tmp, enc1], dim=1)
            # del enc1
            tmp = self.dec_conv1(tmp)

            tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            return tmp
        
        else:
            raise ValueError("Unsupported feat_channels length")

# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 8, 64, 64).to(device) # B, C, T, H, W
    
    model = UNet3DWithGroupedMultiBranch(
        num_channels=3, 
        feat_channels=[16, 32, 64, 128],
        T_pooling=False
    ).to(device)
    
    # 打印每个 Block 的分支情况验证 divmod
    print("Layer 1 Spatial Splits:", model.enc_conv1.s_splits) # [6, 5] if 16->16, branches=2 -> 16//2=8. Wait. 
    # Logic check: Conv1 out is 16. s_dilations len=2. 16//2=8, rem=0. splits=[8,8].
    # Example 2: Layer 2 out is 32. s_dilations len=3. 32//3=10, rem=2. splits=[12, 10, 10].
    print("Layer 2 Spatial Splits:", model.enc_conv2.s_splits) 
    
    with torch.no_grad():
        out = model(x)
    print("Output shape:", out.shape)