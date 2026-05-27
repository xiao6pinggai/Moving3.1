import math
import time
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, ReLU, Sigmoid

# ==========================================
# 基础组件定义
# ==========================================



class Normal_Conv3D_Block(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False,),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        
        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.conv2(self.conv1(x))

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class ATDC_Parallel_Fusion_Block(nn.Module):
    """
    ATDC 并行融合模块 (Adaptive Temporal Decoupling Convolution Block)
    
    改进版结构特点:
    1. 双路并行: 
       - [支路 A] (Decoupled Zero-Sum): 
            Step 1: 时域零和卷积 (Kt*1*1) -> BN -> LeakyReLU (保留运动方向)
            Step 2: 空域常规卷积 (1*Kh*Kw) -> BN -> ReLU (空间聚合)
       - [支路 B] (Standard): 常规 3D 卷积 (Kt*Kh*Kw) -> BN -> ReLU
    2. 软门控机制 (Sigmoid p):
       - 动态调节两路特征的融合比例。
    3. 后置特征提取:
       - 融合后再次卷积整合。
    """
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        # --- 参数解析 ---
        # 兼容 int 或 tuple 输入: (t, h, w)
        kt = kernel[0] if isinstance(kernel, tuple) else kernel
        kh = kernel[1] if isinstance(kernel, tuple) else kernel
        kw = kernel[2] if isinstance(kernel, tuple) else kernel
        
        pt = padding[0] if isinstance(padding, tuple) else padding
        ph = padding[1] if isinstance(padding, tuple) else padding
        pw = padding[2] if isinstance(padding, tuple) else padding

        st = stride[0] if isinstance(stride, tuple) else stride
        sh = stride[1] if isinstance(stride, tuple) else stride
        sw = stride[2] if isinstance(stride, tuple) else stride

        # 1. 定义可学习参数 p
        # 初始化为 -2.1，Sigmoid ≈ 0.1，既有冷启动保护，又有一定梯度
        self.p = nn.Parameter(torch.tensor([-2.1]))

        # 2. [支路 A]: 解耦的零和卷积支路 (Decoupled Zero-Sum Branch)
        # Part 1: 时域零和 (Temporal Zero-Sum)
        # 作用: 提取差分/运动，使用 LeakyReLU 保留正负号(方向)
        self.zs_temp_conv = nn.Conv3d(in_feat, out_feat, kernel_size=(kt, 1, 1), 
                                      stride=(st, 1, 1), padding=(pt, 0, 0), bias=bias)
        self.zs_temp_bn = nn.BatchNorm3d(out_feat)
        self.zs_temp_act = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 使用 LeakyReLU
        # self.zs_temp_act = Abs()  # 使用 Abs 保留幅值信息

        # Part 2: 空域聚合 (Spatial Aggregation)
        # 作用: 聚合时域提取出的运动能量/特征
        # self.zs_spatial_conv = nn.Conv3d(out_feat, out_feat, kernel_size=(1, kh, kw),
        #                                  stride=(1, sh, sw), padding=(0, ph, pw), bias=bias)
        self.zs_spatial_conv = nn.Conv3d(out_feat, out_feat, kernel_size=(1, 5, 5),
                                         stride=(1, sh, sw), padding=(0, 2, 2), bias=bias)
        self.zs_spatial_bn = nn.BatchNorm3d(out_feat)
        self.zs_spatial_act = nn.ReLU(inplace=True) # 空间层后用 ReLU

        # 3. [支路 B]: 常规卷积支路 (Standard Branch)
        # 保持原本的 3D 卷积结构，提取外观特征
        self.conv_std = nn.Conv3d(in_feat, out_feat, kernel, stride, padding, bias=bias)
        self.bn_std = nn.BatchNorm3d(out_feat)
        self.relu_std = nn.ReLU(inplace=True)

        # 4. 后置特征提取层 (Post Fusion Conv)
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        # 1. 常规支路 (Standard Branch): Kaiming Init
        nn.init.kaiming_normal_(self.conv_std.weight, mode='fan_out', nonlinearity='relu')
        
        # 2. 零和支路 (Zero-Sum Branch)
        # --- Part 1: 时域卷积 (Center Pulse Init) ---
        nn.init.constant_(self.zs_temp_conv.weight, 0)
        k_t = self.zs_temp_conv.kernel_size[0]
        # 仅在时域中心设为 1.0 (去均值后会自动变成差分算子)
        with torch.no_grad():
            self.zs_temp_conv.weight[:, :, k_t // 2, 0, 0] = 1.0 
        
        # --- Part 2: 空域卷积 (Kaiming Init) ---
        # 空间部分是常规聚合，正常初始化即可
        nn.init.kaiming_normal_(self.zs_spatial_conv.weight, mode='fan_out', nonlinearity='relu')

        # 3. 后置卷积
        nn.init.kaiming_normal_(self.post_conv[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # -----------------------------------------------
        # Step 1: 计算调和系数 alpha (带梯度放大 Hook)
        # -----------------------------------------------
        p_in = self.p * 1.0
        if self.training:
            p_in.register_hook(lambda grad: grad * 10.0) # 梯度放大 10 倍
        alpha = torch.sigmoid(p_in)

        # -----------------------------------------------
        # Step 2: 支路 A (解耦零和 - Decoupled Zero Sum)
        # -----------------------------------------------
        # >> Sub-step 2.1: 时域零和卷积 <<
        w_zs = self.zs_temp_conv.weight
        # 核心约束：仅在时域维度 (dim=2) 去均值
        w_zs_centered = w_zs - w_zs.mean(dim=2, keepdim=True)
        
        # 时域卷积
        out_zs = F.conv3d(x, w_zs_centered, self.zs_temp_conv.bias, 
                          self.zs_temp_conv.stride, self.zs_temp_conv.padding, 
                          self.zs_temp_conv.dilation, self.zs_temp_conv.groups)
        out_zs = self.zs_temp_bn(out_zs)
        out_zs = self.zs_temp_act(out_zs) # LeakyReLU (保留方向信息)
        
        # >> Sub-step 2.2: 空域常规卷积 <<
        out_zs = self.zs_spatial_conv(out_zs)
        out_zs = self.zs_spatial_bn(out_zs)
        out_zs = self.zs_spatial_act(out_zs) # ReLU (最终特征非负)

        # -----------------------------------------------
        # Step 3: 支路 B (常规卷积 - Standard)
        # -----------------------------------------------
        out_std = self.conv_std(x)
        out_std = self.bn_std(out_std)
        out_std = self.relu_std(out_std)

        # -----------------------------------------------
        # Step 4: 加权融合 (Weighted Fusion)
        # -----------------------------------------------
        merged = alpha * out_zs + (1.0 - alpha) * out_std

        # -----------------------------------------------
        # Step 5: 后置特征提取
        # -----------------------------------------------
        out = self.post_conv(merged)
        
        return out


class SEBlock3D(nn.Module):
    """ 3D Squeeze-and-Excitation Block """
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ATDC_MultiScale_Fusion_Block(nn.Module):
    """
    ATDC 多尺度时空融合模块 (Modified Version)
    
    结构流程:
    1. Input -> [时域零和卷积 (Kt*1*1)] -> BN -> LeakyReLU
    2. -> [空域多路空洞卷积 (1*Kh*Kw, d=1,2,3)] -> Concat
    3. -> [1*1*1 卷积降维] -> [SE Block]
    4. -> [Shortcut Add (Input)]
    5. -> [常规 3D 卷积 (Kt*Kh*Kw)] -> Output
    """
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        # --- 参数解析 ---
        # 兼容 int 或 tuple 输入: (t, h, w)
        kt = kernel[0] if isinstance(kernel, tuple) else kernel
        kh = kernel[1] if isinstance(kernel, tuple) else kernel
        kw = kernel[2] if isinstance(kernel, tuple) else kernel
        
        # 基础 padding (用于最后的常规卷积)
        pt = padding[0] if isinstance(padding, tuple) else padding
        ph = padding[1] if isinstance(padding, tuple) else padding
        pw = padding[2] if isinstance(padding, tuple) else padding

        # Stride
        st = stride[0] if isinstance(stride, tuple) else stride
        sh = stride[1] if isinstance(stride, tuple) else stride
        sw = stride[2] if isinstance(stride, tuple) else stride

        # ------------------------------------------------------------------
        # Step 1: 时域零和卷积 (Kt * 1 * 1)
        # ------------------------------------------------------------------
        # 这里的 stride 负责处理时域下采样 (st)，空域保持不变 (1, 1)
        self.zs_temp_conv = nn.Conv3d(in_feat, out_feat, kernel_size=(kt, 1, 1), 
                                      stride=(st, 1, 1), padding=(pt, 0, 0), bias=bias)
        self.zs_temp_bn = nn.BatchNorm3d(out_feat)
        self.zs_temp_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # ------------------------------------------------------------------
        # Step 2: 多尺度空域空洞卷积 (Dilated Spatial Convs)
        # ------------------------------------------------------------------
        # 仅处理空间维度 (1, kh, kw)，时域维度为 1
        # 注意: 为了保证拼接，这里的 spatial stride 设为 (sh, sw)
        # padding = dilation * (kernel_size - 1) / 2
        
        # Branch 1: Dilation = 1 (常规)
        pad_d1 = (kh - 1) // 2
        self.spatial_d1 = nn.Conv3d(out_feat, out_feat, kernel_size=(1, kh, kw),
                                    stride=(1, sh, sw), padding=(0, pad_d1, pad_d1), 
                                    dilation=(1, 1, 1), bias=bias)
        self.bn_d1 = nn.BatchNorm3d(out_feat)

        # Branch 2: Dilation = 2
        pad_d2 = 2 * (kh - 1) // 2
        self.spatial_d2 = nn.Conv3d(out_feat, out_feat, kernel_size=(1, kh, kw),
                                    stride=(1, sh, sw), padding=(0, pad_d2, pad_d2), 
                                    dilation=(1, 2, 2), bias=bias)
        self.bn_d2 = nn.BatchNorm3d(out_feat)

        # Branch 3: Dilation = 3
        pad_d3 = 3 * (kh - 1) // 2
        self.spatial_d3 = nn.Conv3d(out_feat, out_feat, kernel_size=(1, kh, kw),
                                    stride=(1, sh, sw), padding=(0, pad_d3, pad_d3), 
                                    dilation=(1, 3, 3), bias=bias)
        self.bn_d3 = nn.BatchNorm3d(out_feat)
        
        # 激活函数 (空间卷积后)
        self.spatial_act = nn.ReLU(inplace=True)

        # ------------------------------------------------------------------
        # Step 3: 降维与注意力 (Fusion & Attention)
        # ------------------------------------------------------------------
        # 拼接后通道数变为 3 * out_feat，通过 1x1x1 降回 out_feat
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_feat * 3, out_feat, kernel_size=1, bias=bias),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
        
        # SE 模块
        self.se_block = SEBlock3D(out_feat, reduction=16)

        # ------------------------------------------------------------------
        # Shortcut 处理 (Downsample)
        # ------------------------------------------------------------------
        # 如果 stride > 1 或者 输入输出通道不一致，Shortcut 需要投影
        self.downsample = None
        if (st != 1 or sh != 1 or sw != 1) or (in_feat != out_feat):
            self.downsample = nn.Sequential(
                nn.Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_feat)
            )

        # ------------------------------------------------------------------
        # Step 5: 最后的常规 3D 卷积
        # ------------------------------------------------------------------
        # 注意：这里 stride 设为 1，因为下采样已经在前面的层完成了
        # padding 设为 "same" (即传入的 padding)
        self.final_std_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        # 1. 时域零和卷积初始化 (Center Pulse)
        nn.init.constant_(self.zs_temp_conv.weight, 0)
        k_t = self.zs_temp_conv.kernel_size[0]
        with torch.no_grad():
            self.zs_temp_conv.weight[:, :, k_t // 2, 0, 0] = 1.0 
        
        # # 2. 空域多尺度卷积初始化 (Kaiming)
        # for m in [self.spatial_d1, self.spatial_d2, self.spatial_d3]:
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        # # 3. 融合与最后卷积初始化
        # for m in [self.fusion_conv[0], self.final_std_conv[0]]:
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
        # if self.downsample is not None:
        #      nn.init.kaiming_normal_(self.downsample[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x

        # -------------------------------------------------
        # Step 1: 时域零和卷积 (Temporal Zero-Sum)
        # -------------------------------------------------
        w_zs = self.zs_temp_conv.weight
        # 核心约束：在时域维度 (dim=2) 去均值，实现差分特性
        w_zs_centered = w_zs - w_zs.mean(dim=2, keepdim=True)
        
        out = F.conv3d(x, w_zs_centered, self.zs_temp_conv.bias, 
                       self.zs_temp_conv.stride, self.zs_temp_conv.padding, 
                       self.zs_temp_conv.dilation, self.zs_temp_conv.groups)
        out = self.zs_temp_bn(out)
        out = self.zs_temp_act(out) # LeakyReLU

        # -------------------------------------------------
        # Step 2: 空域多尺度 (Multi-Scale Spatial)
        # -------------------------------------------------
        # 三路并行
        y1 = self.bn_d1(self.spatial_d1(out))
        y2 = self.bn_d2(self.spatial_d2(out))
        y3 = self.bn_d3(self.spatial_d3(out))
        
        # 拼接 (Concat)
        out = torch.cat([y1, y2, y3], dim=1) # dim=1 is channel
        out = self.spatial_act(out)

        # -------------------------------------------------
        # Step 3: 降维与 SE (Reduce & SE)
        # -------------------------------------------------
        out = self.fusion_conv(out) # 1x1x1 conv 降维
        out = self.se_block(out)    # Channel Attention

        # -------------------------------------------------
        # Step 4: 残差连接 (Add Shortcut)
        # -------------------------------------------------
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = out + identity

        # -------------------------------------------------
        # Step 5: 最后的常规卷积 (Final Conv)
        # -------------------------------------------------
        out = self.final_std_conv(out)

        return out
class Upsample3D_Block(Module):
    """3D 上采样块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=2, padding=1, mode="trilinear", T_upsample=True):
        super().__init__()
        self.mode = mode
        
        if mode == "deconv":
            scale_D = 1 if T_upsample else 0
            stride_D = stride if T_upsample else 1
            kernel_D = kernel if T_upsample else 1
            
            self.upsample = Sequential(
                ConvTranspose3d(in_feat, out_feat, kernel_size=(kernel_D, kernel, kernel), 
                                stride=(stride_D, stride, stride), padding=(padding, padding, padding),
                                output_padding=(scale_D, 1, 1), bias=True),
                ReLU(inplace=True)
            )
        elif mode == "trilinear":
            scale = (2, 2, 2) if T_upsample else (1, 2, 2)
            self.upsample = Sequential(
                nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=True),
                Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
                ReLU(inplace=True)
            )
        else:
            raise ValueError(f"不支持的上采样模式: {mode}")

    def forward(self, x):
        return self.upsample(x)


# ==========================================
# 主网络结构 UNet3D
# ==========================================

class UNet3DWithNormalConv3DPATDCSplit(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None, 
                 upsample_mode="trilinear", dropout_prob=0, activation='sigmoid', 
                 T_pooling=True, groups=2, downsample_mode="stride"):
        """
        :param downsample_mode: "stride" (卷积下采样) 或 "maxpool" (最大池化下采样)
        """
        super().__init__()

        self.feat_channels = feat_channels
        self.upsample_mode = upsample_mode
        self.activation = activation
        
        # ========== 1. 构建下采样层 (根据模式选择) ==========
        def build_downsample_layer(channels):
            if downsample_mode == "stride":
                # Stride Convolution
                if T_pooling:
                    k, s, p = (3, 3, 3), (2, 2, 2), (1, 1, 1)
                else:
                    k, s, p = (1, 3, 3), (1, 2, 2), (0, 1, 1)
                # 使用 groups=1 保证信息融合，输入输出通道保持一致
                return Conv3d(channels, channels, kernel_size=k, stride=s, padding=p, groups=1)
            
            elif downsample_mode == "maxpool":
                # Max Pooling
                if T_pooling:
                    k, s = (2, 2, 2), (2, 2, 2)
                else:
                    k, s = (1, 2, 2), (1, 2, 2)
                return MaxPool3d(kernel_size=k, stride=s)
            
            else:
                raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

        self.down1 = build_downsample_layer(feat_channels[0])
        self.down2 = build_downsample_layer(feat_channels[1])
        self.down3 = build_downsample_layer(feat_channels[2])
        # =================================================

        # 编码器：卷积块
        self.enc_conv1 = Normal_Conv3D_Block(num_channels, feat_channels[0], kernel=(3,3,3), stride=1, padding=(1,1,1),)
        self.enc_conv2 = Normal_Conv3D_Block(feat_channels[0], feat_channels[1], kernel=(3,3,3), stride=1, padding=(1,1,1))
        self.enc_conv3 = Normal_Conv3D_Block(feat_channels[1], feat_channels[2], kernel=(3,3,3), stride=1, padding=(1,1,1))
        # self.atdc1 = MultiBranchATDCWithBN(feat_channels[0], feat_channels[0], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc2 = MultiBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc3 = MultiBranchATDCWithBN(feat_channels[2], feat_channels[2], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc = SingleBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, stride=1, padding=0, bias=False)
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(feat_channels[2], feat_channels[3], kernel=(3,3,3), stride=1, padding=(1,1,1)) # 瓶颈层
            # self.atdc4 = MultiBranchATDCWithBN(feat_channels[3], feat_channels[3], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(2 * feat_channels[3], feat_channels[3], kernel=(3,3,3), stride=1, padding=(1,1,1))
        
        self.dec_conv3 = Normal_Conv3D_Block(2 * feat_channels[2], feat_channels[2], kernel=(3,3,3), stride=1, padding=(1,1,1))
        self.dec_conv2 = Normal_Conv3D_Block(2 * feat_channels[1], feat_channels[1], kernel=(3,3,3), stride=1, padding=(1,1,1))
        self.dec_conv1 = Normal_Conv3D_Block(2 * feat_channels[0], feat_channels[0], kernel=(3,3,3), stride=1, padding=(1,1,1))

        # 解码器：上采样块
        if len(self.feat_channels) >= 4:
            self.upsample3 = Upsample3D_Block(feat_channels[3], feat_channels[2], mode=upsample_mode, T_upsample=T_pooling)
        
        self.upsample2 = Upsample3D_Block(feat_channels[2], feat_channels[1], mode=upsample_mode, T_upsample=T_pooling)
        self.upsample1 = Upsample3D_Block(feat_channels[1], feat_channels[0], mode=upsample_mode, T_upsample=T_pooling)

        # 最终分割头
        self.final_conv = Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        ######################################输出概率重置#######################################
        # 1. 设定先验概率 pi，通常取 0.01
        prior_prob = 0.01
        
        # 2. 计算对应的 bias 值
        # logit = log(pi / (1 - pi)) = -log((1 - pi) / pi)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        # 3. 初始化 Bias
        # 这会让网络初始输出的概率图全图接近 0.01，Loss 从很小开始，而不是从 5.0 开始
        self.final_conv.bias.data.fill_(bias_value)
        
        # 4. 初始化 Weight
        # 权重必须初始化为很小的高斯分布，确保 bias 占主导地位
        self.final_conv.weight.data.normal_(0, 0.01)
        ######################################输出概率重置#######################################
        self.sigmoid = Sigmoid()

    def _align_feature_size(self, x, ref_tensor):
        """
        强制对齐 x 的尺寸到 ref_tensor (解决奇数维度下采样不对齐问题)
        """
        x_size = x.shape[2:]
        ref_size = ref_tensor.shape[2:]
        if x_size == ref_size:
            return x
        return F.interpolate(x, size=ref_size, mode='trilinear', align_corners=True)

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
# 测试代码 (计算 FLOPs 和 Params)
# ==========================================
if __name__ == '__main__':
    try:
        from thop import profile
    except ImportError:
        print("请安装 thop: pip install thop")
        profile = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模拟输入：Batch=1, C=3, D=8, H=128, W=128 (适当减小尺寸以加快测试)
    input_tensor = torch.randn(1, 3, 10, 256, 256).to(device)

    # 4种情况测试
    configs = [
        {"T_pooling": True, "mode": "stride"},
        {"T_pooling": True, "mode": "maxpool"},
        {"T_pooling": False, "mode": "stride"},
        {"T_pooling": False, "mode": "maxpool"},
    ]
    configs = [configs[2]]
    print(f"{'T_pool':<10} | {'Mode':<10} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'Check D'}")
    print("-" * 65)

    for cfg in configs:
        model = UNet3DWithNormalConv3DPATDCSplit(
            num_channels=3,
            feat_channels=[16, 32, 64, 128], # 轻量化通道用于测试
            upsample_mode="trilinear",
            T_pooling=cfg["T_pooling"],
            downsample_mode=cfg["mode"],
            
        ).to(device).eval()

        # 运行一次检查输出尺寸
        with torch.no_grad():
            time1 =time.time()
            output = model(input_tensor)
            time2 =time.time()
            print("inference time:",time2-time1)
        
        # 验证维度 D
        out_d = output.shape[2]
        in_d = input_tensor.shape[2]
        check_msg = f"{in_d}->{out_d}"
        if cfg["T_pooling"] and out_d == in_d: check_msg += " (Warning!)"
        if not cfg["T_pooling"] and out_d != in_d: check_msg += " (Error!)"

        # 计算计算量 (如果装了 thop)
        if profile:
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
        else:
            flops_g, params_m = 0, 0

        print(f"{str(cfg['T_pooling']):<10} | {cfg['mode']:<10} | {params_m:<12.4f} | {flops_g:<12.4f} | {check_msg}")