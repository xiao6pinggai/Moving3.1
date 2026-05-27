import math
import time
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, BatchNorm2d, BatchNorm3d, MaxPool2d, MaxPool3d, ReLU, Sigmoid

# ==========================================
# 基础组件定义
# ==========================================


class SpatialDepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(SpatialDepthwiseSeparableConv3d, self).__init__()
        
        # --- 参数处理逻辑 ---
        
        # 1. 强制将 kernel_size 转换为 (1, K, K) 格式
        # 即使传入的是 int 或 tuple，也确保时域维度(T)为 1
        if isinstance(kernel_size, int):
            spatial_kernel = (1, kernel_size, kernel_size)
        else:
            # 如果传入的是 tuple (T, H, W) 或 (H, W)，取后两位作为空域，时域强制为 1
            spatial_kernel = (1, kernel_size[-2], kernel_size[-1])

        # 2. 强制将 stride 转换为 (1, S, S) 格式
        # 假设时域不做下采样，保持时间分辨率
        if isinstance(stride, int):
            spatial_stride = (1, stride, stride)
        else:
            spatial_stride = (1, stride[-2], stride[-1])

        # 3. 强制将 padding 转换为 (0, P, P) 格式
        # 因为时域核为 1，所以时域不需要 padding (0)，只在空域 padding
        if isinstance(padding, int):
            spatial_padding = (0, padding, padding)
        else:
            spatial_padding = (0, padding[-2], padding[-1])

        # --- 网络结构定义 ---

        # 第一步：Depthwise Convolution (只在空域进行)
        # groups = in_channels (每个通道独立卷积)
        # kernel = (1, K, K), padding = (0, P, P)
        self.depthwise = nn.Conv3d(
            in_channels, 
            in_channels,               # 输出通道 = 输入通道
            kernel_size=spatial_kernel, 
            stride=spatial_stride,
            padding=spatial_padding, 
            bias=bias, 
            groups=in_channels         # 关键：深度卷积
        )
        self.bn1 = nn.BatchNorm3d(in_channels, eps=0.001, momentum=0.1, affine=True)
        # self.relu1 = nn.ReLU(inplace=True)

        # 第二步：Pointwise Convolution (通道融合)
        # kernel = (1, 1, 1), groups = 1
        self.pointwise = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias, 
            groups=1                   # 关键：点卷积融合信息
        )
        self.bn2 = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Depthwise 部分
        x = self.depthwise(x)
        x = self.bn1(x)
        # x = self.relu1(x)
        
        # Pointwise 部分
        x = self.pointwise(x)
        x = self.bn2(x)
        # x = self.relu2(x)
        return x



class SingleBranchATDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2, stride=1, padding=1, bias=False):
        """
        Args:
            dilation (int): 固定空洞率，默认为 2。
            padding (int): 空间维度(H, W)的padding。
        """
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]  # 仅考虑时序维度的kernel_size
        if isinstance(stride, tuple):
            stride = stride[0]            # 仅考虑时序维度的stride
        if isinstance(padding, tuple):
            padding = padding[1]          # 仅考虑空间维度的padding

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # --- 1. 计算时序 Padding ---
        # 计算需要的时序 padding 数量
        # Pad = dilation * (kernel_size - 1) // 2
        self.time_pad = dilation * (kernel_size - 1) // 2
        
        # 空间 padding 由参数传入
        spatial_padding = padding 
        
        # --- 2. 构建单分支卷积 ---
        # 注意：padding设为(0, spatial, spatial)，时序维度(dim=2)由forward中手动pad
        self.conv = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=(kernel_size, 1, 1), 
                              stride=stride, 
                              padding=(0, spatial_padding, spatial_padding), 
                              dilation=(dilation, 1, 1), 
                              bias=bias)

        # --- 3. BN ---
        # self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def get_standardized_weight(self, conv_layer):
        """
        权重标准化：Zero-Mean + Unit-Variance
        实现“零和”特性：减去均值后，权重之和趋近于0。
        """
        W = conv_layer.weight
        
        # W shape: [out, in, D, H, W]
        # dim=2 对应时序维度 D。在此维度做归一化有助于提取时序差分特征。
        mean = W.mean(dim=2, keepdim=True)
        std = W.std(dim=2, keepdim=True) + 1e-5
        
        return (W - mean) / std

    def forward(self, x):
        # 1. 镜像填充 (Mirror Padding / Replicate Padding)
        # F.pad 参数顺序: (Left, Right, Top, Bottom, Front, Back)
        if self.time_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.time_pad, self.time_pad), mode='replicate')
        
        # 2. 获取标准化权重 (实现零和卷积核心逻辑)
        W_std = self.get_standardized_weight(self.conv)
        
        # 3. 执行卷积
        # 使用 functional 接口传入标准化后的权重
        out = F.conv3d(x, W_std, self.conv.bias, self.conv.stride, 
                       self.conv.padding, self.conv.dilation, self.conv.groups)

        # 4. BN
        # out = self.bn(out)
        
        return out
    

class Conv3D_Block(Module):
    """3D 卷积块"""
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
        
        # 深度可分离卷积 (保留结构，forward中未启用)
        self.spatialconv1 = Sequential(
            SpatialDepthwiseSeparableConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
            Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.spatialconv2 = Sequential(
            SpatialDepthwiseSeparableConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
            Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),  
            ReLU(inplace=True)  
        )

        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.spatialconv2(self.spatialconv1(x))
    

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
        
        # # 深度可分离卷积 (保留结构，forward中未启用)
        # self.spatialconv1 = Sequential(
        #     SpatialDepthwiseSeparableConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
        #     Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        #     BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
        #     ReLU(inplace=True)
        # )
        # self.spatialconv2 = Sequential(
        #     SpatialDepthwiseSeparableConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
        #     Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        #     BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),  
        #     ReLU(inplace=True)  
        # )

        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.conv2(self.conv1(x))

class Normal_Conv3D_Block_tdilation(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1,dilation=(2,1,1)):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False,dilation=(2,1,1)),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=1, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        
        # # 深度可分离卷积 (保留结构，forward中未启用)
        # self.spatialconv1 = Sequential(
        #     SpatialDepthwiseSeparableConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
        #     Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        #     BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
        #     ReLU(inplace=True)
        # )
        # self.spatialconv2 = Sequential(
        #     SpatialDepthwiseSeparableConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
        #     Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        #     BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),  
        #     ReLU(inplace=True)  
        # )

        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.conv2(self.conv1(x))
# 定义一个封装了 abs 函数的 Module
class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
class Conv3D_Block_with_ATDC(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()

        
        # 深度可分离卷积 (保留结构，forward中未启用)
        self.spatialconv1 = Sequential(
            SpatialDepthwiseSeparableConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
            
        )
        self.spatialconv2 = Sequential(
            SpatialDepthwiseSeparableConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False),
            
        )
        self.atdcconv1 = Sequential(
            SingleBranchATDC(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            Abs()
        )
        self.atdcconv2 = Sequential(    
            SingleBranchATDC(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),  
            Abs()
        )
        self.temporalconv1 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.temporalconv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积
        s1 = self.spatialconv1(x)
        t1 = self.temporalconv1(s1)
        s2 = self.spatialconv2(t1)  
        t2 = self.temporalconv2(s2) 
        atdc1 = self.atdcconv1(s1)
        atdc2 = self.atdcconv2(atdc1)
        del s1, s2, atdc1
        return t2+atdc2

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


class Normal_Conv2D_Block(Module):
    """2D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm2d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv2d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm2d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )

        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Upsample2D_Block(Module):
    """2D 上采样块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=2, padding=1, mode="trilinear"):
        super().__init__()
        self.mode = mode

        if mode == "deconv":
            self.upsample = Sequential(
                ConvTranspose2d(
                    in_feat,
                    out_feat,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=1,
                    bias=True,
                ),
                ReLU(inplace=True)
            )
        elif mode == "trilinear":
            self.upsample = Sequential(
                nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
                Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
                ReLU(inplace=True)
            )
        else:
            raise ValueError(f"不支持的上采样模式: {mode}")

    def forward(self, x):
        return self.upsample(x)


# ==========================================
# 主网络结构 UNet3D
# ==========================================

class UNet2DWithNormalConv2D(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None,
                 upsample_mode="trilinear", dropout_prob=0, activation='sigmoid',
                 T_pooling=True, groups=2, downsample_mode="stride", use_final_conv=True):
        """
        2D 版本 UNet。
        输入输出接口与 UNet3DWithNormalConv3D 保持一致：
        [B, C, T, H, W] -> [B, C_out, T, H, W]
        """
        super().__init__()

        self.feat_channels = feat_channels
        self.upsample_mode = upsample_mode
        self.activation = activation
        self.use_final_conv = use_final_conv

        def build_downsample_layer(channels):
            if downsample_mode == "stride":
                return Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1)
            elif downsample_mode == "maxpool":
                return MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

        self.down1 = build_downsample_layer(feat_channels[0])
        self.down2 = build_downsample_layer(feat_channels[1])
        self.down3 = build_downsample_layer(feat_channels[2])

        self.enc_conv1 = Normal_Conv2D_Block(num_channels, feat_channels[0], kernel=(3, 3), stride=1, padding=(1, 1))
        self.enc_conv2 = Normal_Conv2D_Block(feat_channels[0], feat_channels[1], kernel=(3, 3), stride=1, padding=(1, 1))
        self.enc_conv3 = Normal_Conv2D_Block(feat_channels[1], feat_channels[2], kernel=(3, 3), stride=1, padding=(1, 1))

        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv2D_Block(feat_channels[2], feat_channels[3], kernel=(3, 3), stride=1, padding=(1, 1))

        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv2D_Block(2 * feat_channels[3], feat_channels[3], kernel=(3, 3), stride=1, padding=(1, 1))

        self.dec_conv3 = Normal_Conv2D_Block(2 * feat_channels[2], feat_channels[2], kernel=(3, 3), stride=1, padding=(1, 1))
        self.dec_conv2 = Normal_Conv2D_Block(2 * feat_channels[1], feat_channels[1], kernel=(3, 3), stride=1, padding=(1, 1))
        self.dec_conv1 = Normal_Conv2D_Block(2 * feat_channels[0], feat_channels[0], kernel=(3, 3), stride=1, padding=(1, 1))

        if len(self.feat_channels) >= 4:
            self.upsample3 = Upsample2D_Block(feat_channels[3], feat_channels[2], mode=upsample_mode)

        self.upsample2 = Upsample2D_Block(feat_channels[2], feat_channels[1], mode=upsample_mode)
        self.upsample1 = Upsample2D_Block(feat_channels[1], feat_channels[0], mode=upsample_mode)

        self.final_conv = Conv2d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.final_conv.bias.data.fill_(bias_value)
        self.final_conv.weight.data.normal_(0, 0.01)

        self.sigmoid = Sigmoid()

    def _align_feature_size(self, x, ref_tensor):
        x_size = x.shape[2:]
        ref_size = ref_tensor.shape[2:]
        if x_size == ref_size:
            return x
        return F.interpolate(x, size=ref_size, mode='bilinear', align_corners=True)

    def _forward_2d(self, x):
        if len(self.feat_channels) - 1 == 3:
            enc1 = self.enc_conv1(x)
            down1 = self.down1(enc1)

            enc2 = self.enc_conv2(down1)
            down2 = self.down2(enc2)

            enc3 = self.enc_conv3(down2)
            down3 = self.down3(enc3)

            bottleneck = self.enc_conv4(down3)

            tmp = self.upsample3(bottleneck)
            tmp = torch.cat([tmp, enc3], dim=1)
            tmp = self.dec_conv3(tmp)

            tmp = self.upsample2(tmp)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            tmp = torch.cat([tmp, enc1], dim=1)
            tmp = self.dec_conv1(tmp)

            if self.use_final_conv:
                tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            return tmp

        elif len(self.feat_channels) - 1 == 2:
            enc1 = self.enc_conv1(x)
            down1 = self.down1(enc1)

            enc2 = self.enc_conv2(down1)
            down2 = self.down2(enc2)

            bottleneck = self.enc_conv3(down2)

            tmp = self.upsample2(bottleneck)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            tmp = torch.cat([tmp, enc1], dim=1)
            tmp = self.dec_conv1(tmp)

            if self.use_final_conv:
                tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            return tmp

        else:
            raise ValueError("Unsupported feat_channels length")

    def forward(self, x):
        b, c, t, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        out_2d = self._forward_2d(x_2d)
        _, out_c, out_h, out_w = out_2d.shape
        return out_2d.view(b, t, out_c, out_h, out_w).permute(0, 2, 1, 3, 4).contiguous()

class UNet3DWithNormalConv3D(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None,
                 upsample_mode="trilinear", dropout_prob=0, activation='sigmoid',
                 T_pooling=True, groups=2, downsample_mode="stride", use_final_conv=True,
                 TConvOnly=False):
        """
        :param downsample_mode: "stride" (卷积下采样) 或 "maxpool" (最大池化下采样)
        :param use_final_conv: 是否经过 final_conv 输出，默认 True
        :param TConvOnly: True 时卷积核为 (3,1,1)，False 时为 (3,3,3)
        """
        super().__init__()

        self.feat_channels = feat_channels
        self.upsample_mode = upsample_mode
        self.activation = activation
        self.use_final_conv = use_final_conv

        # 根据 TConvOnly 选择卷积核大小和 padding
        if TConvOnly:
            conv_kernel = (3, 1, 1)
            conv_padding = (1, 0, 0)
        else:
            conv_kernel = (3, 3, 3)
            conv_padding = (1, 1, 1)
        
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
        self.enc_conv1 = Normal_Conv3D_Block(num_channels, feat_channels[0], kernel=conv_kernel, stride=1, padding=conv_padding)
        self.enc_conv2 = Normal_Conv3D_Block(feat_channels[0], feat_channels[1], kernel=conv_kernel, stride=1, padding=conv_padding)
        self.enc_conv3 = Normal_Conv3D_Block(feat_channels[1], feat_channels[2], kernel=conv_kernel, stride=1, padding=conv_padding)
        # self.atdc1 = MultiBranchATDCWithBN(feat_channels[0], feat_channels[0], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc2 = MultiBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc3 = MultiBranchATDCWithBN(feat_channels[2], feat_channels[2], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc = SingleBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, stride=1, padding=0, bias=False)
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(feat_channels[2], feat_channels[3], kernel=conv_kernel, stride=1, padding=conv_padding) # 瓶颈层
            # self.atdc4 = MultiBranchATDCWithBN(feat_channels[3], feat_channels[3], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(2 * feat_channels[3], feat_channels[3], kernel=conv_kernel, stride=1, padding=conv_padding)

        self.dec_conv3 = Normal_Conv3D_Block(2 * feat_channels[2], feat_channels[2], kernel=conv_kernel, stride=1, padding=conv_padding)
        self.dec_conv2 = Normal_Conv3D_Block(2 * feat_channels[1], feat_channels[1], kernel=conv_kernel, stride=1, padding=conv_padding)
        self.dec_conv1 = Normal_Conv3D_Block(2 * feat_channels[0], feat_channels[0], kernel=conv_kernel, stride=1, padding=conv_padding)

        # 解码器：上采样块
        if len(self.feat_channels) >= 4:
            self.upsample3 = Upsample3D_Block(feat_channels[3], feat_channels[2], mode=upsample_mode, T_upsample=T_pooling)
        
        self.upsample2 = Upsample3D_Block(feat_channels[2], feat_channels[1], mode=upsample_mode, T_upsample=T_pooling)
        self.upsample1 = Upsample3D_Block(feat_channels[1], feat_channels[0], mode=upsample_mode, T_upsample=T_pooling)

        # 最终输出头。use_final_conv=True 时输出 num_classes，False 时保留多通道特征。
        final_out_channels = num_classes if use_final_conv else feat_channels[0]
        self.final_conv = Conv3d(feat_channels[0], final_out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ######################################输出概率重置#######################################
        # 1. 设定先验概率 pi，通常取 0.01
        prior_prob = 0.01
        
        # 2. 计算对应的 bias 值
        # logit = log(pi / (1 - pi)) = -log((1 - pi) / pi)
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # 3. 初始化 Weight
        # 权重必须初始化为很小的高斯分布，确保 bias 占主导地位
        self.final_conv.weight.data.normal_(0, 0.01)

        if use_final_conv:
            # 单通道/类别输出沿用原来的低先验初始化。
            self.final_conv.bias.data.fill_(bias_value)
        else:
            # 多通道输出时，让通道均值具备相同低先验，
            # 但各通道保持轻微差异，避免完全相同的初始化。
            channel_bias_noise = torch.randn(final_out_channels) * 0.1
            channel_bias_noise = channel_bias_noise - channel_bias_noise.mean()
            self.final_conv.bias.data.copy_(bias_value + channel_bias_noise)
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
            bottleneck = self.enc_conv4(down3)    ###########32-->64################ # [B, C3, D/s3, H/8, W/8]
            # del down3
            # === Decoder ===
            tmp = self.upsample3(bottleneck) ##############64-->32#############
            # tmp = self._align_feature_size(tmp, enc3) 
            tmp = torch.cat([tmp, enc3], dim=1) ###########################
            # tmp = torch.cat([enc3, enc3], dim=1) ###########################
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


class EncoderOnlyConv3DProposalNet(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32, 64, 128, 256], residual=None,
                 upsample_mode="trilinear", dropout_prob=0, activation=None,
                 T_pooling=True, groups=2, downsample_mode="stride", use_final_conv=True,
                 TConvOnly=False):
        """
        Encoder-only 3D proposal network.
        It keeps standard Conv3D blocks in the encoder, removes the decoder,
        and predicts a low-resolution proposal logit map from the deepest stage.
        """
        super().__init__()

        if len(feat_channels) < 2:
            raise ValueError("feat_channels must contain at least two stages for encoder-only mode")

        self.feat_channels = feat_channels
        self.activation = activation
        self.use_final_conv = use_final_conv

        if TConvOnly:
            conv_kernel = (3, 1, 1)
            conv_padding = (1, 0, 0)
        else:
            conv_kernel = (3, 3, 3)
            conv_padding = (1, 1, 1)

        def build_downsample_layer(channels):
            if downsample_mode == "stride":
                if T_pooling:
                    k, s, p = (3, 3, 3), (2, 2, 2), (1, 1, 1)
                else:
                    k, s, p = (1, 3, 3), (1, 2, 2), (0, 1, 1)
                return Conv3d(channels, channels, kernel_size=k, stride=s, padding=p, groups=1)

            if downsample_mode == "maxpool":
                if T_pooling:
                    k, s = (2, 2, 2), (2, 2, 2)
                else:
                    k, s = (1, 2, 2), (1, 2, 2)
                return MaxPool3d(kernel_size=k, stride=s)

            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        in_channels = num_channels
        for idx, out_channels in enumerate(feat_channels):
            self.encoder_blocks.append(
                Normal_Conv3D_Block(
                    in_channels,
                    out_channels,
                    kernel=conv_kernel,
                    stride=1,
                    padding=conv_padding
                )
            )
            in_channels = out_channels
            if idx < len(feat_channels) - 1:
                self.down_blocks.append(build_downsample_layer(out_channels))

        self.proposal_head = Conv3d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.proposal_head.bias.data.fill_(bias_value)
        self.proposal_head.weight.data.normal_(0, 0.01)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        tmp = x
        for idx, encoder_block in enumerate(self.encoder_blocks):
            tmp = encoder_block(tmp)
            if idx < len(self.down_blocks):
                tmp = self.down_blocks[idx](tmp)

        proposal_logits = self.proposal_head(tmp) if self.use_final_conv else tmp
        if self.activation == 'sigmoid':
            proposal_logits = self.sigmoid(proposal_logits)

        return {
            'encoded_features': tmp,
            'proposal_logits': proposal_logits
        }


class LightWeightedConv3D(Module):
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None,
                 upsample_mode="trilinear", dropout_prob=0, activation='sigmoid',
                 T_pooling=True, groups=2, downsample_mode="stride", use_final_conv=True):
        """
        轻量级 3D 卷积网络。
        保持与 UNet3DWithNormalConv3D 相同的输入输出接口，但内部仅使用按 feat_channels
        定义的 Conv3D 顺序堆叠，不包含上采样、下采样与跳跃连接。
        """
        super().__init__()

        if len(feat_channels) == 0:
            raise ValueError("feat_channels must contain at least one channel definition")

        self.feat_channels = feat_channels
        self.activation = activation
        self.use_final_conv = use_final_conv

        conv_blocks = []
        in_channels = num_channels
        for out_channels in feat_channels:
            conv_blocks.append(
                Normal_Conv3D_Block(
                    in_channels,
                    out_channels,
                    kernel=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                )
            )
            in_channels = out_channels
        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.final_conv = Conv3d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.final_conv.bias.data.fill_(bias_value)
        self.final_conv.weight.data.normal_(0, 0.01)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        tmp = x
        for conv_block in self.conv_blocks:
            tmp = conv_block(tmp)

        if self.use_final_conv:
            tmp = self.final_conv(tmp)
        if self.activation == 'sigmoid':
            tmp = self.sigmoid(tmp)
        return tmp


            

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
        model = UNet3DWithNormalConv3D(
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