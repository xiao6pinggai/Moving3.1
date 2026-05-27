import math
import time
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, ReLU, Sigmoid
import sys
ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from lib.models.TSConv_CUDA import TSConv_CUDA
# ==========================================
# 基础组件定义
# ==========================================


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
        

        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.spatialconv2(self.spatialconv1(x))
    

class Normal_Conv3D_Block(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, bias=False):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        
       

        self.residual = residual
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=bias)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积

        return self.conv2(self.conv1(x))

class TSC3D_Block(Module):
    """3D 卷积块：一层标准空域卷积 + 一层时域 TSC"""
    def __init__(self, in_feat, out_feat, kernel=3, kernel_t=3, stride=1, padding=1, residual=None, bias=False, extend_scope=None):
        super().__init__()
        # 单层标准卷积
        self.conv = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )
        # 默认 extend_scope
        if extend_scope is None:
            extend_scope = 3
        # 时域蛇形卷积（TSC）
        self.tsc = Sequential(
            TSConv_CUDA(
                in_channels=out_feat,
                out_channels=out_feat,
                kernel_size_t=kernel_t,
                kernel_size_s=kernel,
                padding=(kernel_t//2, padding, padding),
                bias=bias,
                extend_scope=extend_scope
            ),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True)
        )

        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return self.tsc(x)

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

class UNet3DWithTSC3D_CUDA(Module):
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
        # enc1: TSC with kt=3 and extend_scope=3 (suitable for small motion ~1-3 px)
        self.enc_conv1 = TSC3D_Block(num_channels, feat_channels[0], kernel=3, stride=1, padding=1, bias=False, kernel_t=3, extend_scope=3)
        # enc2: TSC with kt=5 and extend_scope=2 (captures slightly larger motion / temporal context)
        self.enc_conv2 = TSC3D_Block(feat_channels[0], feat_channels[1], kernel=3, stride=1, padding=1, bias=False, kernel_t=3, extend_scope=2)
        self.enc_conv3 = Normal_Conv3D_Block(feat_channels[1], feat_channels[2], kernel=3, stride=1, padding=1)
        
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(feat_channels[2], feat_channels[3], kernel=3, stride=1, padding=1) # 瓶颈层
            # self.atdc4 = MultiBranchATDCWithBN(feat_channels[3], feat_channels[3], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(2 * feat_channels[3], feat_channels[3], kernel=3, stride=1, padding=1)
        
        self.dec_conv3 = Normal_Conv3D_Block(2 * feat_channels[2], feat_channels[2], kernel=3, stride=1, padding=1)
        self.dec_conv2 = Normal_Conv3D_Block(2 * feat_channels[1], feat_channels[1], kernel=3, stride=1, padding=1)
        self.dec_conv1 = Normal_Conv3D_Block(2 * feat_channels[0], feat_channels[0], kernel=3, stride=1, padding=1)

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
    input_tensor = torch.randn(4, 3, 10, 256, 256).to(device)

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
        model = UNet3DWithTSC3D_CUDA(
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