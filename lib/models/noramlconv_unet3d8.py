import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, ReLU, Sigmoid
import math
# ==========================================
# 基础组件定义
# ==========================================

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False, groups=1):
        super(BasicConv3d, self).__init__()
        # 自动调整 groups 以防止报错
        if in_channels % groups != 0:
            groups = 1
        if out_channels % groups != 0:
            groups = 1

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv3D_Block(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        
        # 深度可分离卷积 (保留结构，forward中未启用)
        self.DepthwiseSeparableConv3d1 = Sequential(
            BasicConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False, groups=groups),
            # BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False, groups=groups),
            # BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )

        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积
        return self.DepthwiseSeparableConv3d2(self.DepthwiseSeparableConv3d1(x))


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





class GroupTemporalAtrousPyramid2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        
        self.k = kernel_size[0]
        # 定义扩张率：第1分支正常卷积(d=1)，后3个分支为不同尺度的零和卷积
        self.d_normal = 1
        self.d_zero1 = 1
        self.d_zero2 = 2
        self.d_zero3 = 3
        
        # --- 1. 计算通道分配 (Split Channels by 4) ---
        # 依据您的指示："其余三个分支"，意味着总共有 1(普通) + 3(零和) = 4 个分支
        num_groups = 4
        
        # 输入通道分配
        in_div, in_rem = divmod(in_channels, num_groups)
        # 余数通道分配给第一个分支（外观分支）
        self.in_splits = [in_div + in_rem] + [in_div] * (num_groups - 1)
        
        # 输出通道分配
        out_div, out_rem = divmod(out_channels, num_groups)
        self.out_splits = [out_div + out_rem] + [out_div] * (num_groups - 1)
        
        # --- 2. 定义分支 ---
        # 注意：bias=False，因为后面紧跟 BN
        
        # [Branch 0] 普通卷积分支：捕捉外观 (Appearance)
        # Dilation=1, 不做 Zero-sum
        self.branch0 = nn.Conv3d(self.in_splits[0], self.out_splits[0], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d_normal, 1, 1), bias=False)
        
        # [Branch 1] 零和分支 1：捕捉快速运动 (Small Motion)
        self.branch1 = nn.Conv3d(self.in_splits[1], self.out_splits[1], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d_zero1, 1, 1), bias=False)
        
        # [Branch 2] 零和分支 2：捕捉中等运动 (Medium Motion)
        self.branch2 = nn.Conv3d(self.in_splits[2], self.out_splits[2], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d_zero2, 1, 1), bias=False)
        
        # [Branch 3] 零和分支 3：捕捉慢速运动 (Large Motion)
        self.branch3 = nn.Conv3d(self.in_splits[3], self.out_splits[3], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d_zero3, 1, 1), bias=False)

        # --- 3. 定义 BN 和 ReLU ---
        # 每个分支需要独立的 BN，因为特征统计分布不同 (一个是外观，一个是差分)
        self.bn0 = nn.BatchNorm3d(self.out_splits[0])
        self.bn1 = nn.BatchNorm3d(self.out_splits[1])
        self.bn2 = nn.BatchNorm3d(self.out_splits[2])
        self.bn3 = nn.BatchNorm3d(self.out_splits[3])
        
        # 激活函数
        # self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward_zero_sum_conv(self, x, conv_layer, dilation_t):
        """
        零和约束卷积流程：Reflect Pad -> Zero-sum Weight -> Conv
        """
        # 1. Pad
        pad_size = dilation_t
        x_padded = F.pad(x, (0, 0, 0, 0, pad_size, pad_size), mode='reflect')
        
        # 2. Zero-sum constraint
        w = conv_layer.weight # Cout   Cin/Group   Kt   Kh   Kw
        w_zero_sum = (w - w.mean(dim=[1,2,3,4], keepdim=True))/(w.std(dim=[1,2,3,4], keepdim=True) + 1e-5) # 应该比dim=2更稳定
        w_zero_sum = w - w.mean(dim=2, keepdim=True)
        out = F.conv3d(x_padded, w_zero_sum, 
                       bias=None, # bias由BN处理
                       stride=conv_layer.stride, 
                       padding=0, 
                       dilation=conv_layer.dilation, 
                       groups=conv_layer.groups)
        return out

    def forward_normal_conv(self, x, conv_layer, dilation_t):
        """
        普通卷积流程：Replicate Pad -> Conv (保留背景/外观)
        """
        # 1. Pad
        # 对于外观特征，使用 replicate (复制边缘) 通常比 reflect 更好，避免引入虚假边缘
        pad_size = dilation_t
        x_padded = F.pad(x, (0, 0, 0, 0, pad_size, pad_size), mode='replicate')
        
        # 2. Conv (直接使用原始权重)
        out = F.conv3d(x_padded, conv_layer.weight, 
                       bias=None, 
                       stride=conv_layer.stride, 
                       padding=0, 
                       dilation=conv_layer.dilation, 
                       groups=conv_layer.groups)
        return out

    def forward(self, x):
        # x: (B, C_in, T, H, W)
        
        # --- 1. 通道切分 (Split into 4 groups) ---
        x_splits = torch.split(x, self.in_splits, dim=1)
        
        # --- 2. 分支处理 ---
        
        # [Branch 0] Normal Conv (Appearance)
        out0 = self.forward_normal_conv(x_splits[0], self.branch0, dilation_t=self.d_normal)
        out0 = self.bn0(out0)
        out0 = self.leakyrelu(out0)
        
        # [Branch 1] Zero-sum (Motion Small)
        out1 = self.forward_zero_sum_conv(x_splits[1], self.branch1, dilation_t=self.d_zero1)
        out1 = self.bn1(out1)
        out1 = self.leakyrelu(out1) # 注意：如果是差分特征，LeakyReLU 其实比 ReLU 更好
        
        # [Branch 2] Zero-sum (Motion Medium)
        out2 = self.forward_zero_sum_conv(x_splits[2], self.branch2, dilation_t=self.d_zero2)
        out2 = self.bn2(out2)
        out2 = self.leakyrelu(out2)
        
        # [Branch 3] Zero-sum (Motion Large)
        out3 = self.forward_zero_sum_conv(x_splits[3], self.branch3, dilation_t=self.d_zero3)
        out3 = self.bn3(out3)
        out3 = self.leakyrelu(out3)
        
        # --- 3. 拼接 (Concat) ---
        
        return torch.cat([out0, out1, out2, out3], dim=1)

class Conv3D_Block_Group_ATDC_Dilation(Module):
    """3D 卷积块"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        
        # 深度可分离卷积 
        self.DepthwiseSeparableConv3d1 = Sequential(
            BasicConv3d(in_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False, groups=groups),
            # BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            GroupTemporalAtrousPyramid2(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        )
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,kernel), stride=(1,stride,stride), padding=(0,padding,padding), bias=False, groups=groups),
            # BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            GroupTemporalAtrousPyramid2(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False),
        )

        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积
        return self.DepthwiseSeparableConv3d2(self.DepthwiseSeparableConv3d1(x))
       

# ==========================================
# 主网络结构 UNet3D
# ==========================================

class UNet3DGroupATDCDilation4Branch(Module):
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
                return BasicConv3d(channels, channels, kernel_size=k, stride=s, padding=p, groups=1)
            
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
        self.enc_conv1 = Conv3D_Block_Group_ATDC_Dilation(num_channels, feat_channels[0], residual=residual, groups=groups)
        self.enc_conv2 = Conv3D_Block_Group_ATDC_Dilation(feat_channels[0], feat_channels[1], residual=residual, groups=groups)
        self.enc_conv3 = Conv3D_Block_Group_ATDC_Dilation(feat_channels[1], feat_channels[2], residual=residual, groups=groups)
        
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Conv3D_Block_Group_ATDC_Dilation(feat_channels[2], feat_channels[3], residual=residual, groups=groups) # 瓶颈层

        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual, groups=groups)
        
        self.dec_conv3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual, groups=groups)
        self.dec_conv2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual, groups=groups)
        self.dec_conv1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual, groups=groups)

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
            down1 = self.down1(enc1)                # Downsample

            enc2 = self.enc_conv2(down1)            # [B, C1, D/s, H/2, W/2]
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
    input_tensor = torch.randn(1, 3, 5, 512, 512).to(device)

    # 4种情况测试
    configs = [
        {"T_pooling": True, "mode": "stride"},
        {"T_pooling": True, "mode": "maxpool"},
        {"T_pooling": False, "mode": "stride"},
        {"T_pooling": False, "mode": "maxpool"},
    ]
    configs = [configs[3]]
    print(f"{'T_pool':<10} | {'Mode':<10} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'Check D'}")
    print("-" * 65)

    for cfg in configs:
        model = UNet3DGroupATDCDilation4Branch(
            num_channels=3,
            feat_channels=[16, 32, 64], # 轻量化通道用于测试
            upsample_mode="trilinear",
            T_pooling=cfg["T_pooling"],
            downsample_mode=cfg["mode"],
            groups=2
        ).to(device).eval()

        # 运行一次检查输出尺寸
        with torch.no_grad():
            output = model(input_tensor)
        
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