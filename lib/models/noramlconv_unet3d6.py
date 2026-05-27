import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, ReLU, Sigmoid

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

class BasicATDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        super().__init__()
        # 自动调整 groups 以防止报错
        if in_channels % groups != 0:
            groups = 1
        if out_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias, groups=groups)


    def forward(self, x):
        # 硬零和约束
        W = self.conv.weight
        if W.shape[2]!=1:
            W = W - W.mean(dim=2, keepdim=True)

        # 卷积
        out = F.conv3d(x, W, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        # out = self.bn(out)

        # **将 gamma 放在卷积之后**
        # out = self.gamma * out  # <-- 可训练缩放

        # out = self.relu(out)
        return out
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
            BasicConv3d(in_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
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



class Conv3D_Block_ATDC(Module):
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
            BasicConv3d(in_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            BasicATDC(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            BasicATDC(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )

        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 切换逻辑：使用深度可分离卷积
        return self.DepthwiseSeparableConv3d2(self.DepthwiseSeparableConv3d1(x))
       


class GroupTemporalAtrousPyramid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
        super().__init__()
        
        self.k = kernel_size[0]
        # 定义三种扩张率
        self.d1 = 1
        self.d2 = 2
        self.d3 = 3
        
        # --- 1. 计算通道分配 (Split Channels) ---
        # 逻辑：将通道分为3组。如果有余数，优先分配给第一个分支（Dilation=1）
        
        # 输入通道分配
        in_div, in_rem = divmod(in_channels, 3)
        self.in_splits = [in_div + in_rem, in_div, in_div]
        
        # 输出通道分配
        out_div, out_rem = divmod(out_channels, 3)
        self.out_splits = [out_div + out_rem, out_div, out_div]
        
        # --- 2. 定义三个分组时域分支 ---
        # padding=0，因为我们将在 forward 中手动进行 reflect padding
        
        # 分支 1 (Dilation=1): 承担 base + remainder 的通道数
        self.branch1 = nn.Conv3d(self.in_splits[0], self.out_splits[0], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d1, 1, 1), bias=bias)
        
        # 分支 2 (Dilation=2): 承担 base 通道数
        self.branch2 = nn.Conv3d(self.in_splits[1], self.out_splits[1], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d2, 1, 1), bias=bias)
        
        # 分支 3 (Dilation=3): 承担 base 通道数
        self.branch3 = nn.Conv3d(self.in_splits[2], self.out_splits[2], 
                                 kernel_size=(self.k, 1, 1), 
                                 stride=1, padding=0, dilation=(self.d3, 1, 1), bias=bias)

        # 移除了 self.fusion 层，因为拼接后的输出即为 target out_channels

    def forward_zero_sum_conv(self, x, conv_layer, dilation_t):
        """
        保持原有的：Reflect Padding -> Zero-sum Constraint -> Conv -> Crop 逻辑
        """
        # --- 1. 手动对称镜像填充 (Reflect Padding) ---
        pad_size = dilation_t
        x_padded = F.pad(x, (0, 0, 0, 0, pad_size, pad_size), mode='reflect')
        
        # --- 2. 零和约束 ---
        w = conv_layer.weight
        # 减去时域维度的均值，确保 w[0] + w[1] + w[2] = 0
        w_zero_sum = w - w.mean(dim=2, keepdim=True)
        
        # --- 3. 卷积计算 ---
        out = F.conv3d(x_padded, w_zero_sum, 
                       bias=conv_layer.bias, 
                       stride=conv_layer.stride, 
                       padding=0, 
                       dilation=conv_layer.dilation, 
                       groups=conv_layer.groups)
        
        # # --- 4. 尺寸对齐 ---
        # if out.shape[2] != x.shape[2]:
        #     t_in = x.shape[2]
        #     t_out = out.shape[2]
        #     start = (t_out - t_in) // 2
        #     out = out[:, :, start : start + t_in, :, :]
            
        return out

    def forward(self, x):
        # x: (B, C_in, T, H, W)
        
        # --- 1. 通道切分 (Split) ---
        # 根据 __init__ 中计算的 split 列表将输入 tensor 在 dim=1 切开
        # split_size_or_sections 接受一个 list，指定每份的大小
        x_splits = torch.split(x, self.in_splits, dim=1)
        
        # --- 2. 分别处理 (Transform) ---
        # 每一组输入对应一个特定的 Dilation 分支
        
        # Group 1 -> Dilation 1 (可能包含余数通道)
        out1 = self.forward_zero_sum_conv(x_splits[0], self.branch1, dilation_t=self.d1)
        
        # Group 2 -> Dilation 2
        out2 = self.forward_zero_sum_conv(x_splits[1], self.branch2, dilation_t=self.d2)
        
        # Group 3 -> Dilation 3
        out3 = self.forward_zero_sum_conv(x_splits[2], self.branch3, dilation_t=self.d3)
        
        # --- 3. 拼接 (Merge) ---
        # 此时 out1 channels + out2 channels + out3 channels = total out_channels
        out = torch.cat([out1, out2, out3], dim=1)
        
        return out

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
            BasicConv3d(in_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            GroupTemporalAtrousPyramid(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
        )
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False, groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False, groups=groups),
            GroupTemporalAtrousPyramid(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False, groups=groups),
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

class UNet3DGroupATDCDilation(Module):
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
        model = UNet3DGroupATDCDilation(
            num_channels=3,
            feat_channels=[15, 30, 60, 120], # 轻量化通道用于测试
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