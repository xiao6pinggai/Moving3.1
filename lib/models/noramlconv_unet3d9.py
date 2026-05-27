import math
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
        # self.bn = nn.BatchNorm3d(out_channels,
        #                          eps=0.001, # value found in tensorflow
        #                          momentum=0.1, # default pytorch value
        #                          affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # 或者
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # 可训练缩放系数
        # self.relu = nn.PReLU()
        # self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1, 1)) # 通道级，即卷积核级

        # self._init_diff_kernel()

    # def _init_diff_kernel(self):
    #     with torch.no_grad():
    #         W = self.conv.weight
    #         C_out, C_in, Kt, Kh, Kw = W.shape
    #         W.zero_()
    #         if Kt == 1:
    #             W[:, :, 0, :, :] = 10.0
    #         else:
    #             if Kt==3: #
    #                 W[:, :, 0, :, :] += 10.0
    #                 W[:, :, 1, :, :] -= -5
    #                 W[:, :, 2, :, :] -= -5
    #             elif Kt==5: #
    #                 W[:, :, 0, :, :] += 10.0
    #                 W[:, :, 1, :, :] -= -2.5
    #                 W[:, :, 2, :, :] -= -2.5
    #                 W[:, :, 3, :, :] -= -2.5
    #                 W[:, :, 4, :, :] -= -2.5
    #         # W += 0.01 * torch.randn_like(W)

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

class BasicATDCWithGN(nn.Module):# 时域卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        num_groups = 1
        # # 如果通道数太少（小于32），则将组数设为通道数的一半，或者直接设为1 (LayerNorm)
        # if out_channels < 32:
        #     num_groups = max(1, out_channels // 2)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        

    def forward(self, x):
        # 硬零和约束
        W = self.conv.weight
        if W.shape[2]!=1:
            W = W - W.mean(dim=2, keepdim=True)

        # 卷积
        out = F.conv3d(x, W, stride=self.conv.stride, padding=self.conv.padding)
        # out = self.bn(out)
        out = self.gn(out)
        return out
class MultiBranchATDCWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations=[2, 4], stride=1, padding=1, bias=False):
        """
        Args:
            dilations (list): 空洞率列表，长度决定分支数量。例如 [1, 2, 4] 表示3个分支。
            padding (int): 空间维度(H, W)的padding，时序维度会自动计算镜像padding。
        """
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]  # 仅考虑时序维度的kernel_size

        self.num_branches = len(dilations)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        
        # --- 1. 计算各分支通道数 (divmod逻辑) ---
        # 输入通道分配
        base_in, remain_in = divmod(in_channels, self.num_branches)
        self.in_splits = [base_in + remain_in] + [base_in] * (self.num_branches - 1)
        
        # 输出通道分配 (同样逻辑，确保拼接后等于 out_channels)
        base_out, remain_out = divmod(out_channels, self.num_branches)
        self.out_splits = [base_out + remain_out] + [base_out] * (self.num_branches - 1)
        
        # --- 2. 构建卷积分支 ---
        self.convs = nn.ModuleList()
        self.time_pads = [] # 记录每个分支需要的时序填充量
        
        for idx, d in enumerate(dilations):
            # 计算当前分支需要的时序 padding 数量
            # Pad = dilation * (kernel_size - 1) // 2
            # t_pad = d * (kernel_size[0] - 1) // 2 if isinstance(kernel_size, tuple) else d * (kernel_size - 1) // 2
            t_pad = d * (kernel_size - 1) // 2
            self.time_pads.append(t_pad)
            
            # 空间 padding 保持由参数传入 (通常是 spatial padding)
            # 注意：这里 Conv3d 的 padding 设为 (0, padding, padding)
            # 因为时序维度我们将在 forward 里手动做镜像填充
            spatial_padding = padding 
            
            self.convs.append(
                nn.Conv3d(self.in_splits[idx], self.out_splits[idx], 
                          kernel_size=(kernel_size,1,1), stride=stride, 
                          padding=(0, spatial_padding, spatial_padding), # Time维填0，Space维正常填
                          dilation=(d, 1, 1), bias=bias)
            )

        # --- 3. 统一的 BN ---
        # 建议：如果显存允许，依然推荐改为 nn.GroupNorm
        self.bn = nn.BatchNorm3d(out_channels, affine=True)

    def get_standardized_weight(self, conv_layer):
        """
        权重标准化：Zero-Mean + Unit-Variance
        """
        W = conv_layer.weight
        
        # W shape: [out, in, D, H, W], 在时域维度 dim=2 做统计
        mean = W.mean(dim=2, keepdim=True)
        std = W.std(dim=2, keepdim=True) + 1e-5
        
        return (W - mean) / std

    def forward(self, x):
        # 1. 通道切分 (Split)
        x_splits = torch.split(x, self.in_splits, dim=1)
        
        branch_outputs = []
        
        for i, conv in enumerate(self.convs):
            x_i = x_splits[i]
            pad_t = self.time_pads[i]
            
            # 2. 镜像填充 (Mirror Padding / Replicate Padding)
            # F.pad 参数顺序: (Left, Right, Top, Bottom, Front, Back)
            # 我们只需要填充时序维度 (Front, Back) -> (pad_t, pad_t)
            if pad_t > 0:
                # 使用 'replicate' 模式进行边缘复制，避免补0带来的虚假边缘响应
                x_i = F.pad(x_i, (0, 0, 0, 0, pad_t, pad_t), mode='replicate')
            
            # 3. 获取标准化权重
            W_std = self.get_standardized_weight(conv)
            
            # 4. 执行卷积
            # 由于已经手动 pad 过了，这里卷积的时序 padding 实际上是 0 (在init里已设置)
            out_i = F.conv3d(x_i, W_std, conv.bias, conv.stride, 
                             conv.padding, conv.dilation, conv.groups)
            
            branch_outputs.append(out_i)

        # 5. 拼接 (Concat)
        out = torch.cat(branch_outputs, dim=1)

        # 6. BN
        out = self.bn(out)
        
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


# ==========================================
# 主网络结构 UNet3D
# ==========================================

class UNet3DAddATDC(Module):
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
        self.enc_conv1 = Conv3D_Block(num_channels, feat_channels[0], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        self.enc_conv2 = Conv3D_Block(feat_channels[0], feat_channels[1], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        self.enc_conv3 = Conv3D_Block(feat_channels[1], feat_channels[2], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        self.atdc1 = MultiBranchATDCWithBN(feat_channels[0], feat_channels[0], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        self.atdc2 = MultiBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        self.atdc3 = MultiBranchATDCWithBN(feat_channels[2], feat_channels[2], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Conv3D_Block(feat_channels[2], feat_channels[3], kernel=3, stride=1, padding=1,residual=residual, groups=groups) # 瓶颈层
            self.atdc4 = MultiBranchATDCWithBN(feat_channels[3], feat_channels[3], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        
        self.dec_conv3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        self.dec_conv2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], kernel=3, stride=1, padding=1,residual=residual, groups=groups)
        self.dec_conv1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], kernel=3, stride=1, padding=1,residual=residual, groups=groups)

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
            enc1 = enc1 + self.atdc1(enc1)
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
        
        # 深度可分离卷积 (保留结构，forward中未启用)
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
    configs = [configs[2]]
    print(f"{'T_pool':<10} | {'Mode':<10} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'Check D'}")
    print("-" * 65)

    for cfg in configs:
        model = UNet3DAddATDC(
            num_channels=3,
            feat_channels=[16, 32, 64, 128], # 轻量化通道用于测试
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