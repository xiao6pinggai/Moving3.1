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

class SpatialCDC3d(nn.Module):
    """
    3D环境下的空间中心差分卷积 (Spatial Central Difference Convolution)
    核心思想：利用重参数化技巧，修改卷积核权重来实现差分计算，无需自定义算子。
    卷积核尺寸强制为 (1, K, K)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=0.7):
        super().__init__()
        # 强制 kernel_size 第一个维度(T)为1，只做空间处理
        self.conv = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=(1, kernel_size, kernel_size), 
                              stride=stride, 
                              padding=(0, padding, padding), # T维度不padding，HW padding
                              bias=bias)
        self.theta = theta

    def forward(self, x):
        # 获取原始权重 (C_out, C_in, 1, H, W)
        weights = self.conv.weight
        
        # 计算空间维度 (dim=3,4) 的权重和
        # Shape: (C_out, C_in, 1, 1, 1)
        kernel_sum = weights.sum(dim=(3, 4), keepdim=True)
        
        # 构造 CDC 权重
        cdc_weights = weights.clone()
        
        # 找到空间中心索引
        center_idx_h = self.conv.kernel_size[1] // 2
        center_idx_w = self.conv.kernel_size[2] // 2
        
        # 【修改处】: 确保 kernel_sum 的形状与切片后的 cdc_weights (C_out, C_in) 完全一致
        # 原始 weights 是 (Out, In, 1, K, K)
        # kernel_sum 是 (Out, In, 1, 1, 1) -> view 为 (Out, In)
        cdc_weights[:, :, 0, center_idx_h, center_idx_w] -= kernel_sum.view(weights.shape[0], weights.shape[1])
        
        # 广义 CDC
        final_weights = self.theta * weights + (1 - self.theta) * cdc_weights
        
        return F.conv3d(x, final_weights, self.conv.bias, 
                        self.conv.stride, self.conv.padding)

class TemporalZeroSumConv3d(nn.Conv3d):
    """
    时域零和约束卷积 + 镜像 Padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias) # 内部padding设为0，手动pad
        # 计算 T 维度的 padding 量
        self.pad_t = kernel_size[0] // 2
        self.pad_h = kernel_size[1] // 2
        self.pad_w = kernel_size[2] // 2
        
    def forward(self, x):
        # 1. 权重零和约束 (针对 dim=2 即 Time 维度)
        # weight shape: (out, in, T, H, W)
        w_mean = self.weight.mean(dim=2, keepdim=True)
        zero_sum_w = self.weight - w_mean
        
        # 2. 镜像 Padding (Mirror Padding / Reflect)
        # F.pad 参数顺序: (W_left, W_right, H_top, H_bottom, T_front, T_back)
        # 空间维度使用常规 padding (通常是 zeros，这里假设保持原逻辑)，T维度使用 reflect
        # 注意：为了节省显存，尽量合并操作，但 reflect padding 必须单独做
        
        # 先处理 T 维度的镜像 padding
        if self.pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad_t, self.pad_t), mode='reflect')
            
        # 空间维度通常用 zeros padding (由 conv3d 自动处理或手动处理)
        # 这里为了配合 conv3d 接口，我们在 conv3d 调用时传入空间 padding
        
        return F.conv3d(x, zero_sum_w, self.bias, self.stride, 
                        (0, self.pad_h, self.pad_w), self.dilation, self.groups)

class MultiBranch_Conv3D_Block(nn.Module):
    """
    重构后的多分支 3D 卷积块
    Branch 1: 常规 3D
    Branch 2: 空间特征 (Vanilla 或 CDC)
    Branch 3: 时域特征 (Zero-Sum + Mirror Pad)
    """
    def __init__(self, in_feat, out_feat, kernel_dict=None, stride=1, padding=1, use_cdc=True, residual=None):
        super().__init__()
        
        # 默认卷积核配置
        if kernel_dict is None:
            kernel_dict = {
                'std': (3, 3, 3),      # 常规分支
                'spatial': 3,          # 空间分支 (1, k, k)
                'temporal': (3, 3, 3)  # 时域分支
            }
        
        self.use_cdc = use_cdc
        
        # === Part 1: 保持不变的 Conv1 ===
        # 这里的 kernel_size 取 std 配置，通常是 3x3x3
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_feat, out_feat, kernel_size=kernel_dict['std'], stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        # === Part 2: 多分支结构 (替代原 Conv2) ===
        
        # Branch 1: 常规 3D 卷积
        self.b1_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel_dict['std'], stride=1, padding=padding, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
        self.b1_proj = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

        # Branch 2: 空间特征 (每帧独立) -> (1, K, K)
        ks_s = kernel_dict['spatial']
        pad_s = ks_s // 2
        if self.use_cdc:
            self.b2_conv = SpatialCDC3d(out_feat, out_feat, kernel_size=ks_s, stride=1, padding=pad_s, bias=False)
        else:
            self.b2_conv = nn.Conv3d(out_feat, out_feat, kernel_size=(1, ks_s, ks_s), stride=1, padding=(0, pad_s, pad_s), bias=False)
        
        self.b2_bn_relu = nn.Sequential(nn.BatchNorm3d(out_feat), nn.ReLU(inplace=True))
        self.b2_proj = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

        # Branch 3: 时域零和约束
        self.b3_conv = TemporalZeroSumConv3d(out_feat, out_feat, kernel_size=kernel_dict['temporal'], stride=1, bias=False)
        self.b3_bn = nn.Sequential(nn.BatchNorm3d(out_feat)) # , nn.ReLU(inplace=True)
        self.b3_proj = nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

        # === Part 3: 融合后处理 ===
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True) # 最后一次激活
        )

        # 残差连接
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = nn.Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # 1. 原始残差保留 (如果是 conv 残差，针对的是输入 x)
        
        if self.residual == "conv":
            res = self.residual_conv(x)
        elif self.residual is not None:
            res = x
        # 2. 通过 Conv1 (特征提取)
        x = self.conv1(x)
        
        # 3. 多分支处理 (显存优化写法：累加到 buffer)
        
        # --- Branch 1: Standard ---
        # 这里的 buffer 直接作为融合结果的容器
        buffer = self.b1_conv(x) 
        buffer = self.b1_proj(buffer) 

        # --- Branch 2: Spatial (CDC or Normal) ---
        # 复用临时变量 tmp，计算完立即加到 buffer 并释放
        tmp = self.b2_conv(x)
        tmp = self.b2_bn_relu(tmp)
        tmp = self.b2_proj(tmp)
        buffer = buffer + tmp 
        
        # --- Branch 3: Temporal Zero-Sum ---
        tmp = self.b3_conv(x) # 内部包含 mirror padding
        tmp = self.b3_bn(tmp)
        tmp = self.b3_proj(tmp)
        buffer = buffer + tmp # 此时 buffer = b1 + b2 + b3
        
        # 释放 tmp
        del tmp
        
        # 4. 最终融合 (1x1 conv)
        out = self.fusion_conv(buffer)
        
        # 5. Add Residual (Shortcut)
        # 注意：这里需要确保维度对齐。如果 Conv1 改变了通道或分辨率，res 需要对应调整
        # 原代码逻辑是 enc 节点，通常分辨率不变(stride=1) 或 变小。
        # 如果 res 来自输入 x (in_feat)，而 out 是 (out_feat)，相加前需保证 residual_conv 执行了
        if self.residual is not None:
            out = out + res
        
        return out

class Normal_Conv3D_Block(Module):
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

        return self.conv2(self.conv1(x))

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

class UNet3DWithNormalConv3D3Branch(Module):
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
        # self.enc_conv1 = Normal_Conv3D_Block(num_channels, feat_channels[0], kernel=3, stride=1, padding=1)
        # self.enc_conv2 = Normal_Conv3D_Block(feat_channels[0], feat_channels[1], kernel=3, stride=1, padding=1)
        # self.enc_conv3 = Normal_Conv3D_Block(feat_channels[1], feat_channels[2], kernel=3, stride=1, padding=1)
        k_dic = {
            'std': (3, 3, 3),      # 常规分支
            'spatial': 3,          # 空间分支 (1, k, k)
            'temporal': (3, 3, 3)  # 时域分支
        }
        self.enc_conv1 = MultiBranch_Conv3D_Block(num_channels, feat_channels[0], kernel_dict=k_dic, stride=1, padding=1,use_cdc=True,residual=None)
        self.enc_conv2 = MultiBranch_Conv3D_Block(feat_channels[0], feat_channels[1], kernel_dict=k_dic, stride=1, padding=1,use_cdc=True,residual=None)
        self.enc_conv3 = MultiBranch_Conv3D_Block(feat_channels[1], feat_channels[2], kernel_dict=k_dic, stride=1, padding=1,use_cdc=True,residual=None)

        # self.atdc1 = MultiBranchATDCWithBN(feat_channels[0], feat_channels[0], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc2 = MultiBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc3 = MultiBranchATDCWithBN(feat_channels[2], feat_channels[2], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc = SingleBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, stride=1, padding=0, bias=False)
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
        model = UNet3DWithNormalConv3D3Branch(
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