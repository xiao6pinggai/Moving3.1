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
    
    结构特点:
    1. 双路并行 (不拆通道): 
       - 支路 A (Zero-Sum): 强制时域零和，提取纯运动 (配合 Abs 激活)。
       - 支路 B (Standard): 常规 3D 卷积，保留外观 (配合 ReLU 激活)。
    2. 软门控机制 (Sigmoid p):
       - p 初始化为 -4.9，Sigmoid(p) ≈ 0.007。
       - 初始状态下几乎只使用常规卷积分支，训练中自动提升零和分支的权重。
    3. 后置特征提取:
       - 融合后的特征再经过一层常规卷积进行整合。
    """
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        # 1. 定义可学习参数 p
        # 初始化为 -4.9，使得 sigmoid(p) 接近 0 (约 0.007)
        # 含义: 初始时刻，ATDC(零和)支路的权重极低，保证冷启动稳定
        self.p = nn.Parameter(torch.tensor([-2.1]))

        # 2. 定义两个并行支路
        # 注意：这里两个支路都不拆分通道，都是 in -> out (或 hidden)
        # 为了控制参数量，通常这两个支路的输出通道可以设为 out_feat
        
        # [支路 A]: 零和卷积支路 (Zero-Sum Branch)
        # 这里的 Conv3d 只是容器，forward 里会对其权重做去均值处理
        self.conv_zs = nn.Conv3d(in_feat, out_feat, kernel, stride, (3,1,1), bias=bias,dilation=(3,1,1))
        self.bn_zs = nn.BatchNorm3d(out_feat)
        
        # [支路 B]: 常规卷积支路 (Standard Branch)
        self.conv_std = nn.Conv3d(in_feat, out_feat, kernel, stride, padding, bias=bias)
        self.bn_std = nn.BatchNorm3d(out_feat)
        self.relu = nn.ReLU(inplace=True)

        # 3. 后置特征提取层 (Post Fusion Conv)
        # 将融合后的结果进一步卷积，整合特征
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
        self._init_weights()
    def _init_weights(self):
            # 1. 对常规则卷积支路 (Standard Branch)：使用标准的 Kaiming 初始化
            nn.init.kaiming_normal_(self.conv_std.weight, mode='fan_out', nonlinearity='relu')
            
            # 2. 对零和卷积支路 (Zero-Sum Branch)：使用 "中心脉冲" 初始化
            # 这种初始化配合 w - w.mean() 会自动变成 [-1/3, 2/3, -1/3] 的差分算子
            
            # 先全置为 0
            nn.init.constant_(self.conv_zs.weight, 0)
            
            # 获取卷积核中心索引
            # weight shape: (Out, In, T, H, W)
            k_t, k_h, k_w = self.conv_zs.kernel_size
            c_t, c_h, c_w = k_t // 2, k_h // 2, k_w // 2
            
            # 将时空中心的权重置为 1 (类似于 nn.init.dirac_)
            with torch.no_grad():
                # 这里我们让空间上也是中心脉冲，保证初始时不搞乱空间特征，只关注时间差分
                self.conv_zs.weight[:, :, c_t, c_h, c_w] = 1.0 

                # 如果你想增加随机性，防止所有通道都一样，可以加一点极小的噪声
                # self.conv_zs.weight += torch.randn_like(self.conv_zs.weight) * 0.01

            # 3. 对后置卷积初始化
            nn.init.kaiming_normal_(self.post_conv[0].weight, mode='fan_out', nonlinearity='relu')

        # 别忘了在 __init__ 最后调用 self._init_weights()
    def forward(self, x):
        # -----------------------------------------------
        # Step 1: 计算调和系数 alpha
        # -----------------------------------------------
        # 技巧: 我们不直接用 self.p，而是由 self.p 产生一个临时变量 p_in
        # 这个 p_in 数值上等于 self.p，但我们给它注册一个 hook
        
        p_in = self.p * 1.0  # 乘 1.0 是为了创建一个新的计算图节点，不改变数值
        
        # 如果是训练阶段，注册一个 hook，把回传的梯度放大 10 倍 (或 100 倍)
        if self.training:
            # lambda grad: grad * 10.0 表示: 收到梯度 grad，传给 self.p 时变成 grad * 10
            p_in.register_hook(lambda grad: grad * 10.0)
            
        alpha = torch.sigmoid(p_in)
        # alpha 控制零和支路的权重，(1-alpha) 控制常规支路
        # alpha = torch.sigmoid(self.p)

        # -----------------------------------------------
        # Step 2: 支路 A (零和卷积 - Zero Sum)
        # -----------------------------------------------
        w_zs = self.conv_zs.weight
        # 核心约束：仅在时域维度 (dim=2) 去均值
        # kernel shape: (Out, In, T, H, W) -> index (0, 1, 2, 3, 4)
        w_zs_centered = w_zs - w_zs.mean(dim=2, keepdim=True)
        
        # 使用去均值后的权重卷积
        out_zs = F.conv3d(x, w_zs_centered, self.conv_zs.bias, 
                          self.conv_zs.stride, self.conv_zs.padding, 
                          self.conv_zs.dilation, self.conv_zs.groups)
        out_zs = self.bn_zs(out_zs)
        # 关键：零和卷积输出有正有负（运动变化），必须用 Abs 捕获能量
        out_zs = torch.abs(out_zs) 
        
        # -----------------------------------------------
        # Step 3: 支路 B (常规卷积 - Standard)
        # -----------------------------------------------
        out_std = self.conv_std(x)
        out_std = self.bn_std(out_std)
        out_std = self.relu(out_std) # 常规特征用 ReLU

        # -----------------------------------------------
        # Step 4: 加权融合 (Weighted Fusion)
        # -----------------------------------------------
        # 初始状态下 alpha ≈ 0，相当于 out ≈ out_std
        # 随着 p 增大，网络开始引入 out_zs (运动特征)
        merged = alpha * out_zs + (1.0 - alpha) * out_std

        # -----------------------------------------------
        # Step 5: 后置特征提取
        # -----------------------------------------------
        out = self.post_conv(merged)
        
        return out
class ATDC_Block(nn.Module):
   
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        

        # 2. 定义两个并行支路
        # 注意：这里两个支路都不拆分通道，都是 in -> out (或 hidden)
        # 为了控制参数量，通常这两个支路的输出通道可以设为 out_feat
        
        # [支路 A]: 零和卷积支路 (Zero-Sum Branch)
        # 这里的 Conv3d 只是容器，forward 里会对其权重做去均值处理
        self.conv_zs = nn.Conv3d(in_feat, out_feat, kernel, stride, (3,1,1), bias=bias,dilation=(3,1,1))
        self.bn_zs = nn.BatchNorm3d(out_feat)
        
        # [支路 B]: 常规卷积支路 (Standard Branch)
        self.conv_std = nn.Conv3d(in_feat, out_feat, kernel, stride, padding, bias=bias)
        self.bn_std = nn.BatchNorm3d(out_feat)
        self.relu = nn.ReLU(inplace=True)

        # 3. 后置特征提取层 (Post Fusion Conv)
        # 将融合后的结果进一步卷积，整合特征
        self.post_conv = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=1, padding=padding, bias=bias),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )
        self._init_weights()
    def _init_weights(self):
            # 1. 对常规则卷积支路 (Standard Branch)：使用标准的 Kaiming 初始化
            nn.init.kaiming_normal_(self.conv_std.weight, mode='fan_out', nonlinearity='relu')
            
            # 2. 对零和卷积支路 (Zero-Sum Branch)：使用 "中心脉冲" 初始化
            # 这种初始化配合 w - w.mean() 会自动变成 [-1/3, 2/3, -1/3] 的差分算子
            
            # 先全置为 0
            nn.init.constant_(self.conv_zs.weight, 0)
            
            # 获取卷积核中心索引
            # weight shape: (Out, In, T, H, W)
            k_t, k_h, k_w = self.conv_zs.kernel_size
            c_t, c_h, c_w = k_t // 2, k_h // 2, k_w // 2
            
            # 将时空中心的权重置为 1 (类似于 nn.init.dirac_)
            with torch.no_grad():
                # 这里我们让空间上也是中心脉冲，保证初始时不搞乱空间特征，只关注时间差分
                self.conv_zs.weight[:, :, c_t, c_h, c_w] = 1.0 

                # 如果你想增加随机性，防止所有通道都一样，可以加一点极小的噪声
                # self.conv_zs.weight += torch.randn_like(self.conv_zs.weight) * 0.01

            # 3. 对后置卷积初始化
            nn.init.kaiming_normal_(self.post_conv[0].weight, mode='fan_out', nonlinearity='relu')

        # 别忘了在 __init__ 最后调用 self._init_weights()
    def forward(self, x):
        # -----------------------------------------------
        # Step 1: 计算调和系数 alpha
        # -----------------------------------------------
        # 技巧: 我们不直接用 self.p，而是由 self.p 产生一个临时变量 p_in
        # 这个 p_in 数值上等于 self.p，但我们给它注册一个 hook
        
        p_in = self.p * 1.0  # 乘 1.0 是为了创建一个新的计算图节点，不改变数值
        
        # 如果是训练阶段，注册一个 hook，把回传的梯度放大 10 倍 (或 100 倍)
        if self.training:
            # lambda grad: grad * 10.0 表示: 收到梯度 grad，传给 self.p 时变成 grad * 10
            p_in.register_hook(lambda grad: grad * 10.0)
            
        alpha = torch.sigmoid(p_in)
        # alpha 控制零和支路的权重，(1-alpha) 控制常规支路
        # alpha = torch.sigmoid(self.p)

        # -----------------------------------------------
        # Step 2: 支路 A (零和卷积 - Zero Sum)
        # -----------------------------------------------
        w_zs = self.conv_zs.weight
        # 核心约束：仅在时域维度 (dim=2) 去均值
        # kernel shape: (Out, In, T, H, W) -> index (0, 1, 2, 3, 4)
        w_zs_centered = w_zs - w_zs.mean(dim=2, keepdim=True)
        
        # 使用去均值后的权重卷积
        out_zs = F.conv3d(x, w_zs_centered, self.conv_zs.bias, 
                          self.conv_zs.stride, self.conv_zs.padding, 
                          self.conv_zs.dilation, self.conv_zs.groups)
        out_zs = self.bn_zs(out_zs)
        # 关键：零和卷积输出有正有负（运动变化），必须用 Abs 捕获能量
        out_zs = torch.abs(out_zs) 
        
        # -----------------------------------------------
        # Step 3: 支路 B (常规卷积 - Standard)
        # -----------------------------------------------
        out_std = self.conv_std(x)
        out_std = self.bn_std(out_std)
        out_std = self.relu(out_std) # 常规特征用 ReLU

        # -----------------------------------------------
        # Step 4: 加权融合 (Weighted Fusion)
        # -----------------------------------------------
        # 初始状态下 alpha ≈ 0，相当于 out ≈ out_std
        # 随着 p 增大，网络开始引入 out_zs (运动特征)
        merged = alpha * out_zs + (1.0 - alpha) * out_std

        # -----------------------------------------------
        # Step 5: 后置特征提取
        # -----------------------------------------------
        out = self.post_conv(merged)
        
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

class UNet3DWithNormalConv3DPATDC(Module):
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
        self.enc_conv2 = ATDC_Parallel_Fusion_Block(feat_channels[0], feat_channels[1], kernel=(3,3,3), stride=1, padding=(1,1,1))
        self.enc_conv3 = Normal_Conv3D_Block(feat_channels[1], feat_channels[2], kernel=(3,3,3), stride=1, padding=(1,1,1))
        # self.atdc1 = MultiBranchATDCWithBN(feat_channels[0], feat_channels[0], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc2 = MultiBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc3 = MultiBranchATDCWithBN(feat_channels[2], feat_channels[2], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # self.atdc = SingleBranchATDCWithBN(feat_channels[1], feat_channels[1], kernel_size=3, stride=1, padding=0, bias=False)
        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(feat_channels[2], feat_channels[3], kernel=(5,3,3), stride=1, padding=(2,1,1)) # 瓶颈层
            # self.atdc4 = MultiBranchATDCWithBN(feat_channels[3], feat_channels[3], kernel_size=3, dilations=[3, 5], stride=1, padding=0, bias=False)
        # 解码器：卷积块
        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(2 * feat_channels[3], feat_channels[3], kernel=(5,3,3), stride=1, padding=(2,1,1))
        
        self.dec_conv3 = Normal_Conv3D_Block(2 * feat_channels[2], feat_channels[2], kernel=(5,3,3), stride=1, padding=(2,1,1))
        self.dec_conv2 = Normal_Conv3D_Block(2 * feat_channels[1], feat_channels[1], kernel=(5,3,3), stride=1, padding=(2,1,1))
        self.dec_conv1 = Normal_Conv3D_Block(2 * feat_channels[0], feat_channels[0], kernel=(5,3,3), stride=1, padding=(2,1,1))

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
        model = UNet3DWithNormalConv3DPATDC(
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