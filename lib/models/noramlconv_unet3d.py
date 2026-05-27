import warnings
import torch
from torch import nn
from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, Dropout3d, ReLU, Sigmoid


class UNet3D(Module):
    """
    3D UNet 实现（时间维度D不做池化/上采样）
    结构：4层下采样（仅H/W池化） → 瓶颈层 → 4层上采样（仅H/W上采样） + 跳跃连接
                                                            # 输出层32-->num_classes
    # __                            __  # 3-->32            # 64-->32, +32-->32
    #  1|__   ________________   __|1   # 32-->64           # 128-->64, +64-->64
    #     2|__  ____________  __|2      # 64-->128          # 256-->128, +128-->128
    #        3|__  ______  __|3         # 128-->256         # 512-->256, +256-->256
    #           4|__ __ __|4            # 256-->512         
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity
    """
    def __init__(self, num_channels=3, num_classes=1, feat_channels=[32,64,128,256], residual=None, 
                 upsample_mode="trilinear", dropout_prob=0, activation='sigmoid', T_pooling = True, groups=2):
        super().__init__()
        # assert len(feat_channels) == 5, "feat_channels 需包含5个元素: [c1, c2, c3, c4, c5]"
        # assert upsample_mode in ["deconv", "trilinear"], "upsample_mode 仅支持 deconv/trilinear"
        # assert residual in [None, "conv"], "residual 仅支持 None/conv"

        # 配置参数
        self.feat_channels = feat_channels
        self.upsample_mode = upsample_mode
        if T_pooling:
            # 编码器：下采样（T/H/W池化）
            self.pool1 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 关键：stride显式(1,2,2)
            self.pool2 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.pool3 = MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            # self.pool4 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            # 编码器：下采样（仅H/W池化，D维度kernel=1、stride=1，不池化）
            self.pool1 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 关键：stride显式(1,2,2)
            self.pool2 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.pool3 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            # self.pool4 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 编码器：卷积块（带残差）
        self.enc_conv1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual,groups=groups)
        self.enc_conv2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual,groups=groups)
        self.enc_conv3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual,groups=groups)
        if len(self.feat_channels)>=4:
            self.enc_conv4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual,groups=groups) # 瓶颈层
        # self.enc_conv5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # 解码器：卷积块（带残差）
        if len(self.feat_channels)>=4:
            self.dec_conv4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual,groups=groups)
        self.dec_conv3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual,groups=groups)
        self.dec_conv2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual,groups=groups)
        self.dec_conv1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual,groups=groups)

        # 解码器：上采样块（H/W上采样，D维度根据T_upsample判断是否上采样）
        # self.upsample4 = Upsample3D_Block(feat_channels[4], feat_channels[3], mode=upsample_mode)
        if len(self.feat_channels)>=4:
            self.upsample3 = Upsample3D_Block(feat_channels[3], feat_channels[2], mode=upsample_mode, T_upsample=T_pooling)
        self.upsample2 = Upsample3D_Block(feat_channels[2], feat_channels[1], mode=upsample_mode, T_upsample=T_pooling)
        self.upsample1 = Upsample3D_Block(feat_channels[1], feat_channels[0], mode=upsample_mode, T_upsample=T_pooling)

        # 最终分割头（1x1卷积 + Sigmoid）
        self.final_conv = Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = activation
        self.sigmoid = Sigmoid()

    def forward(self, x):
        if len(self.feat_channels)-1==3:
            """
            前向传播
            :param x: 输入张量，shape [B, C, D, H, W]
            :return: 分割结果，shape [B, C, D, H, W]
            下采样每个阶段包括先卷积增加通道维度，再池化降低空间尺寸
            上采样每个阶段包括先上采样增加空间尺寸同时降低通道维度，再拼接增加通道维度，再卷积减少通道维度
            """
            # 编码器：下采样（仅H/W池化，D维度不变） + 卷积
            enc1 = self.enc_conv1(x)            # [B, 16, D, H, W]
            pool1 = self.pool1(enc1)            # [B, 16, D, H/2, W/2]（D维度不变）

            enc2 = self.enc_conv2(pool1)        # [B, 32, D, H/2, W/2]
            # del pool1
            pool2 = self.pool2(enc2)            # [B, 32, D, H/4, W/4]

            enc3 = self.enc_conv3(pool2)        # [B, 64, D, H/4, W/4]
            # del pool2
            pool3 = self.pool3(enc3)            # [B, 64, D, H/8, W/8]

            # enc4 = self.enc_conv4(pool3)      # [B, 128, D, H/8, W/8]
            # pool4 = self.pool4(enc4)          # [B, 128, D, H/16, W/16]

            bottleneck = self.enc_conv4(pool3)# [B, 128, D, H/8, W/8] # 瓶颈层空间尺寸不变但是通道数翻倍
            # del pool3

            # 解码器：上采样（仅H/W上采样） + 尺寸对齐 + 拼接 + 卷积
            # upsample4 = self.upsample4(bottleneck)  # [B, 64, D, H/8, W/8]
            # # upsample4 = self._align_feature_size(upsample4, enc4)
            # concat4 = torch.cat([upsample4, enc4], dim=1)  # [B, 128, D, H/8, W/8]
            # dec4 = self.dec_conv4(concat4)               # [B, 64, D, H/8, W/8]

            tmp = self.upsample3(bottleneck)          # [B, 64, D, H/4, W/4] # 上采样的同时完成通道数减半
            # del bottleneck
            # upsample3 = self._align_feature_size(upsample3, enc3)
            tmp = torch.cat([tmp, enc3], dim=1)# [B, 128, D, H/4, W/4]
            # del enc3
            tmp = self.dec_conv3(tmp)               # [B, 64, D, H/4, W/4]

            tmp = self.upsample2(tmp)            # [B, 32, D, H/2, W/2]
            # upsample2 = self._align_feature_size(upsample2, enc2)
            tmp = torch.cat([tmp, enc2], dim=1)# [B, 64, D, H/2, W/2]
            del enc2
            tmp = self.dec_conv2(tmp)               # [B, 32, D, H/2, W/2]

            tmp = self.upsample1(tmp)            # [B, 16, D, H, W]
            # upsample1 = self._align_feature_size(upsample1, enc1)
            tmp = torch.cat([tmp, enc1], dim=1)# [B, 32, D, H, W]
            del enc1
            tmp = self.dec_conv1(tmp)               # [B, 16, D, H, W]

            # 最终分割
            tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            # device = x.device
            # print(f"net运行完毕\n\t",torch.cuda.memory_summary(device, abbreviated=True))

            return tmp
        elif len(self.feat_channels)-1 == 2:
            """
            前向传播
            :param x: 输入张量，shape [B, C, D, H, W]
            :return: 分割结果，shape [B, C, D, H, W]
            下采样每个阶段包括先卷积增加通道维度，再池化降低空间尺寸
            上采样每个阶段包括先上采样增加空间尺寸同时降低通道维度，再拼接增加通道维度，再卷积减少通道维度
            """
            # 编码器：下采样（仅H/W池化，D维度不变） + 卷积
            enc1 = self.enc_conv1(x)            # [B, 16, D, H, W]
            pool1 = self.pool1(enc1)            # [B, 16, D, H/2, W/2]（D维度不变）

            enc2 = self.enc_conv2(pool1)        # [B, 32, D, H/2, W/2]
            # del pool1
            pool2 = self.pool2(enc2)            # [B, 32, D, H/4, W/4]

            # enc3 = self.enc_conv3(pool2)        # [B, 64, D, H/4, W/4]
            # # del pool2
            # pool3 = self.pool3(enc3)            # [B, 64, D, H/8, W/8]

            # enc4 = self.enc_conv4(pool3)      # [B, 128, D, H/8, W/8]
            # pool4 = self.pool4(enc4)          # [B, 128, D, H/16, W/16]

            bottleneck = self.enc_conv3(pool2)# [B, 128, D, H/8, W/8] # 瓶颈层空间尺寸不变但是通道数翻倍
            # del pool3

            # 解码器：上采样（仅H/W上采样） + 尺寸对齐 + 拼接 + 卷积
            # upsample4 = self.upsample4(bottleneck)  # [B, 64, D, H/8, W/8]
            # # upsample4 = self._align_feature_size(upsample4, enc4)
            # concat4 = torch.cat([upsample4, enc4], dim=1)  # [B, 128, D, H/8, W/8]
            # dec4 = self.dec_conv4(concat4)               # [B, 64, D, H/8, W/8]

            # tmp = self.upsample3(bottleneck)          # [B, 64, D, H/4, W/4] # 上采样的同时完成通道数减半
            # # del bottleneck
            # # upsample3 = self._align_feature_size(upsample3, enc3)
            # tmp = torch.cat([tmp, enc3], dim=1)# [B, 128, D, H/4, W/4]
            # # del enc3
            # tmp = self.dec_conv3(tmp)               # [B, 64, D, H/4, W/4]

            tmp = self.upsample2(bottleneck)            # [B, 32, D, H/2, W/2]
            # upsample2 = self._align_feature_size(upsample2, enc2)
            tmp = torch.cat([tmp, enc2], dim=1)# [B, 64, D, H/2, W/2]
            del enc2
            tmp = self.dec_conv2(tmp)               # [B, 32, D, H/2, W/2]

            tmp = self.upsample1(tmp)            # [B, 16, D, H, W]
            # upsample1 = self._align_feature_size(upsample1, enc1)
            tmp = torch.cat([tmp, enc1], dim=1)# [B, 32, D, H, W]
            del enc1
            tmp = self.dec_conv1(tmp)               # [B, 16, D, H, W]

            # 最终分割
            tmp = self.final_conv(tmp)
            if self.activation == 'sigmoid':
                tmp = self.sigmoid(tmp)
            # device = x.device
            # print(f"net运行完毕\n\t",torch.cuda.memory_summary(device, abbreviated=True))

            return tmp
    # def forward(self, x):
    #     """
    #     前向传播
    #     :param x: 输入张量，shape [B, C, D, H, W]
    #     :return: 分割结果，shape [B, C, D, H, W]
    #     下采样每个阶段包括先卷积增加通道维度，再池化降低空间尺寸
    #     上采样每个阶段包括先上采样增加空间尺寸同时降低通道维度，再拼接增加通道维度，再卷积减少通道维度
    #     """
    #     # 编码器：下采样（仅H/W池化，D维度不变） + 卷积
    #     enc1 = self.enc_conv1(x)            # [B, 16, D, H, W]
    #     pool1 = self.pool1(enc1)            # [B, 16, D, H/2, W/2]（D维度不变）

    #     enc2 = self.enc_conv2(pool1)        # [B, 32, D, H/2, W/2]
    #     pool2 = self.pool2(enc2)            # [B, 32, D, H/4, W/4]

    #     enc3 = self.enc_conv3(pool2)        # [B, 64, D, H/4, W/4]
    #     pool3 = self.pool3(enc3)            # [B, 64, D, H/8, W/8]

    #     # enc4 = self.enc_conv4(pool3)      # [B, 128, D, H/8, W/8]
    #     # pool4 = self.pool4(enc4)          # [B, 128, D, H/16, W/16]

    #     bottleneck = self.enc_conv4(pool3)# [B, 128, D, H/8, W/8] # 瓶颈层空间尺寸不变但是通道数翻倍

    #     # 解码器：上采样（仅H/W上采样） + 尺寸对齐 + 拼接 + 卷积
    #     # upsample4 = self.upsample4(bottleneck)  # [B, 64, D, H/8, W/8]
    #     # # upsample4 = self._align_feature_size(upsample4, enc4)
    #     # concat4 = torch.cat([upsample4, enc4], dim=1)  # [B, 128, D, H/8, W/8]
    #     # dec4 = self.dec_conv4(concat4)               # [B, 64, D, H/8, W/8]

    #     upsample3 = self.upsample3(bottleneck)          # [B, 64, D, H/4, W/4] # 上采样的同时完成通道数减半
    #     # upsample3 = self._align_feature_size(upsample3, enc3)
    #     concat3 = torch.cat([upsample3, enc3], dim=1)# [B, 128, D, H/4, W/4]
    #     dec3 = self.dec_conv3(concat3)               # [B, 64, D, H/4, W/4]

    #     upsample2 = self.upsample2(dec3)            # [B, 32, D, H/2, W/2]
    #     # upsample2 = self._align_feature_size(upsample2, enc2)
    #     concat2 = torch.cat([upsample2, enc2], dim=1)# [B, 64, D, H/2, W/2]
    #     dec2 = self.dec_conv2(concat2)               # [B, 32, D, H/2, W/2]

    #     upsample1 = self.upsample1(dec2)            # [B, 16, D, H, W]
    #     # upsample1 = self._align_feature_size(upsample1, enc1)
    #     concat1 = torch.cat([upsample1, enc1], dim=1)# [B, 32, D, H, W]
    #     dec1 = self.dec_conv1(concat1)               # [B, 16, D, H, W]

    #     # 最终分割
    #     segs = self.final_conv(dec1)
    #     if self.activation == 'sigmoid':
    #         segs = self.sigmoid(segs)
   

    #     return segs
'''
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，shape [B, C, D, H, W]
        :return: 分割结果，shape [B, C, D, H, W]
        """
        # 编码器：下采样（仅H/W池化，D维度不变） + 卷积
        enc1 = self.enc_conv1(x)          # [B, 64, D, H, W]
        pool1 = self.pool1(enc1)          # [B, 64, D, H/2, W/2]（D维度不变）

        enc2 = self.enc_conv2(pool1)      # [B, 256, D, H/2, W/2]
        pool2 = self.pool2(enc2)          # [B, 256, D, H/4, W/4]

        enc3 = self.enc_conv3(pool2)      # [B, 256, D, H/4, W/4]
        pool3 = self.pool3(enc3)          # [B, 256, D, H/8, W/8]

        enc4 = self.enc_conv4(pool3)      # [B, 512, D, H/8, W/8]
        pool4 = self.pool4(enc4)          # [B, 512, D, H/16, W/16]

        bottleneck = self.enc_conv5(pool4)# [B, 1024, D, H/16, W/16]

        # 解码器：上采样（仅H/W上采样） + 尺寸对齐 + 拼接 + 卷积
        upsample4 = self.upsample4(bottleneck)  # [B, 512, D, H/8, W/8]
        # upsample4 = self._align_feature_size(upsample4, enc4)
        concat4 = torch.cat([upsample4, enc4], dim=1)  # [B, 1024, D, H/8, W/8]
        dec4 = self.dec_conv4(concat4)               # [B, 512, D, H/8, W/8]

        upsample3 = self.upsample3(dec4)             # [B, 256, D, H/4, W/4]
        # upsample3 = self._align_feature_size(upsample3, enc3)
        concat3 = torch.cat([upsample3, enc3], dim=1)# [B, 512, D, H/4, W/4]
        dec3 = self.dec_conv3(concat3)               # [B, 256, D, H/4, W/4]

        upsample2 = self.upsample2(dec3)             # [B, 256, D, H/2, W/2]
        # upsample2 = self._align_feature_size(upsample2, enc2)
        concat2 = torch.cat([upsample2, enc2], dim=1)# [B, 512, D, H/2, W/2]
        dec2 = self.dec_conv2(concat2)               # [B, 256, D, H/2, W/2]

        upsample1 = self.upsample1(dec2)             # [B, 64, D, H, W]
        # upsample1 = self._align_feature_size(upsample1, enc1)
        concat1 = torch.cat([upsample1, enc1], dim=1)# [B, 128, D, H, W]
        dec1 = self.dec_conv1(concat1)               # [B, 64, D, H, W]

        # 最终分割
        seg_logits = self.final_conv(dec1)
        seg = self.sigmoid(seg_logits)

        return seg'''

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,bias=False,groups=2):
        super(BasicConv3d, self).__init__()
                # ========== 核心修改：自动调整groups，替代断言报错 ==========
        # 1. 检查输入通道是否能被groups整除，不满足则重置为1
        if in_channels % groups != 0:
            warnings.warn(
                f"in_channels ({in_channels}) is not divisible by groups ({groups}), "
                f"automatically reset groups to 1 (normal convolution)."
            )
            groups = 1  # 重置为普通卷积
        
        # 2. （可选）检查输出通道是否能被groups整除，不满足则重置为1
        if out_channels % groups != 0:
            warnings.warn(
                f"out_channels ({out_channels}) is not divisible by groups ({groups}), "
                f"automatically reset groups to 1 (normal convolution)."
            )
            groups = 1  # 重置为普通卷积
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
    """3D 卷积块（双层卷积 + 可选残差连接）"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        # 第一层常规卷积
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        # 第二层常规卷积
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat),
            ReLU(inplace=True)
        )
        # 残差连接（1x1卷积匹配通道数）
        self.residual = residual
        if self.residual == "conv":
            self.residual_conv = Conv3d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)
        
        # 第一层深度可分离卷积
        self.DepthwiseSeparableConv3d1 = Sequential(
            BasicConv3d(in_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False,groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False,groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False,groups=groups),
        )
        # 第二层深度可分离卷积
        self.DepthwiseSeparableConv3d2 = Sequential(
            BasicConv3d(out_feat, out_feat, kernel_size=(1,1,kernel), stride=(1,1,stride), padding=(0,0,padding), bias=False,groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(1,kernel,1), stride=(1,stride,1), padding=(0,padding,0), bias=False,groups=groups),
            BasicConv3d(out_feat, out_feat, kernel_size=(kernel,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False,groups=groups),
        )

    def forward(self, x):
        if False:
            out = self.conv2(self.conv1(x))
            # out = self.conv1(x)
            # 残差连接
            if self.residual == "conv":
                out += self.residual_conv(x)
            return out
        else:
            out = self.DepthwiseSeparableConv3d2(self.DepthwiseSeparableConv3d1(x))
            # out = self.DepthwiseSeparableConv3d1(x)
            # # 残差连接
            # if self.residual == "conv":
            #     out += self.residual_conv(x)
            return out


class Upsample3D_Block(Module):
    """3D 上采样块（仅H/W上采样，D维度不处理）"""
    def __init__(self, in_feat, out_feat, kernel=3, stride=2, padding=1, mode="deconv",T_upsample=True):
        super().__init__()
        self.mode = mode
        self.out_feat = out_feat

        if mode == "deconv": # 有groups参数吗？
            if T_upsample:
                self.upsample = Sequential(
                ConvTranspose3d(
                    in_feat, out_feat,
                    kernel_size=(kernel, kernel, kernel),  # D维度kernel=1，不卷积
                    stride=(kernel, stride, stride),       # D维度stride=1，不上采样
                    padding=(padding, padding, padding),
                    output_padding=(1, 1, 1),         # 仅H/W补1，匹配stride=2
                    bias=True
                ),
                ReLU(inplace=True)
            )
            else:
                # 反卷积上采样：仅H/W上采样，D维度kernel=1、stride=1、output_padding=0
                self.upsample = Sequential(
                    ConvTranspose3d(
                        in_feat, out_feat,
                        kernel_size=(1, kernel, kernel),  # D维度kernel=1，不卷积
                        stride=(1, stride, stride),       # D维度stride=1，不上采样
                        padding=(padding, padding, padding),
                        output_padding=(0, 1, 1),         # 仅H/W补1，匹配stride=2
                        bias=True
                    ),
                    ReLU(inplace=True)
                )
        elif mode == "trilinear":
            if T_upsample:
                self.upsample = Sequential(
                    nn.Upsample(
                        scale_factor=(2, 2, 2),  # D维度scale=1，不缩放
                        mode="trilinear",
                        align_corners=True
                    ),
                    Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
                    ReLU(inplace=True)
                )
            else:
                # 三线性插值上采样：仅H/W缩放2倍，D维度缩放1倍
                self.upsample = Sequential(
                    nn.Upsample(
                        scale_factor=(1, 2, 2),  # D维度scale=1，不缩放
                        mode="trilinear",
                        align_corners=True
                    ),
                    Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
                    ReLU(inplace=True)
                )
        else:
            raise ValueError(f"不支持的上采样模式: {mode}，仅支持 deconv/trilinear或时间维度上采样错误！")

    def forward(self, x):
        return self.upsample(x)

if __name__ == '__main__':
    import time
    import torch
    try:
        from thop import profile, clever_format
    except ImportError:
        print("提示：未安装thop（pip install thop），跳过参数量/计算量计算")
        profile = None
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 测试两种上采样方式（验证时间维度不池化）
    for upsample_mode in ["trilinear", ]: # "deconv"
        print(f"\n=== 测试上采样模式: {upsample_mode} ===")
        # 初始化模型
        model = UNet3D(
            num_channels=3,
            feat_channels=[16,32,64], # 三次下采样一个瓶颈层 # [16,32,64,128]
            residual=None,
            upsample_mode=upsample_mode,
            dropout_prob=0,
            activation=None,# 'sigmoid',
            T_pooling=False,
            groups=2
        ).to(device).eval()
        # print("\n=== 网络结构概要 ===")
        # print(model)  # 打印完整模型结构
        # 测试输入：[B, C, D, H, W] = [1, 3, 5, 512, 512]（D=5为时间维度）
        test_input = torch.randn(1, 3, 5, 512, 512).to(device)  # 移除过时的Variable

        # 前向传播 & 推理耗时
        start_time = time.time()
        # with torch.no_grad():
        #     output = model(test_input)
        model.train() # 切换到训练模式以启用梯度计算检查显存占用
        output = model(test_input)
        infer_time = time.time() - start_time

        # 基础信息打印
        print(f"输入尺寸: {test_input.shape}")
        print(f"输出尺寸: {output.shape}")
        print(f"推理耗时: {infer_time:.4f}s")
        # assert output.shape[2] == test_input.shape[2], "时间维度（D）尺寸被错误修改！"

        # 参数量/计算量计算（thop）
        if profile is not None:
            flops, params = profile(model, inputs=(test_input,), verbose=False)
            # Params 通常以 M (Million) 为单位
            print(f"Total Parameters: {params / 1e6:.8f} M") 
            # FLOPs 通常以 G (Billion) 为单位
            print(f"Total FLOPs (MACs): {flops / 1e9:.8f} G")
        
        # 网络结构打印（简易版，替代torchsummaryX）
        
        # 可选：仅打印层结构统计
        # total_layers = sum(1 for _ in model.named_modules() if not isinstance(getattr(model, _.split('.')[0]), Module) or _.count('.') == 0)
        # print(f"模型总层数（顶层）: {total_layers}")