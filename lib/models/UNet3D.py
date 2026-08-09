from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn import (
    Module,
    Sequential,
    Conv3d,
    ConvTranspose3d,
    BatchNorm3d,
    MaxPool3d,
    ReLU,
)

from TVDCN.dcn_nd import DeformConv3d, SnakeDeformConv3d
from lib.models.TZSConv import TZSConvBlock


class Normal_Conv3D_Block(Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, groups=1):
        super().__init__()
        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )
        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            BatchNorm3d(out_feat, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class TemporalSnakeConv3D_Block(Module):
    def __init__(self, channels, kernel=3, extend_scope=10.0):
        super().__init__()
        self.block = Sequential(
            SnakeDeformConv3d(
                channels,
                channels,
                kernel_size=(kernel, 1, 1),
                padding=(kernel // 2, 0, 0),
                bias=False,
                extend_scope=extend_scope,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalConv3D_Block(Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.block = Sequential(
            Conv3d(
                channels,
                channels,
                kernel_size=(kernel, 1, 1),
                padding=(kernel // 2, 0, 0),
                bias=False,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalDCN3D_Block(Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.block = Sequential(
            DeformConv3d(
                channels,
                channels,
                kernel_size=(kernel, 1, 1),
                padding=(kernel // 2, 0, 0),
                bias=False,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Upsample3D_Block(Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        kernel=3,
        stride=2,
        padding=1,
        mode="trilinear",
        T_upsample=True,
    ):
        super().__init__()

        if mode == "deconv":
            scale_D = 1 if T_upsample else 0
            stride_D = stride if T_upsample else 1
            kernel_D = kernel if T_upsample else 1
            pad_D = padding if T_upsample else 0

            self.upsample = Sequential(
                ConvTranspose3d(
                    in_feat,
                    out_feat,
                    kernel_size=(kernel_D, kernel, kernel),
                    stride=(stride_D, stride, stride),
                    padding=(pad_D, padding, padding),
                    output_padding=(scale_D, 1, 1),
                    bias=True,
                ),
                ReLU(inplace=True),
            )
        elif mode == "trilinear":
            scale = (2, 2, 2) if T_upsample else (1, 2, 2)
            self.upsample = Sequential(
                nn.Upsample(scale_factor=scale, mode="trilinear", align_corners=True),
                Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=True),
                ReLU(inplace=True),
            )
        else:
            raise ValueError("Unsupported upsample mode: {}".format(mode))

    def forward(self, x):
        return self.upsample(x)


class UNet3DWithNormalConv3D(Module):
    """3D UNet backbone with normal convolutions."""

    def __init__(
        self,
        num_channels=3,
        num_classes=64,
        feat_channels=None,
        upsample_mode="trilinear",
        T_pooling=True,
        downsample_mode="stride",
        Snack_skip=None,
        Snack_max_offset=None,
        UNet3D_skip="snack",
        TZSConv_skip=None,
    ):
        super().__init__()
        if feat_channels is None:
            feat_channels = [16, 32, 64, 128]
        if Snack_skip is None:
            Snack_skip = [1, 1, 1]
        if Snack_max_offset is None:
            Snack_max_offset = [10, 10, 10]
        if TZSConv_skip is None:
            TZSConv_skip = [1, 1, 1]
        if len(Snack_skip) != 3:
            raise ValueError("Snack_skip must contain three switches for [skip1, skip2, skip3]")
        if len(Snack_max_offset) != 3:
            raise ValueError("Snack_max_offset must contain three values for [skip1, skip2, skip3]")
        if len(TZSConv_skip) != 3:
            raise ValueError("TZSConv_skip must contain three switches for [skip1, skip2, skip3]")
        self.Snack_skip = [int(v) for v in Snack_skip]
        if any(v not in (0, 1) for v in self.Snack_skip):
            raise ValueError("Snack_skip values must be 0 or 1")
        self.TZSConv_skip = [int(v) for v in TZSConv_skip]
        if any(v not in (0, 1) for v in self.TZSConv_skip):
            raise ValueError("TZSConv_skip values must be 0 or 1")
        self.Snack_max_offset = [float(v) for v in Snack_max_offset]
        if UNet3D_skip not in ("snack", "tconv", "tdcn"):
            raise ValueError("UNet3D_skip must be 'snack', 'tconv', or 'tdcn'")
        self.UNet3D_skip = UNet3D_skip

        self.feat_channels = feat_channels

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

            raise ValueError("Unknown downsample_mode: {}".format(downsample_mode))

        self.down1 = build_downsample_layer(feat_channels[0])
        self.down2 = build_downsample_layer(feat_channels[1])
        self.down3 = build_downsample_layer(feat_channels[2])

        self.enc_conv1 = Normal_Conv3D_Block(
            num_channels, feat_channels[0], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.enc_conv2 = Normal_Conv3D_Block(
            feat_channels[0], feat_channels[1], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.enc_conv3 = Normal_Conv3D_Block(
            feat_channels[1], feat_channels[2], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )

        if len(self.feat_channels) >= 4:
            self.enc_conv4 = Normal_Conv3D_Block(
                feat_channels[2], feat_channels[3], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
            )

        if len(self.feat_channels) >= 4:
            self.dec_conv4 = Normal_Conv3D_Block(
                2 * feat_channels[3], feat_channels[3], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
            )

        self.dec_conv3 = Normal_Conv3D_Block(
            2 * feat_channels[2], feat_channels[2], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.dec_conv2 = Normal_Conv3D_Block(
            2 * feat_channels[1], feat_channels[1], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )
        self.dec_conv1 = Normal_Conv3D_Block(
            2 * feat_channels[0], feat_channels[0], kernel=(3, 3, 3), stride=1, padding=(1, 1, 1)
        )

        if self.TZSConv_skip[0]:
            self.tzs_skip1 = TZSConvBlock(
                feat_channels[0],
                feat_channels[0],
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                bias=False,
                skip_add=True,
            )
        if self.TZSConv_skip[1]:
            self.tzs_skip2 = TZSConvBlock(
                feat_channels[1],
                feat_channels[1],
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                bias=False,
                skip_add=True,
            )
        if self.TZSConv_skip[2] and len(self.feat_channels) >= 4:
            self.tzs_skip3 = TZSConvBlock(
                feat_channels[2],
                feat_channels[2],
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                bias=False,
                skip_add=True,
            )

        if self.Snack_skip[0]:
            if self.UNet3D_skip == "snack":
                self.skip1 = TemporalSnakeConv3D_Block(feat_channels[0], kernel=5, extend_scope=self.Snack_max_offset[0])
            if self.UNet3D_skip == "tconv":
                self.skip1 = TemporalConv3D_Block(feat_channels[0], kernel=5)
            if self.UNet3D_skip == "tdcn":
                self.skip1 = TemporalDCN3D_Block(feat_channels[0], kernel=5)
        if self.Snack_skip[1]:
            if self.UNet3D_skip == "snack":
                self.skip2 = TemporalSnakeConv3D_Block(feat_channels[1], kernel=5, extend_scope=self.Snack_max_offset[1])
            if self.UNet3D_skip == "tconv":
                self.skip2 = TemporalConv3D_Block(feat_channels[1], kernel=5)
            if self.UNet3D_skip == "tdcn":
                self.skip2 = TemporalDCN3D_Block(feat_channels[1], kernel=5)
        if self.Snack_skip[2] and len(self.feat_channels) >= 4:
            if self.UNet3D_skip == "snack":
                self.skip3 = TemporalSnakeConv3D_Block(feat_channels[2], kernel=5, extend_scope=self.Snack_max_offset[2])
            if self.UNet3D_skip == "tconv":
                self.skip3 = TemporalConv3D_Block(feat_channels[2], kernel=5)
            if self.UNet3D_skip == "tdcn":
                self.skip3 = TemporalDCN3D_Block(feat_channels[2], kernel=5)

        if len(self.feat_channels) >= 4:
            self.upsample3 = Upsample3D_Block(
                feat_channels[3], feat_channels[2], mode=upsample_mode, T_upsample=T_pooling
            )

        self.upsample2 = Upsample3D_Block(
            feat_channels[2], feat_channels[1], mode=upsample_mode, T_upsample=T_pooling
        )
        self.upsample1 = Upsample3D_Block(
            feat_channels[1], feat_channels[0], mode=upsample_mode, T_upsample=T_pooling
        )

        # self.final_conv = Conv3d(feat_channels[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.final_conv.bias.data.fill_(bias_value)
        # self.final_conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if len(self.feat_channels) - 1 == 3:
            enc1 = self.enc_conv1(x)
            down1 = self.down1(enc1)

            enc2 = self.enc_conv2(down1)
            down2 = self.down2(enc2)

            enc3 = self.enc_conv3(down2)
            down3 = self.down3(enc3)

            bottleneck = self.enc_conv4(down3)

            tmp = self.upsample3(bottleneck)
            if self.TZSConv_skip[2]:
                enc3 = self.tzs_skip3(enc3)
            if self.Snack_skip[2]:
                enc3 = self.skip3(enc3)
            tmp = torch.cat([tmp, enc3], dim=1)
            tmp = self.dec_conv3(tmp)

            tmp = self.upsample2(tmp)
            if self.TZSConv_skip[1]:
                enc2 = self.tzs_skip2(enc2)
            if self.Snack_skip[1]:
                enc2 = self.skip2(enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            if self.TZSConv_skip[0]:
                enc1 = self.tzs_skip1(enc1)
            if self.Snack_skip[0]:
                enc1 = self.skip1(enc1)
            tmp = torch.cat([tmp, enc1], dim=1)
            tmp = self.dec_conv1(tmp)

        elif len(self.feat_channels) - 1 == 2:
            enc1 = self.enc_conv1(x)
            down1 = self.down1(enc1)

            enc2 = self.enc_conv2(down1)
            down2 = self.down2(enc2)

            bottleneck = self.enc_conv3(down2)

            tmp = self.upsample2(bottleneck)
            if self.TZSConv_skip[1]:
                enc2 = self.tzs_skip2(enc2)
            if self.Snack_skip[1]:
                enc2 = self.skip2(enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            if self.TZSConv_skip[0]:
                enc1 = self.tzs_skip1(enc1)
            if self.Snack_skip[0]:
                enc1 = self.skip1(enc1)
            tmp = torch.cat([tmp, enc1], dim=1)
            tmp = self.dec_conv1(tmp)

        else:
            raise ValueError("Unsupported feat_channels length")

        # return self.final_conv(tmp)
        return tmp


class UNet3D(nn.Module):
    """3D UNet backbone + detection heads."""

    def __init__(
        self,
        heads,
        input_channels=3,
        feat_channels=None,
        T_pooling=False,
        downsample_mode="stride",
        upsample_mode="deconv",
        Snack_skip=None,
        Snack_max_offset=None,
        UNet3D_skip="snack",
        TZSConv_skip=None,
    ):
        super().__init__()
        if feat_channels is None:
            feat_channels = [16, 32, 64, 128]

        self.backbone = UNet3DWithNormalConv3D(
            num_channels=input_channels,
            num_classes=1,
            feat_channels=feat_channels,
            upsample_mode=upsample_mode,
            T_pooling=T_pooling,
            downsample_mode=downsample_mode,
            Snack_skip=Snack_skip,
            Snack_max_offset=Snack_max_offset,
            UNet3D_skip=UNet3D_skip,
            TZSConv_skip=TZSConv_skip,
        )

        self.sigmoid = nn.Sigmoid()
        self.heads = heads

        head_conv = 64
        head_input_channel = feat_channels[0]
        for head in self.heads:
            classes = self.heads[head]
            if "hm" in head:
                fc = nn.Sequential(
                    nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                    nn.BatchNorm3d(head_conv),
                    nn.ReLU(),
                    nn.Conv3d(head_conv, classes, 3, padding=1, bias=True),
                )
            else:
                fc = nn.Sequential(
                    nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(head_conv, classes, 3, padding=1, bias=False),
                )

            if "hm" in head:
                last_layer = fc[-1] if isinstance(fc, nn.Sequential) else fc
                last_layer.bias.data.fill_(-2.19)

            self.__setattr__(head, fc)

    def forward(self, batch):
        # Backbone
        features = self.backbone(batch["input"])

        # Detection heads
        z = {}
        for head in self.heads:
            out = self.__getattr__(head)(features)
            if "hm" in head:
                out = torch.clamp(self.sigmoid(out), min=1e-4, max=1 - 1e-4)
            z[head] = out
        return [z]
