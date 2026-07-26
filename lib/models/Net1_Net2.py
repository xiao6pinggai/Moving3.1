import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)

from lib.models.noramlconv_unet3d2_1 import (
    DynamicTOSConvNet,
    EncoderOnlyConv3DProposalNet,
    LightWeightedConv3D,
    TOSConvNet,
    TPConvNet,
    TZSConvNet,
    UNet2DWithNormalConv2D,
    UNet3DWithNormalConv3D,
)


def _build_net1(net1name, feat_channels, T_pooling, groups, downsample_mode, img_num, opt):
    temporal_mode = getattr(opt, 'use_tzsconv', 'tzsconv')

    if net1name in ('UNet3DWithNormalConv3D', 'Unet3', 'UNet3D'):
        return UNet3DWithNormalConv3D(
            num_channels=3,
            num_classes=1,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            TConvOnly=False,
        )
    if net1name in ('UNet2DWithNormalConv2D', 'Unet2', 'UNet2D'):
        return UNet2DWithNormalConv2D(
            num_channels=3,
            num_classes=1,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
        )
    if net1name == 'LightWeightedConv3D':
        return LightWeightedConv3D(
            num_channels=3,
            num_classes=1,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
        )
    if net1name == 'EncoderOnlyConv3DProposalNet':
        return EncoderOnlyConv3DProposalNet(
            num_channels=3,
            num_classes=1,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            TConvOnly=False,
        )
    if net1name == 'TOSConvNet':
        return TOSConvNet(
            num_channels=3,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            use_tzsconv=temporal_mode,
            seq_len=img_num,
        )
    if net1name == 'TPConvNet':
        return TPConvNet(
            num_channels=3,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            use_tzsconv=temporal_mode,
            seq_len=img_num,
            tpdilation=getattr(opt, 'tpdilation', [1]),
            tprepeat=getattr(opt, 'tprepeat', 3),
        )
    if net1name in ('DynamicTOSConvNet', 'DynamicTOSconvNet'):
        return DynamicTOSConvNet(
            num_channels=3,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            use_tzsconv=temporal_mode,
            seq_len=img_num,
        )
    if net1name in ('TZSConvNet', 'TZSconvNet'):
        return TZSConvNet(
            num_channels=3,
            feat_channels=feat_channels,
            residual=None,
            upsample_mode='trilinear',
            activation=None,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            use_final_conv=True,
            use_tzsconv=temporal_mode,
            seq_len=img_num,
        )
    raise ValueError(f'Unknown net1name: {net1name}')


def _as_triple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


def _output_padding_from_stride(stride):
    stride = _as_triple(stride)
    return tuple(1 if s > 1 else 0 for s in stride)


def _align_to(x, ref):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode='trilinear', align_corners=True)


class DensePostActBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'inverseconv':
            conv = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=_output_padding_from_stride(stride),
                bias=False,
            )
        elif conv_type in ('subm', 'spconv'):
            conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        else:
            raise NotImplementedError(f'Unknown conv_type: {conv_type}')

        super().__init__(
            conv,
            nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )


class DenseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, eps=1e-3, momentum=0.01)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class DenseUNetV2_3_TNoDownV2(nn.Module):
    """Dense counterpart of the common sparse UNetV2_3_T_nodown_v2 path.

    This keeps the sparse model's channel schedule and UR decoder logic, but
    removes sparse tensors, thresholding, top-k/MFE modules, and adaptive
    sampling. It intentionally contains no cosv23/v23 dependency.
    """

    def __init__(self, input_channels, use_bottle_block=True):
        super().__init__()
        self.conv_input = DensePostActBlock(input_channels, 16, 3, padding=1)

        self.conv1 = nn.Sequential(
            DensePostActBlock(16, 16, 3, padding=1),
            DensePostActBlock(16, 16, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            DensePostActBlock(16, 32, 3, stride=(1, 2, 2), padding=1, conv_type='spconv'),
            DensePostActBlock(32, 32, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            DensePostActBlock(32, 64, 3, stride=(1, 2, 2), padding=1, conv_type='spconv'),
            DensePostActBlock(64, 64, 3, padding=1),
        )

        self.use_bottle_block = use_bottle_block
        self.x_bottle_down = DensePostActBlock(64, 128, 3, stride=(1, 2, 2), padding=1, conv_type='spconv')
        self.x_bottle_enhancement = DensePostActBlock(128, 128, 3, padding=1) if use_bottle_block else nn.Identity()
        self.x_bottle_up = DensePostActBlock(128, 64, 3, stride=(1, 2, 2), padding=1, conv_type='inverseconv')

        self.conv_up_t3 = nn.Identity()
        self.conv_up_m3 = DensePostActBlock(128, 64, 3, padding=1)
        self.inv_conv3 = DensePostActBlock(64, 32, 3, stride=(1, 2, 2), padding=1, conv_type='inverseconv')

        self.conv_up_t2 = nn.Identity()
        self.conv_up_m2 = DensePostActBlock(64, 32, 3, padding=1)
        self.inv_conv2 = DensePostActBlock(32, 16, 3, stride=(1, 2, 2), padding=1, conv_type='inverseconv')

        self.conv_up_t1 = nn.Identity()
        self.conv_up_m1 = DensePostActBlock(32, 16, 3, padding=1)
        self.conv5 = DensePostActBlock(16, 16, 3, padding=1)

        self.num_point_features = 16
        self.out_channel = 16

    @staticmethod
    def channel_reduction(x, out_channels):
        b, in_channels, t, h, w = x.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)
        return x.view(b, out_channels, -1, t, h, w).sum(dim=2)

    def ur_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv, target=None):
        x_trans = conv_t(x_lateral)
        x_bottom = _align_to(x_bottom, x_trans)
        x = torch.cat((x_bottom, x_trans), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.shape[1])
        x = x_m + x
        x = conv_inv(x)
        if target is not None:
            x = _align_to(x, target)
        return x

    def forward(self, x):
        x = self.conv_input(x)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)

        x_bottle = self.x_bottle_down(x_conv3)
        x_bottle = self.x_bottle_enhancement(x_bottle)
        x_bottle = self.x_bottle_up(x_bottle)
        x_bottle = _align_to(x_bottle, x_conv3)

        x_up3 = self.ur_block_forward(
            x_conv3, x_bottle, self.conv_up_t3, self.conv_up_m3, self.inv_conv3, target=x_conv2
        )
        x_up2 = self.ur_block_forward(
            x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2, target=x_conv1
        )
        x_up1 = self.ur_block_forward(
            x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5
        )
        return x_up1


class Net1Net2(nn.Module):
    def __init__(
        self,
        heads,
        image_size=[512, 512],
        img_num=20,
        layers=3.61,
        thresh=None,
        input_channels=1,
        feat_channels=[16, 32, 64],
        T_pooling=False,
        groups=2,
        downsample_mode='maxpool',
        net1name='UNet3D',
        opt=None,
    ):
        super().__init__()
        self.net1name = net1name
        self.heads = heads
        self.sigmoid = nn.Sigmoid()
        self.thresh = thresh

        self.I2PNet = _build_net1(
            net1name=net1name,
            feat_channels=feat_channels,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            img_num=img_num,
            opt=opt,
        )

        if float(layers) != 3.61:
            raise ValueError('Net1_Net2 currently supports layers=3.61 for the dense Net2 ablation.')

        bottle_mode = getattr(opt, 'bottle_enhancement', 'block')
        use_bottle_block = str(bottle_mode).strip().lower() not in ('none', 'null', 'false', '0', '')
        self.net2 = DenseUNetV2_3_TNoDownV2(feat_channels[0], use_bottle_block=use_bottle_block)

        head_conv = 128
        head_input_channel = self.net2.num_point_features
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                if 'hm' in head:
                    fc = nn.Sequential(
                        nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                        nn.ReLU(),
                        nn.Conv3d(head_conv, classes, 3, padding=1, bias=True),
                    )
                else:
                    fc = nn.Sequential(
                        nn.Conv3d(head_input_channel, head_conv, 3, padding=1, bias=False),
                        nn.ReLU(),
                        nn.Conv3d(head_conv, classes, 3, padding=1, bias=False),
                    )
            else:
                fc = nn.Conv3d(head_input_channel, classes, 3, padding=1, bias=True)

            if 'hm' in head:
                last = fc[-1] if isinstance(fc, nn.Sequential) else fc
                last.bias.data.fill_(-2.19)
            self.__setattr__(head, fc)

    def _forward_patch(self, batch, patch_w):
        b, c, t, h, w = batch['input'].shape
        net1_output = self.I2PNet(batch['input'])
        dense_features = self.net2(net1_output)

        z = {}
        for head in self.heads:
            spatial_features = self.__getattr__(head)(dense_features)
            if 'hm' in head:
                spatial_features = self.sigmoid(spatial_features)
                spatial_features = torch.clamp(spatial_features, min=1e-4, max=1 - 1e-4)
            z[head] = spatial_features[..., :patch_w]

        if net1_output.shape[1] > 1:
            soft_mask = self.sigmoid(net1_output.mean(dim=1, keepdim=True))
        else:
            soft_mask = self.sigmoid(net1_output)
        z['hm_large_heatmap'] = net1_output[..., :patch_w]
        z['soft_mask'] = soft_mask[..., :patch_w]
        z['sampling_rate'] = torch.ones((), device=batch['input'].device)
        return z

    def forward(self, batch):
        b, c, t, h, w = batch['input'].shape

        if not self.training and w >= 1920:
            w_half = w // 2

            batch_left = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_left['input'] = batch['input'][..., :w_half]
            with torch.no_grad():
                z_left = self._forward_patch(batch_left, patch_w=w_half)

            batch_right = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_right['input'] = batch['input'][..., w_half:]
            with torch.no_grad():
                z_right = self._forward_patch(batch_right, patch_w=w - w_half)

            z_merged = {}
            for head in self.heads:
                z_merged[head] = torch.cat([z_left[head], z_right[head]], dim=-1)
            z_merged['hm_large_heatmap'] = torch.cat(
                [z_left['hm_large_heatmap'], z_right['hm_large_heatmap']], dim=-1
            )
            z_merged['soft_mask'] = torch.cat([z_left['soft_mask'], z_right['soft_mask']], dim=-1)
            z_merged['sampling_rate'] = torch.ones((), device=batch['input'].device)
            return [z_merged]

        return [self._forward_patch(batch, patch_w=w)]


def Net1_Net2(
    heads,
    image_size=[512, 512],
    img_num=20,
    layers=3.61,
    thresh=None,
    input_channels=1,
    feat_channels=[16, 32, 64],
    T_pooling=False,
    groups=2,
    downsample_mode='maxpool',
    net1name='UNet3D',
    opt=None,
):
    return Net1Net2(
        heads,
        image_size=image_size,
        img_num=img_num,
        layers=layers,
        thresh=thresh,
        input_channels=input_channels,
        feat_channels=feat_channels,
        T_pooling=T_pooling,
        groups=groups,
        downsample_mode=downsample_mode,
        net1name=net1name,
        opt=opt,
    )


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    heads = {'hm': 1, 'wh': 2, 'reg': 2}
    model = Net1_Net2(
        heads=heads,
        image_size=[128, 128],
        img_num=10,
        layers=3.61,
        feat_channels=[8, 8, 8, 8],
        net1name='Unet3',
    ).to(device)
    batch = {'input': torch.randn(1, 3, 10, 128, 128, device=device)}
    out = model(batch)[0]
    for k, v in out.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape) if v.dim() > 0 else v.item())
