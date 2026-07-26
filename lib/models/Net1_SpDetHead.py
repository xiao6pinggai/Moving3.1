import os
import sys

import torch
import torch.nn as nn

_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)

from lib.models.spconv_utils import replace_feature, spconv
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


class Net1SparseDetHead(nn.Module):
    """I2PSOD ablation: SRC + ACS + sparse detection head."""

    def __init__(
        self,
        heads,
        image_size=[512, 512],
        img_num=20,
        layers=3,
        thresh=None,
        input_channels=1,
        feat_channels=[16, 32, 64, 128],
        T_pooling=False,
        groups=1,
        downsample_mode='stride',
        net1name='UNet3DWithNormalConv3D',
        opt=None,
    ):
        super().__init__()
        self.net1name = net1name
        self.thresh = thresh
        self.heads = heads
        self.sigmoid = nn.Sigmoid()
        self.net1_feature_channels = feat_channels[0]
        self.points_all = img_num * image_size[0] * image_size[1]

        self.I2PNet = self._build_i2pnet(
            feat_channels=feat_channels,
            T_pooling=T_pooling,
            groups=groups,
            downsample_mode=downsample_mode,
            img_num=img_num,
            opt=opt,
        )
        self._build_sparse_heads(head_input_channel=self.net1_feature_channels)

    def _build_i2pnet(self, feat_channels, T_pooling, groups, downsample_mode, img_num, opt):
        temporal_mode = getattr(opt, 'use_tzsconv', 'tzsconv')

        if self.net1name in ('UNet3DWithNormalConv3D', 'Unet3'):
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
        if self.net1name in ('UNet2DWithNormalConv2D', 'Unet2'):
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
        if self.net1name == 'LightWeightedConv3D':
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
        if self.net1name == 'EncoderOnlyConv3DProposalNet':
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
        if self.net1name == 'TOSConvNet':
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
        if self.net1name == 'TPConvNet':
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
                tpdilation=getattr(opt, 'tpdilation', 1),
                tprepeat=getattr(opt, 'tprepeat', 1),
            )
        if self.net1name in ('DynamicTOSConvNet', 'DynamicTOSconvNet'):
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
        if self.net1name in ('TZSConvNet', 'TZSconvNet'):
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

        raise ValueError(f'Unsupported net1name: {self.net1name}')

    def _build_sparse_heads(self, head_input_channel):
        head_conv = 128
        for head in self.heads:
            classes = self.heads[head]
            name_1 = 'subm1' + head
            name_2 = 'subm2' + head
            if head_conv > 0:
                if 'hm' in head:
                    fc = spconv.SparseSequential(
                        spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=True, indice_key=name_2),
                    )
                else:
                    fc = spconv.SparseSequential(
                        spconv.SubMConv3d(head_input_channel, head_conv, 3, padding=1, bias=False, indice_key=name_1),
                        nn.ReLU(),
                        spconv.SubMConv3d(head_conv, classes, 3, padding=1, bias=False, indice_key=name_2),
                    )
            else:
                fc = spconv.SubMConv3d(head_input_channel, classes, 3, padding=1, bias=True, indice_key=name_1)

            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            setattr(self, head, fc)

    def get_mask_by_mean_std(self, soft_mask, var_coeff=3):
        var_coeff = var_coeff if var_coeff is not None else 3
        B, C, T, H, W = soft_mask.shape
        assert C == 1, 'soft_mask channel must be 1'

        mask_mean = torch.mean(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        mask_std = torch.std(soft_mask, dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        dynamic_thresh = mask_mean + var_coeff * mask_std
        # dynamic_thresh = 0.5
        binary_mask = (soft_mask > dynamic_thresh).float()

        binary_mask_flat = binary_mask.view(B, -1)
        soft_mask_flat = soft_mask.view(B, -1)
        valid_pts_count = torch.sum(binary_mask_flat, dim=-1)
        min_points = min(50, soft_mask_flat.shape[-1])

        for batch_idx in range(B):
            if valid_pts_count[batch_idx] < min_points:
                topk_idx = torch.topk(soft_mask_flat[batch_idx], k=min_points, dim=0, largest=True).indices
                binary_mask_flat[batch_idx].zero_()
                binary_mask_flat[batch_idx, topk_idx] = 1.0

        return binary_mask_flat.view(B, 1, T, H, W)

    def _make_sparse_tensor(self, net1_output, binary_mask):
        device = net1_output.device
        b, channels, t, h, w = net1_output.shape
        coords = torch.nonzero(binary_mask.squeeze(1)).contiguous()
        total_points = b * t * h * w
        sampling_rate = coords.shape[0] / total_points if total_points > 0 else 0

        batch_idx = coords[:, 0]
        t_idx = coords[:, 1]
        h_idx = coords[:, 2]
        w_idx = coords[:, 3]
        flattened_indices = batch_idx * t * h * w + t_idx * h * w + h_idx * w + w_idx
        voxel_features_flat = net1_output.permute(0, 2, 3, 4, 1).reshape(b * t * h * w, channels)
        voxel_features = voxel_features_flat[flattened_indices]
        voxel_coords = coords.to(device)

        if coords.shape[0] == 0:
            print('Warning: No points generated from I2PNet!')
        if voxel_features.shape[0] == 0:
            print('Warning: No voxel features selected for sparse det head!')

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=[int(t), int(h), int(w)],
            batch_size=b,
        )
        return input_sp_tensor, voxel_coords, sampling_rate

    def _run_sparse_heads(self, input_sp_tensor, patch_w):
        z = {}
        for head in self.heads:
            out_h = getattr(self, head)(input_sp_tensor)
            if 'hm' in head:
                out_h = replace_feature(out_h, self.sigmoid(out_h.features))
                spatial_features = torch.clamp(out_h.dense(), min=1e-4, max=1 - 1e-4)
            else:
                spatial_features = out_h.dense()
            z[head] = spatial_features[..., :patch_w]
        return z

    def _forward_patch(self, batch, patch_w):
        device = batch['input'].device
        net1_output = self.I2PNet(batch['input'])
        if isinstance(net1_output, dict):
            raise ValueError('EncoderOnlyConv3DProposalNet returns a dict and is not supported by Net1_SpDetHead')

        if net1_output.shape[1] > 1:
            voxel_score_logits = net1_output.mean(dim=1, keepdim=True)
        else:
            voxel_score_logits = net1_output
        soft_mask = self.sigmoid(voxel_score_logits)
        binary_mask = self.get_mask_by_mean_std(soft_mask=soft_mask, var_coeff=self.thresh)

        input_sp_tensor, voxel_coords, sampling_rate = self._make_sparse_tensor(net1_output, binary_mask)
        z = self._run_sparse_heads(input_sp_tensor, patch_w=patch_w)
        z['hm_large_heatmap'] = net1_output
        z['voxel_coords'] = voxel_coords
        z['soft_mask'] = soft_mask
        z['sampling_rate'] = torch.tensor(sampling_rate, device=device)
        return z

    def forward(self, batch):
        b, c, t, h, w = batch['input'].shape

        if not self.training and w >= 1920:
            w_half = w // 2

            batch_left = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_left['input'] = batch['input'][..., :w_half]
            with torch.no_grad():
                z_left = self._forward_patch(batch_left, patch_w=w_half)
            del batch_left
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_right = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch_right['input'] = batch['input'][..., w_half:]
            with torch.no_grad():
                z_right = self._forward_patch(batch_right, patch_w=w - w_half)
            del batch_right
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            z_merged = {}
            for head in self.heads:
                z_merged[head] = torch.cat([z_left[head], z_right[head]], dim=-1)
            z_merged['hm_large_heatmap'] = torch.cat([z_left['hm_large_heatmap'], z_right['hm_large_heatmap']], dim=-1)
            z_merged['soft_mask'] = torch.cat([z_left['soft_mask'], z_right['soft_mask']], dim=-1)

            coords_left = z_left['voxel_coords']
            coords_right = z_right['voxel_coords'].clone()
            coords_right[:, 3] += w_half
            z_merged['voxel_coords'] = torch.cat([coords_left, coords_right], dim=0)
            total_points = b * t * h * w
            sampling_rate = z_merged['voxel_coords'].shape[0] / total_points if total_points > 0 else 0
            z_merged['sampling_rate'] = torch.tensor(sampling_rate, device=batch['input'].device)
            return [z_merged]

        return [self._forward_patch(batch, patch_w=w)]


def Net1_SpDetHead(
    heads,
    image_size=[512, 512],
    img_num=20,
    layers=4,
    thresh=None,
    input_channels=1,
    feat_channels=[16, 32, 64],
    T_pooling=False,
    groups=2,
    downsample_mode='maxpool',
    net1name='UNet3D',
    opt=None,
):
    return Net1SparseDetHead(
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
