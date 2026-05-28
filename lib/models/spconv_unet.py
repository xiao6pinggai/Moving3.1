from functools import partial

import numpy as np
import torch
import torch.nn as nn
import os, sys

ROOT_DIR = "/root/autodl-tmp/Moving3.1"
# 确保根目录在sys.path首位（覆盖默认的子目录）
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from lib.models.spconv_utils import replace_feature, spconv
# from lib.utils import common_utils
from lib.models.spconv_backbone import GroupedDilatedBlock, GroupedDilatedBlock2, post_act_block
from lib.models.cos_matchv10 import SparseSymmetricCosineAttention
# from lib.models.se import SparseSymmetricCosineAttention, SparseSEModule
class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out
class SparseBasicBlock2D3D(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock2D3D, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), bias=False, indice_key=f'{indice_key}_temp'
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

class UNetV2_3(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    

class UNetV2_2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        # self.conv3 = spconv.SparseSequential(
        #     # [800, 704, 21] <- [400, 352, 11]
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        # )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        # self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        # self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        # self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv2)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        # x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_conv2, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    

class UNetV2_3_32(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(128, 256, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(256),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(128, 128, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(256, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(64, 64, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(128, 64, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(32, 32, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 32

        self.out_channel = 32

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

class UNetV2_3_T_nodown(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict



class UNetV2_3_T_nodown_maxpool(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_maxpool = True
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # === 核心修改：构建下采样 Block 的辅助函数 ===
        # === 核心修改：构建下采样 Block 的辅助函数 ===
        def make_down_block(in_ch, out_ch, key, stride):
            if self.use_maxpool:
                return spconv.SparseSequential(
                    # 1. 修正点：将 indice_key 给 MaxPool
                    # 这样 InverseConv 才能通过这个 key 找到下采样时的坐标映射
                    spconv.SparseMaxPool3d(kernel_size=3, stride=stride, padding=1, indice_key=key),
                    
                    # 2. 修正点：后面的 SubMConv 必须用一个新的 key
                    # 如果复用 key，会覆盖掉 MaxPool 的索引信息，导致报错。
                    # 这里简单的加上 '_subm' 后缀即可
                    block(in_ch, out_ch, 3, norm_fn=norm_fn, padding=1, indice_key=key + '_subm', conv_type='subm') 
                )
            else:
                return block(in_ch, out_ch, 3, norm_fn=norm_fn, stride=stride, padding=1, 
                             indice_key=key, conv_type='spconv')
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            # block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'),
            make_down_block(16,32,'spconv2',stride=(1,2,2)),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            # block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'),
            make_down_block(32,64,'spconv3',stride=(1,2,2)),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )

        if self.model_cfg is not None:
            if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
                last_pad = self.model_cfg.get('last_pad', 0)

                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv3)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict
    



class UNetV2_3_T_nodown_v2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        # self.cos_match2 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.cos_match3 = SparseSymmetricCosineAttention(kernel_size=7)
        # self.cos_match4 = SparseSymmetricCosineAttention(kernel_size=5)
        self.conv1 = spconv.SparseSequential(
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'), # d1
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'), #d2
            # block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )
        self.x_bottle = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv_b', conv_type='spconv'),#d3
            # 1. 升维: 64 -> 128 (使用 SubMConv 保持形状，或者 stride=1 的 Conv)
            # block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_1'),
            # 真正的bottle
            # 2. (可选) 中间深层处理: 128 -> 128
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_2'),
            # 3. 降维: 128 -> 64
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'),

            block(128, 64, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv_b', conv_type='inverseconv')
        )
        

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]

        # self.conv_up_t1 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.se1 = SparseSEModule(channel=16, reduction=2)
        # self.se2 = SparseSEModule(channel=32, reduction=2)
        # self.se3 = SparseSEModule(channel=64, reduction=2)
        # self.sptial2d1 = block(16, 16, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm1_1') # 3d
        # self.sptial2d2 = block(32, 32, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm2_2')
        # self.sptial2d3 = block(64, 64, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm3_3')


        # mfe # V2!!!
        self.sptial2d1 = block(16, 16, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm1_1') # 2d
        self.sptial2d2 = block(32, 32, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm2_2')
        self.sptial2d3 = block(64, 64, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm3_3')
        self.shortcut1 = SparseSymmetricCosineAttention(in_channels=16, kernel_size=41, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d1)
        self.shortcut2 = SparseSymmetricCosineAttention(in_channels=32, kernel_size=21, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d2)
        self.shortcut3 = SparseSymmetricCosineAttention(in_channels=64, kernel_size=11, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=1, conv=self.sptial2d3)
        # mfe
        
        
        # self.shortcut1fusion = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        # self.shortcut2fusion = block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        # self.shortcut3fusion = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')


        # mfe
        self.shortcut1fusion = GroupedDilatedBlock(16, 16, 3, dilations=[1, 2, 3,4], indice_key='subm1')
        self.shortcut2fusion = GroupedDilatedBlock(32, 32, 3, dilations=[1, 2, 3,4], indice_key='subm2')
        self.shortcut3fusion = GroupedDilatedBlock(64, 64, 3, dilations=[1, 2, 3,4], indice_key='subm3')
        # mfe


        # self.shortcut1fusion = SparseBasicBlock2D3D(16, 16, norm_fn=norm_fn, indice_key='subm1')
        # self.shortcut2fusion = SparseBasicBlock2D3D(32, 32, norm_fn=norm_fn, indice_key='subm2')
        # self.shortcut3fusion = SparseBasicBlock2D3D(64, 64, norm_fn=norm_fn, indice_key='subm3')
        
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x_trans = x_lateral
    #     x = x_trans
    #     x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x = replace_feature(x, x_m.features + x.features)
    #     x = conv_inv(x)
    #     return x
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
            # x_trans = conv_t(x_lateral)
            x_trans = x_lateral
            x = x_trans
            x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1)) # concat
            x_m = conv_m(x) # 64-->32
            x = self.channel_reduction(x, x_m.features.shape[1]) # reduce channel
            x = replace_feature(x, x_m.features + x.features) # resuial x是融合之前，x_m是融合之后
            x = conv_inv(x) # upsample 32-->16
            return x
    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        ###
        # enhanced_feat1,_ = self.cos_match2(x_conv1.indices, x_conv1.features)
        # x_conv1 = replace_feature(x_conv1, enhanced_feat1)

        x_conv2 = self.conv2(x_conv1)
        ###
        # enhanced_feat2,_ = self.cos_match3(x_conv2.indices, x_conv2.features)
        # x_conv2 = replace_feature(x_conv2, enhanced_feat2)

        x_conv3 = self.conv3(x_conv2)
        ###
        # enhanced_feat3,_ = self.cos_match4(x_conv3.indices, x_conv3.features)
        # x_conv3 = replace_feature(x_conv3, enhanced_feat3)
        # x_conv4 = self.conv4(x_conv3)

        
        x_bottle = self.x_bottle(x_conv3)
        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11

        # w/o shortcut process
        # x_conv3 = self.se3(x_conv3)
        x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
        x_conv3 = self.shortcut3fusion(x_conv3)
        x_up3 = self.UR_block_forward(x_conv3, x_bottle, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        # w/o shortcut process
        # x_conv2 = self.se2(x_conv2)
        x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
        x_conv2 = self.shortcut2fusion(x_conv2)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        # w/o shortcut process
        # x_conv1 = self.se1(x_conv1)
        x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
        x_conv1 = self.shortcut1fusion(x_conv1)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict


class UNetV2_3_T_nodown_v3(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size=None, point_cloud_range=None, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # None
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block
        # self.cos_match2 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.cos_match3 = SparseSymmetricCosineAttention(kernel_size=7)
        # self.cos_match4 = SparseSymmetricCosineAttention(kernel_size=5)
        self.conv1 = spconv.SparseSequential(
            # block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv2', conv_type='spconv'), # d1
            # block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv3', conv_type='spconv'), #d2
            # block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        #     block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        # )
        self.x_bottle = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=(1,2,2), padding=1, indice_key='spconv_b', conv_type='spconv'),#d3
            # 1. 升维: 64 -> 128 (使用 SubMConv 保持形状，或者 stride=1 的 Conv)
            # block(64, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_1'),
            # 真正的bottle
            # 2. (可选) 中间深层处理: 128 -> 128
            # block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_2'),
            # 3. 降维: 128 -> 64
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='bottle_3'),

            block(128, 64, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv_b', conv_type='inverseconv')
        )
        

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        # self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        # self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        # self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]

        # self.conv_up_t1 = SparseSymmetricCosineAttention(kernel_size=9)
        # self.se1 = SparseSEModule(channel=16, reduction=2)
        # self.se2 = SparseSEModule(channel=32, reduction=2)
        # self.se3 = SparseSEModule(channel=64, reduction=2)
        # self.sptial2d1 = block(16, 16, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm1_1') # 3d
        # self.sptial2d2 = block(32, 32, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm2_2')
        # self.sptial2d3 = block(64, 64, (3,3,3), norm_fn=norm_fn, dialtion=(2,1,1),padding=(2,1,1), indice_key='subm3_3')


        # 
        self.sptial2d1 = block(16, 16, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm1_1') # 2d
        self.sptial2d2 = block(32, 32, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm2_2')
        self.sptial2d3 = block(64, 64, (1,3,3), norm_fn=norm_fn, dialtion=(1,1,1),padding=(0,1,1), indice_key='subm3_3')

        # self.sptial2d1 = GroupedDilatedBlock2(16, 16, 3, dilations=[1, 2, 3], indice_key='subm1') # 2d
        # self.sptial2d2 = GroupedDilatedBlock2(32, 32, 3, dilations=[1, 2, 3], indice_key='subm2')
        # self.sptial2d3 = GroupedDilatedBlock2(64, 64, 3, dilations=[1, 2, 3], indice_key='subm3')
        
        self.shortcut1 = SparseSymmetricCosineAttention(in_channels=16, kernel_size=13, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d1)
        self.shortcut2 = SparseSymmetricCosineAttention(in_channels=32, kernel_size=9, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d2)
        self.shortcut3 = SparseSymmetricCosineAttention(in_channels=64, kernel_size=5, use_qkv=False, use_maxpool=False, alpha=0.5, use_biqkv=False,temporal_dilation=2, conv=self.sptial2d3)
        # 13-->6 9-->4*2  5-->2*4
        
        
        # self.shortcut1fusion = block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        # self.shortcut2fusion = block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2')
        # self.shortcut3fusion = block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')


        # 
        self.shortcut1fusion = GroupedDilatedBlock2(16, 16, 3, dilations=[1, 2, 3, 4], indice_key='subm1')
        self.shortcut2fusion = GroupedDilatedBlock2(32, 32, 3, dilations=[1, 2, 3, 4], indice_key='subm2')
        self.shortcut3fusion = GroupedDilatedBlock2(64, 64, 3, dilations=[1, 2, 3, 4], indice_key='subm3')
        # 


        # self.shortcut1fusion = SparseBasicBlock2D3D(16, 16, norm_fn=norm_fn, indice_key='subm1')
        # self.shortcut2fusion = SparseBasicBlock2D3D(32, 32, norm_fn=norm_fn, indice_key='subm2')
        # self.shortcut3fusion = SparseBasicBlock2D3D(64, 64, norm_fn=norm_fn, indice_key='subm3')
        
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, stride=(1, 2, 2), padding=1,norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

        self.out_channel = 16

    # def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
    #     x_trans = conv_t(x_lateral)
    #     x_trans = x_lateral
    #     x = x_trans
    #     x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
    #     x_m = conv_m(x)
    #     x = self.channel_reduction(x, x_m.features.shape[1])
    #     x = replace_feature(x, x_m.features + x.features)
    #     x = conv_inv(x)
    #     return x
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
            # x_trans = conv_t(x_lateral)
            x_trans = x_lateral
            x = x_trans
            x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1)) # concat
            x_m = conv_m(x) # 64-->32
            x = self.channel_reduction(x, x_m.features.shape[1]) # reduce channel
            x = replace_feature(x, x_m.features + x.features) # resuial x是融合之前，x_m是融合之后
            x = conv_inv(x) # upsample 32-->16
            return x
    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        ###
        # enhanced_feat1,_ = self.cos_match2(x_conv1.indices, x_conv1.features)
        # x_conv1 = replace_feature(x_conv1, enhanced_feat1)

        x_conv2 = self.conv2(x_conv1)
        ###
        # enhanced_feat2,_ = self.cos_match3(x_conv2.indices, x_conv2.features)
        # x_conv2 = replace_feature(x_conv2, enhanced_feat2)

        x_conv3 = self.conv3(x_conv2)
        ###
        # enhanced_feat3,_ = self.cos_match4(x_conv3.indices, x_conv3.features)
        # x_conv3 = replace_feature(x_conv3, enhanced_feat3)
        # x_conv4 = self.conv4(x_conv3)

        
        x_bottle = self.x_bottle(x_conv3)
        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11

        # w/o shortcut process
        # x_conv3 = self.se3(x_conv3)
        x_conv3 = replace_feature(x_conv3, self.shortcut3(x_conv3)[0])
        x_up3 = self.UR_block_forward(x_conv3, x_bottle, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        # w/o shortcut process
        # x_conv2 = self.se2(x_conv2)
        x_conv2 = replace_feature(x_conv2, self.shortcut2(x_conv2)[0])
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        # w/o shortcut process
        # x_conv1 = self.se1(x_conv1)
        x_conv1 = replace_feature(x_conv1, self.shortcut1(x_conv1)[0])
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['point_features'] = x_up1
        batch_dict['encoded_spconv_tensor'] = x_up1
        batch_dict['encoded_spconv_tensor_stride'] = 8
        # point_coords = common_utils.get_voxel_centers(
        #     x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
        #     point_cloud_range=self.point_cloud_range
        # )
        # batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict

if __name__ == '__main__':
    image_size = [512,512]
    img_num = 10
    grid_size = np.array([image_size[1], image_size[0], img_num - 1])
    model = UNetV2_3_T_nodown_v3(1,grid_size)
    print(model)