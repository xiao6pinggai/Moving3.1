from functools import partial

import torch.nn as nn

from lib.models.spconv_utils import replace_feature, spconv

import torch

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import spconv.pytorch as spconv

import torch
import torch.nn as nn
import spconv.pytorch as spconv

class GroupedDilatedBlock2(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 3, 4], 
                 indice_key=None, norm_fn=None, algo=None):
        """
        Grouped Dilated Block with 3x3 Conv Fusion
        """
        super().__init__()
        
        self.dilations = dilations
        num_groups = len(dilations)
        
        # --- 1. 动态计算通道分配 (余数给第一组) ---
        in_base = in_channels // num_groups
        in_remainder = in_channels % num_groups
        self.in_splits = [in_base + in_remainder] + [in_base] * (num_groups - 1)
        
        out_base = out_channels // num_groups
        out_remainder = out_channels % num_groups
        self.out_splits = [out_base + out_remainder] + [out_base] * (num_groups - 1)
        
        # --- 2. 构建并行分组卷积分支 ---
        self.convs = nn.ModuleList()
        for i, d in enumerate(self.dilations):
            # 每个分支由于 dilation 不同，访问的邻域不同，必须拥有唯一的 indice_key
            branch_key = f"{indice_key}_d{d}" 
            
            self.convs.append(spconv.SubMConv3d(
                self.in_splits[i], 
                self.out_splits[i], 
                kernel_size=kernel_size, 
                bias=False, 
                indice_key=branch_key, 
                dilation=d,
                padding=d, # SubM 要求 padding=dilation
                algo=algo
            ))
            
        # --- 3. 融合层 (改为 3x3 卷积) ---
        # 这一步将聚合不同 dilation 分支的信息，并再次融合邻域信息
        # 为了不改变几何形状，kernel=3 必须配合 padding=1
        
        # 为融合层分配独立的 key，因为它不仅是 1x1 混合，而是涉及到空间邻域搜索
        # fusion_key = f"{indice_key}_fusion" if indice_key is not None else None
        
        self.fusion = spconv.SubMConv3d(
            out_channels,     # 输入是拼接后的总通道数
            out_channels,     # 输出保持一致
            kernel_size=1,    # 你要求的 3x3
            padding=0,        # 必须为 1，对应 kernel=3, dilation=1
            bias=False,
            indice_key=indice_key, 
            algo=algo
        )
        
        # --- 4. 归一化与激活 ---
        if norm_fn is None:
            norm_fn = lambda c: nn.BatchNorm1d(c)
            
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn2 = norm_fn(out_channels)
        # bn2 后通常接 relu，这里复用
        
    def forward(self, x):
        features = x.features
        
        # 1. 动态切分 (Split)
        input_chunks = torch.split(features, self.in_splits, dim=1)
        
        output_chunks = []
        
        # 2. 并行卷积循环
        for i, conv in enumerate(self.convs):
            # 替换特征 -> 卷积 -> 取出特征
            # 避免在列表中存储完整的 SparseTensor 对象
            partial_x = x.replace_feature(input_chunks[i])
            output_chunks.append(conv(partial_x).features)
            
        # 3. 拼接 (Concat)
        out_features = torch.cat(output_chunks, dim=1)
        
        # 4. 第一阶段 BN + ReLU (在 Dense 状态下进行更快)
        out_features = self.bn1(out_features)
        out_features = self.relu(out_features)
        
        # 5. 3x3 卷积融合 (必须变回 SparseTensor)
        # 将处理好的特征装回 x
        x = x.replace_feature(out_features)
        
        # 执行 3x3 SubM 卷积 (这一步会建立或查询 _fusion 的哈希表)
        x = self.fusion(x)
        
        # 6. 第二阶段 BN + ReLU
        out_features = x.features
        out_features = self.bn2(out_features)
        out_features = self.relu(out_features)
        
        # 7. 返回结果
        return x.replace_feature(out_features)
class GroupedDilatedBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 3, 4], indice_key=None, norm_fn=None, algo=None):
        """
        3x3x3 分组空洞卷积模块
        包含 dilation=[1, 2, 3, 4] 的四个并行分支
        """
        super().__init__()
        
        # 确保通道数可以被分组数整除
        self.groups = len(dilations)
        assert in_channels % self.groups == 0, f"Input channels {in_channels} must be divisible by {self.groups}"
        assert out_channels % self.groups == 0, f"Output channels {out_channels} must be divisible by {self.groups}"
        
        self.dilations = dilations
        
        branch_in = in_channels // self.groups
        branch_out = out_channels // self.groups
        
        self.convs = nn.ModuleList()
        
        # 构建并行卷积分支
        for d in self.dilations:
            # 关键：为每个分支创建唯一的 indice_key
            current_key = f"{indice_key}_d{d}" if indice_key is not None else None
            
            self.convs.append(spconv.SubMConv3d(
                branch_in, 
                branch_out, 
                kernel_size=kernel_size, 
                bias=False, 
                indice_key=current_key, 
                dilation=d,
                padding=d, # SubMConv 中 padding=dilation 以保持几何对齐
                algo=algo
            ))
            
        # 注意：这里的 indice_key 建议设为 None 或者与主干一致，
        # 因为 1x1 卷积不依赖邻域，通常不需要特定的 geometry key，
        # 或者复用输入的 key 即可。上面循环结束后 current_key 是最后一个分支的 key，复用它也可以。
        self.fusion = spconv.SubMConv3d(
                out_channels,  # 注意：这里输入应该是 out_channels (因为前面 concat 后的维度是 out_channels)
                out_channels, 
                kernel_size=1, 
                bias=False, 
                indice_key=current_key, 
                padding=0, 
                algo=algo
            )
        
        # 归一化层
        if norm_fn is None:
            norm_fn = lambda c: nn.BatchNorm1d(c)
        self.bn = norm_fn(out_channels)
        self.bn2 = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. 通道切分
        input_chunks = x.features.chunk(self.groups, dim=1)
        
        output_chunks = []
        
        # 2. 并行卷积
        for chunk, conv in zip(input_chunks, self.convs):
            # 构造临时 SparseTensor 输入给 conv
            out_chunk = conv(x.replace_feature(chunk))
            output_chunks.append(out_chunk.features)
            
        # 3. 拼接特征 (此时变成普通 Tensor)
        out_features = torch.cat(output_chunks, dim=1)
        
        # 4. 执行第一阶段 BN + ReLU
        out_features = self.bn(out_features)
        out_features = self.relu(out_features)

        # ==================== 修复部分开始 ====================
        
        # 5. 额外 fusion
        # 错误点修复：self.fusion 需要 SparseTensor 输入
        # 将 dense tensor 包装回 SparseTensor
        x = x.replace_feature(out_features)
        
        # 执行 1x1 稀疏卷积
        x = self.fusion(x)
        
        # 再次提取特征进行 BN2 + ReLU
        out_features = x.features
        out_features = self.bn2(out_features)
        out_features = self.relu(out_features)

        # ==================== 修复部分结束 ====================
        
        # 6. 更新 SparseTensor 并返回
        x = x.replace_feature(out_features)
        
        return x

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,dialtion=1,
                   conv_type='subm', norm_fn=None, algo = None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False,padding=padding, dilation=dialtion, indice_key=indice_key,algo=algo)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key,algo=algo)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False,algo=algo)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, algo=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None

        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key,algo=algo
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key,algo=algo
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size, model_cfg=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # pool = spconv.pool.SparseMaxPool()

        algo = spconv.ConvAlgo.Native

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1',algo=algo),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block # 一个block就是一个conv+bn+relu，通过‘indice_key’确定是spconv还是subm

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1',algo=algo),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21] 一般情况是一个spconv跟两个subm层，公共构成了一个sp卷积单位（类比一个卷积层）
            block(16, 32, 3, norm_fn=norm_fn, stride=[2,2,2], padding=1, indice_key='spconv2', conv_type='spconv',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2',algo=algo),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=[2,2,2], padding=1, indice_key='spconv3', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3',algo=algo),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=[2,2,2], padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4',algo=algo),
        )

        self.upconv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup4', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup4',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup4',algo=algo),
        )
        self.up4 = spconv.SparseInverseConv3d(64, 64, 3, indice_key='spconv4', bias=False,algo=algo)
        self.upconv3 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 64, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup3', conv_type='spconv',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup3',algo=algo),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='submup3',algo=algo),
        )
        self.up3 = spconv.SparseInverseConv3d(64, 32, 3, indice_key='spconv3', bias=False, algo=algo)
        self.upconv2 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 32, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup2', conv_type='spconv',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submup2',algo=algo),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='submup2',algo=algo),
        )
        self.up2 = spconv.SparseInverseConv3d(32, 16, 3, indice_key='spconv2', bias=False,algo=algo)
        self.upconv1 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(32, 16, 3, norm_fn=norm_fn, padding=(1, 1, 1), indice_key='spconvup1', conv_type='spconv',algo=algo),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submup1',algo=algo),
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='submup1',algo=algo),
        )
        # self.up1 = spconv.SparseInverseConv3d(16, 16, 3, indice_key='spconv1', bias=False)
        # self.conv_out_debug = spconv.SparseSequential(
        #     # [200, 150, 2] -> [200, 150, 1]
        #     spconv.SparseConv3d(128, 128, 3, stride=(2, 1, 1), padding=1,
        #                         bias=False, indice_key='spconv_down_debug'),
        #     norm_fn(128),
        #     nn.ReLU(),
        # )

        last_pad = 0
        # last_pad = self.model_cfg.get('last_pad', last_pad)
        # self.conv_out = spconv.SparseSequential(
        #     # [200, 150, 5] -> [200, 150, 2]
        #     spconv.SparseConv3d(16, 1, 3, stride=(1, 1, 1), padding=1,
        #                         bias=False, indice_key='spconv_down2',algo=algo),
        #     # norm_fn(1),
        #     # nn.ReLU(),
        #     # spconv.SparseConv3d(16, 1, 3, stride=(1, 1, 1), padding=1,
        #     #                     bias=False, indice_key='spconv_down3'),
        # )
        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False, indice_key='submout',algo=algo),
            norm_fn(16),
            nn.ReLU(),
        )
        self.out_channel = 16
        self.num_point_features = self.out_channel
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
            'x_out': 128,   ######
            'bev': 256,
            'bev_2d': 512,
        }



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
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ## 构建sp_tensor, 最主要的是voxel_features和voxel_coords,
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,          #(N, 4) float32,  eg: torch.Size([64000, C])
            indices=voxel_coords.int(),       #(N, 4)[bs_idx, z, y, x], int32， eg: torch.Size([64000, 4])
            spatial_shape=self.sparse_shape,  #（3）  z, y, x 的容量， eg: array([  41, 1600, 1408])
            batch_size=batch_size             # bs  eg: 4
        )
        ## 创建好之后里面包括了features， indices， spatial_shape 等成分，可以直接用input_sp_tensor.features 方式调用
        
        # 后面的网络创建基本就照葫芦画瓢了，注意一点无论stride还是卷积核的三元组都是（维度3，维度2，维度1），是建议将时间维度T放在开头；单个整数默认是三元组内部元素相同

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_4 = self.upconv4(x_conv4)
        x_up_4 = self.up4(x_4)

        x_conv3 = self.replace_feature(x_conv3, torch.cat((x_conv3.features, x_up_4.features), dim=1))
        x_3 = self.upconv3(x_conv3)
        x_up_3 = self.up3(x_3)

        x_conv2 = self.replace_feature(x_conv2, torch.cat((x_conv2.features, x_up_3.features), dim=1))
        x_2 = self.upconv2(x_conv2)
        x_up_2 = self.up2(x_2)

        x_conv1 = self.replace_feature(x_conv1, torch.cat((x_conv1.features, x_up_2.features), dim=1))
        x_1 = self.upconv1(x_conv1)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_1)

        # out_debug = self.conv_out_debug(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1, #[41, 1600, 1408]; ([240884, 16])
                'x_conv2': x_conv2, #[21, 800, 704];   ([401179, 32])
                'x_conv3': x_conv3, #[11, 400, 352];   ([261878, 64])
                'x_conv4': x_conv4, #[5, 200, 176];    ([115506, 64])
                'x_out': out,   #####[2, 200, 176];    ([94886, 128])                          encoded_spconv_tensor
                # 'out_debug': out_debug #[1, 200, 176]; ([16894, 128])
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_out': [8, 8, 20],   ######
                'out_debug': [8, 8, 40], 
                'bev': 8, ######
                'bev_2d': 8, ######
            }
        })

        return batch_dict

    def replace_feature(self, out, new_features):
        if "replace_feature" in out.__dir__():
            # spconv 2.x behaviour
            return out.replace_feature(new_features)
        else:
            out.features = new_features
            return out


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

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

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict
