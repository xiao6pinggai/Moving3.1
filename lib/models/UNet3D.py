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

from TVDCN.dcn_nd import DeformConv3d, SnakeDeformConv3d, SnakeUnrestDeformConv3d, VelocitySnakeDeformConv3d
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
    def __init__(self, channels, kernel=(3, 1, 1), offset_kernel=None, extend_scope=10.0):
        super().__init__()
        self.block = Sequential(
            SnakeDeformConv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=tuple(size // 2 for size in kernel),
                offset_kernel_size=offset_kernel,
                bias=False,
                extend_scope=extend_scope,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalSnakeUnrestConv3D_Block(Module):
    def __init__(self, channels, kernel=(3, 1, 1), offset_kernel=None, extend_scope=10.0):
        super().__init__()
        self.block = Sequential(
            SnakeUnrestDeformConv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=tuple(size // 2 for size in kernel),
                offset_kernel_size=offset_kernel,
                bias=False,
                extend_scope=extend_scope,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalVelocitySnakeConv3D_Block(Module):
    def __init__(self, channels, kernel=(3, 1, 1), offset_kernel=None, extend_scope=10.0, use_residual=True):
        super().__init__()
        self.block = Sequential(
            VelocitySnakeDeformConv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=tuple(size // 2 for size in kernel),
                offset_kernel_size=offset_kernel,
                bias=False,
                extend_scope=extend_scope,
                use_residual=use_residual,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalConv3D_Block(Module):
    def __init__(self, channels, kernel=(3, 1, 1)):
        super().__init__()
        self.block = Sequential(
            Conv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=tuple(size // 2 for size in kernel),
                bias=False,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TemporalDCN3D_Block(Module):
    def __init__(self, channels, kernel=(3, 1, 1), offset_kernel=None):
        super().__init__()
        self.block = Sequential(
            DeformConv3d(
                channels,
                channels,
                kernel_size=kernel,
                padding=tuple(size // 2 for size in kernel),
                offset_kernel_size=offset_kernel,
                bias=False,
            ),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# class TMixer(Module):
#     """TPro-style multi-head temporal projection at each spatial location."""

#     def __init__(self, channels, num_frames, num_heads=8):
#         super().__init__()
#         if channels % num_heads != 0:
#             raise ValueError("TMixer channels must be divisible by TMixer_num_heads")
#         self.num_frames = num_frames
#         self.num_heads = num_heads
#         self.head_channels = channels // num_heads
#         self.QK_heads = nn.ModuleList([nn.Linear(num_frames, num_frames) for _ in range(num_heads)])
#         self.relu = ReLU(inplace=True)
#         self.norm1 = BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True)
#         self.conv = Sequential(
#             Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True),
#             BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
#             ReLU(inplace=True),
#         )

#     def forward(self, x):
#         if x.shape[2] != self.num_frames:
#             raise ValueError("TMixer input temporal length does not match num_frames")
#         batch_size, channels, _, height, width = x.shape
#         value = x.permute(0, 3, 4, 1, 2) / math.sqrt(self.num_frames)
#         value = value.reshape(batch_size, height, width, self.num_heads, self.head_channels, self.num_frames)
#         projected = torch.cat(
#             [head(value[:, :, :, index]) for index, head in enumerate(self.QK_heads)],
#             dim=3,
#         )
#         projected = projected.permute(0, 3, 4, 1, 2)
#         return self.conv(self.relu(self.norm1(projected)))
class TMixer(Module):
    """Shared global Linear(T, T) projection followed by BN and ReLU."""

    def __init__(self, channels, num_frames, num_heads=8):
        super().__init__()
        del num_heads  # Kept in the signature for backward-compatible CLI use.
        self.channels = channels
        self.num_frames = num_frames
        self.linear = nn.Linear(num_frames, num_frames, bias=True)
        # self.relu = ReLU(inplace=True)
        # self.norm1 = BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        if x.shape[2] != self.num_frames:
            raise ValueError("TMixer input temporal length does not match num_frames")
        if x.shape[1] != self.channels:
            raise ValueError("TMixer input channels do not match channels")
        projected = self.linear(x.permute(0, 1, 3, 4, 2))
        projected = projected.permute(0, 1, 4, 2, 3).contiguous()
        # return self.relu(self.norm1(projected))
        return projected


class Multi_TMixer(Module):
    """Global and stride-two temporal mixing with shared stride-two parameters."""

    def __init__(self, channels, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.global_linear = nn.Linear(num_frames, num_frames)
        self.stride2_length = (num_frames + 1) // 2
        self.stride2_weight = nn.Parameter(torch.empty(self.stride2_length, self.stride2_length))
        self.stride2_bias = nn.Parameter(torch.empty(self.stride2_length))
        nn.init.kaiming_uniform_(self.stride2_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.stride2_length)
        nn.init.uniform_(self.stride2_bias, -bound, bound)
        self.fuse = Sequential(
            Conv3d(2 * channels, channels, kernel_size=(1, 1, 1), bias=False),
            BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True),
            ReLU(inplace=True),
        )

    def _shared_stride2_linear(self, x):
        temporal_length = x.shape[2]
        y = x.permute(0, 1, 3, 4, 2)
        y = torch.nn.functional.linear(
            y,
            self.stride2_weight[:temporal_length, :temporal_length],
            self.stride2_bias[:temporal_length],
        )
        return y.permute(0, 1, 4, 2, 3)

    def forward(self, x):
        if x.shape[2] != self.num_frames:
            raise ValueError("Multi_TMixer input temporal length does not match num_frames")
        global_feature = self.global_linear(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        stride2_feature = torch.empty_like(x)
        stride2_feature[:, :, 0::2] = self._shared_stride2_linear(x[:, :, 0::2])
        if self.num_frames > 1:
            stride2_feature[:, :, 1::2] = self._shared_stride2_linear(x[:, :, 1::2])
        return self.fuse(torch.cat([global_feature, stride2_feature], dim=1))


class ChannelSE3D(Module):
    """Channel excitation pooled over time and kept pointwise in space."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.gate = Sequential(
            Conv3d(channels, hidden_channels, kernel_size=(1, 1, 1), bias=True),
            ReLU(inplace=True),
            Conv3d(hidden_channels, channels, kernel_size=(1, 1, 1), bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        temporal_descriptor = x.mean(dim=2, keepdim=True)
        return x * self.gate(temporal_descriptor)


class Multi_TConv(Module):
    """Residual multi-scale temporal block with temporal-pooled SE fusion."""

    def __init__(self, channels, kernels=(1, 3, 5, 7), add_channels=16):
        super().__init__()
        if not kernels:
            raise ValueError("Multi_TConv requires at least one temporal kernel")
        add_channels = channels
        expanded_channels = channels + add_channels

        self.channels = channels
        self.expanded_channels = expanded_channels
        self.num_groups = len(kernels)
        base_group_channels = expanded_channels // self.num_groups
        self.group_channels = [base_group_channels] * self.num_groups
        self.group_channels[0] += expanded_channels % self.num_groups
        self.expand = Conv3d(channels, expanded_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                group_channels,
                group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel, group_channels in zip(kernels, self.group_channels)
        )
        self.norm = BatchNorm3d(expanded_channels, eps=0.001, momentum=0.1, affine=True)
        self.branch_activation = ReLU(inplace=True)
        self.se = ChannelSE3D(expanded_channels, reduction=2)
        self.fuse = Conv3d(expanded_channels, channels, kernel_size=1, bias=False)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        batch_size, _, num_frames, height, width = x.shape
        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            self.expanded_channels,
            num_frames,
        )
        branch_features = [
            branch(group).reshape(
                batch_size, height, width, group_channels, num_frames
            ).permute(0, 3, 4, 1, 2)
            for branch, group, group_channels in zip(
                self.branches,
                torch.split(temporal_points, self.group_channels, dim=1),
                self.group_channels,
            )
        ]
        x = self.branch_activation(self.norm(torch.cat(branch_features, dim=1)))
        x = self.se(x)
        x = self.fuse(x)
        return self.activation(x + identity)


class Multi_TConv_LinearTT(Module):
    """Configurable local temporal pyramid followed by global Linear(T, T)."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if not kernels:
            raise ValueError("Multi_TConv_LinearTT requires at least one temporal kernel")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        base_group_channels = channels // self.num_groups
        self.group_channels = [base_group_channels] * self.num_groups
        self.group_channels[0] += channels % self.num_groups

        # Exchange channel information before fixed temporal-scale assignment.
        self.channel_shuffle = Conv3d(channels, channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                group_channels,
                group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            )
            for kernel, group_channels in zip(kernels, self.group_channels)
        )
        self.channel_fuse = Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True)
        self.activation = ReLU(inplace=True)
        self.linear = nn.Linear(num_frames, num_frames, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("Multi_TConv_LinearTT input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("Multi_TConv_LinearTT input temporal length does not match num_frames")

        x = self.channel_shuffle(x)
        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        branch_features = [
            branch(group).reshape(
                batch_size, height, width, group_channels, num_frames
            ).permute(0, 3, 4, 1, 2)
            for branch, group, group_channels in zip(
                self.branches,
                torch.split(temporal_points, self.group_channels, dim=1),
                self.group_channels,
            )
        ]
        x = self.channel_fuse(torch.cat(branch_features, dim=1).contiguous())
        x = self.activation(self.norm(x))
        return self.linear(x.permute(0, 1, 3, 4, 2)).permute(
            0, 1, 4, 2, 3
        ).contiguous()


class TMixer_GL(Module):
    """Fuse global Linear(T, T) mixing with grouped local temporal convolutions."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if channels % len(kernels) != 0:
            raise ValueError("TMixer_GL channels must be divisible by the number of temporal groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups

        # Global branch: share one dense temporal projection across all channels
        # and spatial positions.
        self.linear = nn.Linear(num_frames, num_frames, bias=True)

        # Mix all channels before assigning them to fixed temporal-scale groups.
        # self.local_shuffle = Conv3d(channels, channels, kernel_size=1, bias=True)

        # Local branch: fixed channel groups model different pulse widths.
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)

        # Fuse global pulse evidence and local pulse localization pointwise.
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL input temporal length does not match num_frames")

        global_feature = self.linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()
        local_input = x
        # local_input = self.local_shuffle(local_input)
        temporal_points = local_input.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_groups = temporal_points.chunk(self.num_groups, dim=1)
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(self.local_branches, local_groups)
        ]
        local_feature = self.local_fuse(
            torch.cat(local_features, dim=1).contiguous()
        )

        return self.out_fuse(torch.cat([global_feature, local_feature], dim=1))


class TMixer_GL_groupT(Module):
    """GL mixer with configurable local temporal kernel groups."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.kernels = tuple(kernels)
        self.num_groups = len(self.kernels)

        base_channels = channels // self.num_groups
        self.group_channels = [base_channels] * self.num_groups
        self.group_channels[0] += channels % self.num_groups

        # Global branch, identical to the current TMixer_GL.
        self.linear = nn.Linear(num_frames, num_frames, bias=True)

        # Local branch: the first group receives every remainder channel.
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                group_channels,
                group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for group_channels, kernel in zip(self.group_channels, self.kernels)
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL_groupT input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL_groupT input temporal length does not match num_frames")

        global_feature = self.linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group, group_channels in zip(
                self.local_branches,
                torch.split(temporal_points, self.group_channels, dim=1),
                self.group_channels,
            )
        ]
        local_feature = self.local_fuse(
            torch.cat(local_features, dim=1).contiguous()
        )
        return self.out_fuse(torch.cat([global_feature, local_feature], dim=1))


class TMixer_GL_group2(Module):
    """Full-channel global Linear(T,T) and frame-local Linear(1,1) mixer."""

    def __init__(self, channels, num_frames):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames

        self.global_linear = nn.Linear(num_frames, num_frames, bias=False)
        self.local_linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        _, channels, num_frames, _, _ = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL_group2 input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL_group2 input temporal length does not match num_frames")

        global_feature = self.global_linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()
        local_feature = self.local_linear(
            x.permute(0, 1, 3, 4, 2).unsqueeze(-1)
        ).squeeze(-1).permute(0, 1, 4, 2, 3).contiguous()
        return global_feature + local_feature


class TMixer_GLinear_Ldpconv(Module):
    """Global temporal Linear branch plus local depthwise spatial branch."""

    def __init__(self, channels, num_frames):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames

        # Preserve the successful affine global temporal projection.
        self.global_linear = nn.Linear(num_frames, num_frames, bias=False)
        # Spatially local, channel-preserving, and frame-preserving evidence.
        self.local_dpconv = Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        _, channels, num_frames, _, _ = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GLinear_Ldpconv input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GLinear_Ldpconv input temporal length does not match num_frames")

        global_feature = self.global_linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()
        return global_feature + self.local_dpconv(x)


class TMixer_GL_test(Module):
    """TMixer_GL without the pre-group channel shuffle in the local branch."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if channels % len(kernels) != 0:
            raise ValueError("TMixer_GL_test channels must be divisible by the number of temporal groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups

        self.linear = nn.Linear(num_frames, num_frames, bias=True)
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL_test input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL_test input temporal length does not match num_frames")

        global_feature = self.linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(
                self.local_branches,
                temporal_points.chunk(self.num_groups, dim=1),
            )
        ]
        local_feature = self.local_fuse(
            torch.cat(local_features, dim=1).contiguous()
        )
        return self.out_fuse(torch.cat([global_feature, local_feature], dim=1))


class TMixer_GL_Dilation1234(Module):
    """GL mixer whose local groups use 3-tap temporal dilations 1/2/3/4."""

    def __init__(self, channels, num_frames, dilations=(1, 2, 3, 4)):
        super().__init__()
        if channels % len(dilations) != 0:
            raise ValueError("TMixer_GL_Dilation1234 channels must be divisible by the number of temporal groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(dilations)
        self.group_channels = channels // self.num_groups

        self.linear = nn.Linear(num_frames, num_frames, bias=True)
        # No pre-group 1x1x1 channel shuffle, matching the current TMixer_GL.
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                bias=True,
            )
            for dilation in dilations
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL_Dilation1234 input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL_Dilation1234 input temporal length does not match num_frames")

        global_feature = self.linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(
                self.local_branches,
                temporal_points.chunk(self.num_groups, dim=1),
            )
        ]
        local_feature = self.local_fuse(
            torch.cat(local_features, dim=1).contiguous()
        )
        return self.out_fuse(torch.cat([global_feature, local_feature], dim=1))


class TMixer_GL_BNRelu(Module):
    """Independent GL mixer with output BatchNorm3d and ReLU."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if channels % len(kernels) != 0:
            raise ValueError("TMixer_GL_BNRelu channels must be divisible by the number of temporal groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups

        self.linear = nn.Linear(num_frames, num_frames, bias=True)
        # No pre-group 1x1x1 channel shuffle.
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)
        self.norm = BatchNorm3d(channels, eps=0.001, momentum=0.1, affine=True)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_GL_BNRelu input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_GL_BNRelu input temporal length does not match num_frames")

        global_feature = self.linear(
            x.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3).contiguous()

        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(
                self.local_branches,
                temporal_points.chunk(self.num_groups, dim=1),
            )
        ]
        local_feature = self.local_fuse(
            torch.cat(local_features, dim=1).contiguous()
        )
        fused = self.out_fuse(torch.cat([global_feature, local_feature], dim=1))
        return self.relu(self.norm(fused))


class TMixer_Lonly(Module):
    """Grouped multi-scale local temporal mixer without the global Linear branch."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if channels % len(kernels) != 0:
            raise ValueError("TMixer_Lonly channels must be divisible by the number of temporal groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups

        # No pre-group 1x1x1 channel shuffle: isolate the local branch of
        # TMixer_GL_test as a standalone ablation.
        self.local_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )
        self.local_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_Lonly input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_Lonly input temporal length does not match num_frames")

        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        local_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(
                self.local_branches,
                temporal_points.chunk(self.num_groups, dim=1),
            )
        ]
        return self.local_fuse(torch.cat(local_features, dim=1).contiguous())


class TMixer_STP(Module):
    """Fuse symmetric grouped spatial and temporal convolution pyramids."""

    def __init__(self, channels, num_frames, kernels=(1, 3, 5, 7)):
        super().__init__()
        if channels % len(kernels) != 0:
            raise ValueError("TMixer_STP channels must be divisible by the number of pyramid groups")

        self.channels = channels
        self.num_frames = num_frames
        self.num_groups = len(kernels)
        self.group_channels = channels // self.num_groups

        # Mix all input channels before assigning them to fixed scale groups.
        self.input_shuffle = Conv3d(channels, channels, kernel_size=1, bias=True)

        self.temporal_branches = nn.ModuleList(
            nn.Conv1d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )
        self.spatial_branches = nn.ModuleList(
            nn.Conv2d(
                self.group_channels,
                self.group_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=True,
            )
            for kernel in kernels
        )

        self.temporal_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.spatial_fuse = Conv3d(channels, channels, kernel_size=1, bias=True)
        self.out_fuse = Conv3d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if channels != self.channels:
            raise ValueError("TMixer_STP input channels do not match channels")
        if num_frames != self.num_frames:
            raise ValueError("TMixer_STP input temporal length does not match num_frames")

        x = self.input_shuffle(x)

        # [B, C, T, H, W] -> [B*H*W, C, T] for efficient Conv1d.
        temporal_points = x.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width,
            channels,
            num_frames,
        )
        temporal_features = [
            branch(group).reshape(
                batch_size,
                height,
                width,
                self.group_channels,
                num_frames,
            ).permute(0, 3, 4, 1, 2)
            for branch, group in zip(
                self.temporal_branches,
                temporal_points.chunk(self.num_groups, dim=1),
            )
        ]
        temporal_feature = self.temporal_fuse(
            torch.cat(temporal_features, dim=1).contiguous()
        )

        # [B, C, T, H, W] -> [B*T, C, H, W] for efficient Conv2d.
        spatial_frames = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )
        spatial_features = [
            branch(group).reshape(
                batch_size,
                num_frames,
                self.group_channels,
                height,
                width,
            ).permute(0, 2, 1, 3, 4)
            for branch, group in zip(
                self.spatial_branches,
                spatial_frames.chunk(self.num_groups, dim=1),
            )
        ]
        spatial_feature = self.spatial_fuse(
            torch.cat(spatial_features, dim=1).contiguous()
        )

        return self.out_fuse(torch.cat([temporal_feature, spatial_feature], dim=1))


class TMixer_attn(Module):
    """Temporal self-attention independently at each spatial location."""

    def __init__(self, dim, num_frames, num_heads=4, point_batch_size=8192):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.point_batch_size = point_batch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, num_frames, height, width = x.shape
        if num_frames != self.num_frames:
            raise ValueError("TMixer_attn input temporal length does not match num_frames")
        if channels != self.dim:
            raise ValueError("TMixer_attn input channels do not match dim")
        y = x.permute(0, 3, 4, 2, 1).contiguous().view(batch_size * height * width, num_frames, channels)
        y_pos = y + self.pos_embed
        out = torch.cat(
            [
                self.attn(query=y_pos_chunk, key=y_pos_chunk, value=y_chunk, need_weights=False)[0]
                for y_pos_chunk, y_chunk in zip(
                    y_pos.split(self.point_batch_size, dim=0),
                    y.split(self.point_batch_size, dim=0),
                )
            ],
            dim=0,
        )
        y = y - out
        return y.view(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1, 2).contiguous()


class TMixer_ca(Module):
    """Temporal cross-attention with a learned global temporal key projection."""

    def __init__(self, channels, num_frames):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.k_temporal = nn.Linear(num_frames, num_frames, bias=True)
        self.q_proj = Conv3d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = Conv3d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = Conv3d(channels, channels, kernel_size=1, bias=False)
        self.o_proj = Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        """Apply pointwise temporal cross-attention to x: [B, C, T, H, W]."""
        batch_size, channels, num_frames, height, width = x.shape
        if num_frames != self.num_frames:
            raise ValueError("TMixer_ca input temporal length does not match num_frames")
        if channels != self.channels:
            raise ValueError("TMixer_ca input channels do not match channels")

        # Project the complete temporal sequence at every spatial point into K.
        key = self.k_temporal(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        query = self.q_proj(x).permute(0, 3, 4, 2, 1).reshape(-1, num_frames, channels)
        key = self.k_proj(key).permute(0, 3, 4, 2, 1).reshape(-1, num_frames, channels)
        value = self.v_proj(x).permute(0, 3, 4, 2, 1).reshape(-1, num_frames, channels)

        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(channels)
        weight = torch.softmax(score, dim=-1)
        y = torch.bmm(weight, value)
        y = y.view(batch_size, height, width, num_frames, channels).permute(0, 4, 3, 1, 2).contiguous()
        return x + self.o_proj(y)


class GD(Module):
    """Grouped spatial dilated convolutions followed by channel interaction."""

    def __init__(self, channels, dilations=(1, 2, 3, 4)):
        super().__init__()
        if channels % len(dilations) != 0:
            raise ValueError("channels must be divisible by the number of GD dilation groups")
        branch_channels = channels // len(dilations)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            for dilation in dilations
        )
        self.fuse = Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        batch_size, _, num_frames, height, width = x.shape
        groups = x.chunk(len(self.branches), dim=1)
        outputs = []
        for branch, group in zip(self.branches, groups):
            group = group.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, group.shape[1], height, width)
            group = branch(group)
            outputs.append(group.reshape(batch_size, num_frames, group.shape[1], height, width).permute(0, 2, 1, 3, 4))
        return self.fuse(torch.cat(outputs, dim=1))


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
        Snack_repeat=1,
        TKernel=((5, 1, 1), (5, 1, 1), (5, 1, 1)),
        OffsetKernel=None,
        UNet3D_skip="snack",
        TZSConv_skip=None,
        VSnack_residual=1,
        Skip_TMixer=(0, 0, 0),
        TMixer_skip="TMixer",
        TMixer_num_heads=8,
        TMixer_groupT_kernels=(1, 3, 5, 7),
        GD_skip=(0, 0, 0),
        num_frames=5,
    ):
        super().__init__()
        if feat_channels is None:
            feat_channels = [16, 32, 64, 128]
        if Snack_skip is None:
            Snack_skip = [1, 1, 1]
        if Snack_max_offset is None:
            Snack_max_offset = [10, 10, 10]
        if OffsetKernel is None:
            OffsetKernel = TKernel
        if TZSConv_skip is None:
            TZSConv_skip = [1, 1, 1]
        if len(Snack_skip) != 3:
            raise ValueError("Snack_skip must contain three switches for [skip1, skip2, skip3]")
        if len(Snack_max_offset) != 3:
            raise ValueError("Snack_max_offset must contain three values for [skip1, skip2, skip3]")
        if len(TZSConv_skip) != 3:
            raise ValueError("TZSConv_skip must contain three switches for [skip1, skip2, skip3]")
        if len(Skip_TMixer) != 3:
            raise ValueError("Skip_TMixer must contain three switches for [skip1, skip2, skip3]")
        if len(GD_skip) != 3:
            raise ValueError("GD_skip must contain three switches for [skip1, skip2, skip3]")
        self.Snack_skip = [int(v) for v in Snack_skip]
        if any(v not in (0, 1) for v in self.Snack_skip):
            raise ValueError("Snack_skip values must be 0 or 1")
        self.TZSConv_skip = [int(v) for v in TZSConv_skip]
        if any(v not in (0, 1) for v in self.TZSConv_skip):
            raise ValueError("TZSConv_skip values must be 0 or 1")
        self.Skip_TMixer = [int(v) for v in Skip_TMixer]
        if any(v not in (0, 1) for v in self.Skip_TMixer):
            raise ValueError("Skip_TMixer values must be 0 or 1")
        if TMixer_skip not in ("TMixer", "TMixer_attn", "TMixer_ca", "Multi_TMixer", "Multi_TConv", "Multi_TConv_LinearTT", "TMixer_GL", "TMixer_GLinear_Ldpconv", "TMixer_GL_group2", "TMixer_GL_groupT", "TMixer_GL_Dilation1234", "TMixer_GL_BNRelu", "TMixer_GL_test", "TMixer_Lonly", "TMixer_STP"):
            raise ValueError("TMixer_skip must be \"TMixer\", \"TMixer_attn\", \"TMixer_ca\", \"Multi_TMixer\", \"Multi_TConv\", \"TMixer_GL\", \"TMixer_GLinear_Ldpconv\", \"TMixer_GL_group2\", \"TMixer_GL_groupT\", \"TMixer_GL_Dilation1234\", \"TMixer_GL_BNRelu\", \"TMixer_GL_test\", \"TMixer_Lonly\", or \"TMixer_STP\"")
        self.TMixer_skip = TMixer_skip
        if TMixer_num_heads <= 0:
            raise ValueError("TMixer_num_heads must be positive")
        self.TMixer_num_heads = TMixer_num_heads
        self.TMixer_groupT_kernels = TMixer_groupT_kernels
        self.GD_skip = [int(v) for v in GD_skip]
        if any(v not in (0, 1) for v in self.GD_skip):
            raise ValueError("GD_skip values must be 0 or 1")
        self.Snack_max_offset = [float(v) for v in Snack_max_offset]
        self.TKernel = TKernel
        self.OffsetKernel = OffsetKernel
        if int(VSnack_residual) not in (0, 1):
            raise ValueError("VSnack_residual must be 0 or 1")
        self.VSnack_residual = bool(int(VSnack_residual))
        if UNet3D_skip not in ("snack", "snack_unrest", "v_snack", "tconv", "tdcn"):
            raise ValueError("UNet3D_skip must be 'snack', 'snack_unrest', 'v_snack', 'tconv', or 'tdcn'")
        self.UNet3D_skip = UNet3D_skip
        self.Snack_repeat = Snack_repeat

        def build_nsack(channels, index):
            if self.UNet3D_skip == "snack":
                return TemporalSnakeConv3D_Block(channels, kernel=self.TKernel[index], offset_kernel=self.OffsetKernel[index], extend_scope=self.Snack_max_offset[index])
            if self.UNet3D_skip == "snack_unrest":
                return TemporalSnakeUnrestConv3D_Block(channels, kernel=self.TKernel[index], offset_kernel=self.OffsetKernel[index], extend_scope=self.Snack_max_offset[index])
            if self.UNet3D_skip == "v_snack":
                return TemporalVelocitySnakeConv3D_Block(channels, kernel=self.TKernel[index], offset_kernel=self.OffsetKernel[index], extend_scope=self.Snack_max_offset[index], use_residual=self.VSnack_residual)
            if self.UNet3D_skip == "tconv":
                return TemporalConv3D_Block(channels, kernel=self.TKernel[index])
            if self.UNet3D_skip == "tdcn":
                return TemporalDCN3D_Block(channels, kernel=self.TKernel[index], offset_kernel=self.OffsetKernel[index])

        self.nsack1 = nn.ModuleList()
        self.nsack2 = nn.ModuleList()
        self.nsack3 = nn.ModuleList()
        for _ in range(self.Snack_repeat - 1):
            if self.Snack_skip[0]:
                self.nsack1.append(build_nsack(feat_channels[0], 0))
            if self.Snack_skip[1]:
                self.nsack2.append(build_nsack(feat_channels[1], 1))
            if self.Snack_skip[2] and len(feat_channels) >= 4:
                self.nsack3.append(build_nsack(feat_channels[2], 2))

        temporal_sizes = [num_frames]
        for _ in range(2):
            if T_pooling:
                temporal_sizes.append((temporal_sizes[-1] + 1) // 2 if downsample_mode == "stride" else temporal_sizes[-1] // 2)
            else:
                temporal_sizes.append(temporal_sizes[-1])

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
                self.skip1 = TemporalSnakeConv3D_Block(feat_channels[0], kernel=self.TKernel[0], offset_kernel=self.OffsetKernel[0], extend_scope=self.Snack_max_offset[0])
            if self.UNet3D_skip == "snack_unrest":
                self.skip1 = TemporalSnakeUnrestConv3D_Block(feat_channels[0], kernel=self.TKernel[0], offset_kernel=self.OffsetKernel[0], extend_scope=self.Snack_max_offset[0])
            if self.UNet3D_skip == "v_snack":
                self.skip1 = TemporalVelocitySnakeConv3D_Block(feat_channels[0], kernel=self.TKernel[0], offset_kernel=self.OffsetKernel[0], extend_scope=self.Snack_max_offset[0], use_residual=self.VSnack_residual)
            if self.UNet3D_skip == "tconv":
                self.skip1 = TemporalConv3D_Block(feat_channels[0], kernel=self.TKernel[0])
            if self.UNet3D_skip == "tdcn":
                self.skip1 = TemporalDCN3D_Block(feat_channels[0], kernel=self.TKernel[0], offset_kernel=self.OffsetKernel[0])
        if self.Snack_skip[1]:
            if self.UNet3D_skip == "snack":
                self.skip2 = TemporalSnakeConv3D_Block(feat_channels[1], kernel=self.TKernel[1], offset_kernel=self.OffsetKernel[1], extend_scope=self.Snack_max_offset[1])
            if self.UNet3D_skip == "snack_unrest":
                self.skip2 = TemporalSnakeUnrestConv3D_Block(feat_channels[1], kernel=self.TKernel[1], offset_kernel=self.OffsetKernel[1], extend_scope=self.Snack_max_offset[1])
            if self.UNet3D_skip == "v_snack":
                self.skip2 = TemporalVelocitySnakeConv3D_Block(feat_channels[1], kernel=self.TKernel[1], offset_kernel=self.OffsetKernel[1], extend_scope=self.Snack_max_offset[1], use_residual=self.VSnack_residual)
            if self.UNet3D_skip == "tconv":
                self.skip2 = TemporalConv3D_Block(feat_channels[1], kernel=self.TKernel[1])
            if self.UNet3D_skip == "tdcn":
                self.skip2 = TemporalDCN3D_Block(feat_channels[1], kernel=self.TKernel[1], offset_kernel=self.OffsetKernel[1])
        if self.Snack_skip[2] and len(self.feat_channels) >= 4:
            if self.UNet3D_skip == "snack":
                self.skip3 = TemporalSnakeConv3D_Block(feat_channels[2], kernel=self.TKernel[2], offset_kernel=self.OffsetKernel[2], extend_scope=self.Snack_max_offset[2])
            if self.UNet3D_skip == "snack_unrest":
                self.skip3 = TemporalSnakeUnrestConv3D_Block(feat_channels[2], kernel=self.TKernel[2], offset_kernel=self.OffsetKernel[2], extend_scope=self.Snack_max_offset[2])
            if self.UNet3D_skip == "v_snack":
                self.skip3 = TemporalVelocitySnakeConv3D_Block(feat_channels[2], kernel=self.TKernel[2], offset_kernel=self.OffsetKernel[2], extend_scope=self.Snack_max_offset[2], use_residual=self.VSnack_residual)
            if self.UNet3D_skip == "tconv":
                self.skip3 = TemporalConv3D_Block(feat_channels[2], kernel=self.TKernel[2])
            if self.UNet3D_skip == "tdcn":
                self.skip3 = TemporalDCN3D_Block(feat_channels[2], kernel=self.TKernel[2], offset_kernel=self.OffsetKernel[2])

        if self.TMixer_skip == "TMixer":
            tmixer_block = lambda channels, temporal_size: TMixer(channels, temporal_size, self.TMixer_num_heads)
        elif self.TMixer_skip == "TMixer_attn":
            tmixer_block = lambda channels, temporal_size: TMixer_attn(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_ca":
            tmixer_block = lambda channels, temporal_size: TMixer_ca(channels, temporal_size)
        elif self.TMixer_skip == "Multi_TMixer":
            tmixer_block = lambda channels, temporal_size: Multi_TMixer(channels, temporal_size)
        elif self.TMixer_skip == "Multi_TConv":
            tmixer_block = lambda channels, temporal_size: Multi_TConv(
                channels, self.TMixer_groupT_kernels
            )
        elif self.TMixer_skip == "Multi_TConv_LinearTT":
            tmixer_block = lambda channels, temporal_size: Multi_TConv_LinearTT(
                channels, temporal_size, self.TMixer_groupT_kernels
            )
        elif self.TMixer_skip == "TMixer_GL":
            tmixer_block = lambda channels, temporal_size: TMixer_GL(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_GLinear_Ldpconv":
            tmixer_block = lambda channels, temporal_size: TMixer_GLinear_Ldpconv(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_GL_group2":
            tmixer_block = lambda channels, temporal_size: TMixer_GL_group2(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_GL_groupT":
            tmixer_block = lambda channels, temporal_size: TMixer_GL_groupT(channels, temporal_size, self.TMixer_groupT_kernels)
        elif self.TMixer_skip == "TMixer_GL_Dilation1234":
            tmixer_block = lambda channels, temporal_size: TMixer_GL_Dilation1234(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_GL_BNRelu":
            tmixer_block = lambda channels, temporal_size: TMixer_GL_BNRelu(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_GL_test":
            tmixer_block = lambda channels, temporal_size: TMixer_GL_test(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_Lonly":
            tmixer_block = lambda channels, temporal_size: TMixer_Lonly(channels, temporal_size)
        elif self.TMixer_skip == "TMixer_STP":
            tmixer_block = lambda channels, temporal_size: TMixer_STP(channels, temporal_size)

        if self.Skip_TMixer[0]:
            self.tmixer_skip1 = tmixer_block(feat_channels[0], temporal_sizes[0])
        if self.Skip_TMixer[1]:
            self.tmixer_skip2 = tmixer_block(feat_channels[1], temporal_sizes[1])
        if self.Skip_TMixer[2] and len(self.feat_channels) >= 4:
            self.tmixer_skip3 = tmixer_block(feat_channels[2], temporal_sizes[2])

        if self.GD_skip[0]:
            self.gd_skip1 = GD(feat_channels[0])
        if self.GD_skip[1]:
            self.gd_skip2 = GD(feat_channels[1])
        if self.GD_skip[2] and len(self.feat_channels) >= 4:
            self.gd_skip3 = GD(feat_channels[2])

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
            for nsack in self.nsack3:
                enc3 = nsack(enc3)
            if self.Skip_TMixer[2]:
                enc3 = self.tmixer_skip3(enc3)
            if self.GD_skip[2]:
                enc3 = self.gd_skip3(enc3)
            tmp = torch.cat([tmp, enc3], dim=1)
            tmp = self.dec_conv3(tmp)

            tmp = self.upsample2(tmp)
            if self.TZSConv_skip[1]:
                enc2 = self.tzs_skip2(enc2)
            if self.Snack_skip[1]:
                enc2 = self.skip2(enc2)
            for nsack in self.nsack2:
                enc2 = nsack(enc2)
            if self.Skip_TMixer[1]:
                enc2 = self.tmixer_skip2(enc2)
            if self.GD_skip[1]:
                enc2 = self.gd_skip2(enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            if self.TZSConv_skip[0]:
                enc1 = self.tzs_skip1(enc1)
            if self.Snack_skip[0]:
                enc1 = self.skip1(enc1)
            for nsack in self.nsack1:
                enc1 = nsack(enc1)
            if self.Skip_TMixer[0]:
                enc1 = self.tmixer_skip1(enc1)
            if self.GD_skip[0]:
                enc1 = self.gd_skip1(enc1)
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
            for nsack in self.nsack2:
                enc2 = nsack(enc2)
            if self.Skip_TMixer[1]:
                enc2 = self.tmixer_skip2(enc2)
            if self.GD_skip[1]:
                enc2 = self.gd_skip2(enc2)
            tmp = torch.cat([tmp, enc2], dim=1)
            tmp = self.dec_conv2(tmp)

            tmp = self.upsample1(tmp)
            if self.TZSConv_skip[0]:
                enc1 = self.tzs_skip1(enc1)
            if self.Snack_skip[0]:
                enc1 = self.skip1(enc1)
            for nsack in self.nsack1:
                enc1 = nsack(enc1)
            if self.Skip_TMixer[0]:
                enc1 = self.tmixer_skip1(enc1)
            if self.GD_skip[0]:
                enc1 = self.gd_skip1(enc1)
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
        Snack_repeat=1,
        TKernel=((5, 1, 1), (5, 1, 1), (5, 1, 1)),
        OffsetKernel=None,
        UNet3D_skip="snack",
        TZSConv_skip=None,
        VSnack_residual=1,
        Skip_TMixer=(0, 0, 0),
        TMixer_skip="TMixer",
        TMixer_num_heads=8,
        TMixer_groupT_kernels=(1, 3, 5, 7),
        GD_skip=(0, 0, 0),
        num_frames=5,
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
            Snack_repeat=Snack_repeat,
            TKernel=TKernel,
            OffsetKernel=OffsetKernel,
            UNet3D_skip=UNet3D_skip,
            TZSConv_skip=TZSConv_skip,
            VSnack_residual=VSnack_residual,
            Skip_TMixer=Skip_TMixer,
            TMixer_skip=TMixer_skip,
            TMixer_num_heads=TMixer_num_heads,
            TMixer_groupT_kernels=TMixer_groupT_kernels,
            GD_skip=GD_skip,
            num_frames=num_frames,
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
