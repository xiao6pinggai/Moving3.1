import torch
import torch.nn.functional as F
from torch import nn


def _to_tuple(value, name="argument"):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"{name} should be int or a tuple/list of length 3, got {value}")


class DynamicTOSConv(nn.Module):
    """Dynamic temporal softmax convolution for per-frame background estimation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        skip_add=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple(kernel_size, name="kernel_size")
        self.stride = _to_tuple(stride, name="stride")
        self.padding = _to_tuple(padding, name="padding")
        self.dilation = _to_tuple(dilation, name="dilation")
        self.groups = groups
        self.skip_add = skip_add

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if self.kernel_size[1:] != (1, 1):
            raise ValueError("DynamicTOSConv only supports spatial kernel size 1x1")
        if self.stride != (1, 1, 1):
            raise ValueError("DynamicTOSConv only supports stride=1")
        if self.padding != (0, 0, 0):
            raise ValueError("DynamicTOSConv only supports padding=0")
        if self.dilation != (1, 1, 1):
            raise ValueError("DynamicTOSConv only supports dilation=1")
        if self.kernel_size[0] <= 1:
            raise ValueError("DynamicTOSConv requires at least 2 frames to exclude the current frame")

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        self.temporal_size = self.kernel_size[0]

        logits_channels = (
            out_channels
            * self.in_channels_per_group
            * self.temporal_size
            * self.temporal_size
        )
        self.logit_conv = nn.Conv3d(
            in_channels,
            logits_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

        keep_mask = (torch.ones(self.temporal_size, self.temporal_size) - torch.eye(self.temporal_size)).bool()
        self.register_buffer("temporal_keep_mask", keep_mask, persistent=False)

        if skip_add:
            need_proj = in_channels != out_channels
            self.skip_proj = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ) if need_proj else None
        else:
            self.skip_proj = None

        nn.init.zeros_(self.logit_conv.weight)
        if self.logit_conv.bias is not None:
            nn.init.zeros_(self.logit_conv.bias)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"DynamicTOSConv expects a 5D tensor, got shape {tuple(x.shape)}")

        n, _, t, h, w = x.shape
        if t != self.temporal_size:
            raise ValueError(
                f"DynamicTOSConv expects {self.temporal_size} frames, but received {t}"
            )

        logits = self.logit_conv(x)
        logits = logits.view(
            n,
            self.groups,
            self.out_channels_per_group,
            self.in_channels_per_group,
            t,
            t,
            h,
            w,
        )

        keep_mask = self.temporal_keep_mask.view(1, 1, 1, 1, t, t, 1, 1)
        logits = logits.masked_fill(~keep_mask, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=5)

        x_grouped = x.view(n, self.groups, self.in_channels_per_group, t, h, w)
        out = torch.einsum('ngoitshw,ngishw->ngothw', weights, x_grouped)
        out = out.reshape(n, self.out_channels, t, h, w)

        if self.skip_add:
            identity = self.skip_proj(x) if self.skip_proj is not None else x
            out = identity - out

        return out

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.logit_conv.bias is not None}, "
            f"skip_add={self.skip_add}"
        )
