import torch
from torch import nn


def _to_tuple(value, name="argument"):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"{name} should be int or a tuple/list of length 3, got {value}")


class TMF(nn.Module):
    """Per-channel per-location temporal median background estimator."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
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

        if in_channels != out_channels:
            raise ValueError("TMF expects in_channels == out_channels")
        if self.kernel_size[1:] != (1, 1):
            raise ValueError("TMF only supports spatial kernel size 1x1")
        if self.stride != (1, 1, 1):
            raise ValueError("TMF only supports stride=1")
        if self.padding != (0, 0, 0):
            raise ValueError("TMF only supports padding=0")
        if self.dilation != (1, 1, 1):
            raise ValueError("TMF only supports dilation=1")
        if self.kernel_size[0] <= 1:
            raise ValueError("TMF requires at least 2 frames for temporal median filtering")

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"TMF expects a 5D tensor, got shape {tuple(x.shape)}")

        _, _, t, _, _ = x.shape
        temporal_size = self.kernel_size[0]
        if t != temporal_size:
            raise ValueError(f"TMF expects {temporal_size} frames, but received {t}")

        background = x.median(dim=2, keepdim=True).values
        out = background.expand(-1, -1, t, -1, -1)
        if self.skip_add:
            out = x - out
        return out

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, skip_add={self.skip_add}"
        )
