import torch
import torch.nn.functional as F
from torch import nn


def _to_tuple(value, name="argument"):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"{name} should be int or a tuple/list of length 3, got {value}")

class TOSConv(nn.Module):
    """Temporal sum-one convolution for per-frame background estimation.

    For each target frame, the module estimates its background from all other
    frames in the same clip. The temporal weights are non-negative, sum to 1,
    and exclude the current frame via a masked renormalization step.
    """

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
            raise ValueError("TOSConv only supports spatial kernel size 1x1")
        if self.stride != (1, 1, 1):
            raise ValueError("TOSConv only supports stride=1")
        if self.padding != (0, 0, 0):
            raise ValueError("TOSConv only supports padding=0")
        if self.dilation != (1, 1, 1):
            raise ValueError("TOSConv only supports dilation=1")
        if self.kernel_size[0] <= 1:
            raise ValueError("TOSConv requires at least 2 frames to exclude the current frame")

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        temporal_size = self.kernel_size[0]

        # Modified:
        # old shape: [out_channels, in_channels_per_group, T]
        # new shape: [out_channels, in_channels_per_group, T_target, T_source]
        # Each target frame has an independent set of source-frame logits.
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                self.in_channels_per_group,
                temporal_size,
                temporal_size,
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        mask = torch.ones(temporal_size, temporal_size) - torch.eye(temporal_size)
        self.register_buffer("temporal_exclusion_mask", mask, persistent=False)

        if skip_add:
            need_proj = in_channels != out_channels
            self.skip_proj = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ) if need_proj else None
        else:
            self.skip_proj = None

        # Zero logits become a uniform distribution for each target frame;
        # after masking the diagonal, this initializes each frame's background
        # as the mean of the other frames.
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"TOSConv expects a 5D tensor, got shape {tuple(x.shape)}")

        n, _, t, h, w = x.shape
        temporal_size = self.kernel_size[0]
        if t != temporal_size:
            raise ValueError(
                f"TOSConv expects {temporal_size} frames, but received {t}"
            )

        # Modified:
        # self.weight shape: [O, I_g, T_target, T_source]
        # Softmax over source frames for each target frame independently.
        base_weights = F.softmax(self.weight, dim=3)

        masked_weights = (
            base_weights
            * self.temporal_exclusion_mask.unsqueeze(0).unsqueeze(0)
        )

        denom = masked_weights.sum(dim=3, keepdim=True).clamp_min(
            torch.finfo(masked_weights.dtype).eps
        )

        normalized_weights = masked_weights / denom

        x_grouped = x.view(n, self.groups, self.in_channels_per_group, t, h, w)

        grouped_weights = normalized_weights.view(
            self.groups,
            self.out_channels_per_group,
            self.in_channels_per_group,
            t,
            t,
        )

        out = torch.einsum('goits,ngishw->ngothw', grouped_weights, x_grouped)
        out = out.reshape(n, self.out_channels, t, h, w)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        if self.skip_add:
            identity = self.skip_proj(x) if self.skip_proj is not None else x
            out = identity - out

        return out

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"skip_add={self.skip_add}"
        )
