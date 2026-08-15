from math import prod
from typing import Optional, Tuple, Union

import os
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

_TVDCN_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tvdcn-src")
if _TVDCN_SRC not in sys.path:
    sys.path.insert(0, _TVDCN_SRC)

from tvdcn import ops


_Size = Union[int, Tuple[int, ...]]


def _as_tuple(value: _Size, ndim: int) -> Tuple[int, ...]:
    """Convert an int or tuple into the tuple format required by ConvNd."""
    # 将 int 或 tuple 统一转换为 Conv1d/2d/3d 需要的 tuple 格式。
    if ndim == 1:
        return _single(value)
    if ndim == 2:
        return _pair(value)
    if ndim == 3:
        return _triple(value)
    raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")


def _same_padding(kernel_size: Tuple[int, ...], dilation: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute symmetric padding for padding='same'."""
    # 计算 padding='same' 时的对称 padding。非对称情况请显式传入 padding tuple。
    padding = []
    for k, d in zip(kernel_size, dilation):
        effective = d * (k - 1)
        if effective % 2 != 0:
            raise ValueError(
                "padding='same' requires odd effective kernel sizes for this wrapper. "
                "Use an explicit padding tuple for asymmetric cases."
            )
        padding.append(effective // 2)
    return tuple(padding)


def _conv_module(ndim: int):
    """Return the standard ConvNd module used to predict offsets/masks."""
    # 返回用于内部预测 offset/mask 的常规卷积层类型。
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[ndim]


def _deform_op(ndim: int):
    """Return tvdcn's deformable convolution operator for the requested dimension."""
    # 返回 tvdcn 对应维度的可变形卷积算子。
    return {
        1: ops.deform_conv1d,
        2: ops.deform_conv2d,
        3: ops.deform_conv3d,
    }[ndim]


class DeformConvNd(nn.Module):
    """
    ConvNd-like wrapper around tvdcn deformable convolution.

    中文说明：
    - 该类是 tvdcn deform_conv1d/2d/3d 的统一封装，构造接口尽量对齐
      torch.nn.Conv1d/2d/3d。
    - 这里的 1D/2D/3D 指输入张量和算子的维度，而不是卷积核中非 1 维度的个数。
      例如视频/时空特征通常组织为 N,C,T,H,W，此时应该使用 DeformConv3d。
    - 如果只想做时域线核，应使用 DeformConv3d(kernel_size=(k, 1, 1))，其中
      PyTorch Conv3d 的维度顺序是 D,H,W；把 T 放在 D 维时，D 就是时间维。
    - 如果只想做空间平面核，可使用 DeformConv3d(kernel_size=(1, k, k))。
    - 如果 forward() 没有传入 offset，且 auto_offset=True，本类会用内部 offset
      generator 自动预测 offset。offset generator 零初始化，因此初始状态接近普通卷积。
    - 如果你后续要用轨迹网络单独建模运动小目标的时空曲线，可以在 forward()
      中显式传入 offset，此时内部 offset generator 不会被使用。

    English summary:
    The public constructor follows torch.nn.Conv1d/2d/3d closely. By default,
    offset is predicted internally from the input and initialized to zero, so
    the layer starts from the behavior of a regular convolution. External
    offset and mask can still be passed to forward() for trajectory-driven use.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        offset_groups: int = 1,
        modulated: bool = False,
        mask_groups: int = 1,
        offset_bias: bool = True,
        mask_bias: bool = True,
        auto_offset: bool = True,
        offset_kernel_size: Optional[_Size] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        # 参数检查与 nn.ConvNd 基本一致，同时额外检查 offset/mask 分组。
        if ndim not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if in_channels % offset_groups != 0:
            raise ValueError("in_channels must be divisible by offset_groups")
        if out_channels % offset_groups != 0:
            raise ValueError("out_channels must be divisible by offset_groups")
        if in_channels % mask_groups != 0:
            raise ValueError("in_channels must be divisible by mask_groups")
        if out_channels % mask_groups != 0:
            raise ValueError("out_channels must be divisible by mask_groups")
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError("padding_mode must be zeros, reflect, replicate, or circular")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_tuple(kernel_size, ndim)
        self.stride = _as_tuple(stride, ndim)
        self.dilation = _as_tuple(dilation, ndim)
        if isinstance(padding, str):
            if padding == "valid":
                self.padding = (0,) * ndim
            elif padding == "same":
                self.padding = _same_padding(self.kernel_size, self.dilation)
            else:
                raise ValueError("padding must be an int, tuple, 'same', or 'valid'")
        else:
            self.padding = _as_tuple(padding, ndim)
        self.groups = groups
        self.padding_mode = padding_mode
        self.offset_groups = offset_groups
        self.modulated = modulated
        self.mask_groups = mask_groups
        self.auto_offset = auto_offset
        if offset_kernel_size is None:
            self.offset_kernel_size = self.kernel_size
            self.offset_padding = self.padding
        else:
            self.offset_kernel_size = _as_tuple(offset_kernel_size, ndim)
            self.offset_padding = _same_padding(self.offset_kernel_size, self.dilation)

        # 主卷积权重，形状与标准 ConvNd 一致。
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        conv = _conv_module(ndim)
        k_total = prod(self.kernel_size)
        offset_channels = ndim * offset_groups * k_total
        if auto_offset:
            # 内部 offset 预测头。输出通道数为 ndim * offset_groups * kernel_elements。
            # 对 3D 输入而言，offset 每个采样点包含 (d, h, w) 三个方向。
            self.offset_generator = conv(
                in_channels,
                offset_channels,
                kernel_size=self.offset_kernel_size,
                stride=self.stride,
                padding=self.offset_padding,
                dilation=self.dilation,
                groups=offset_groups,
                bias=offset_bias,
                **factory_kwargs,
            )
        else:
            self.offset_generator = None

        if modulated:
            # 可选 modulation mask，对应 DCNv2 风格的幅值调制。
            self.mask_generator = conv(
                in_channels,
                mask_groups * k_total,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=mask_groups,
                bias=mask_bias,
                **factory_kwargs,
            )
        else:
            self.mask_generator = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 主卷积按标准 ConvNd 初始化；offset/mask 预测头置零，让初始行为接近普通卷积。
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels // self.groups * prod(self.kernel_size)
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)
        if self.offset_generator is not None:
            nn.init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                nn.init.zeros_(self.offset_generator.bias)
        if self.mask_generator is not None:
            nn.init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                nn.init.zeros_(self.mask_generator.bias)

    @property
    def _reversed_padding_repeated_twice(self) -> Tuple[int, ...]:
        # F.pad 使用反向展开格式，例如 3D 为 (w0,w1,h0,h1,d0,d1)。
        return tuple(x for p in reversed(self.padding) for x in (p, p))

    def forward(
        self,
        x: Tensor,
        offset: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run deformable convolution.

        中文说明：
        - x: 输入特征。1D 为 N,C,L；2D 为 N,C,H,W；3D 为 N,C,D,H,W。
        - offset: 外部 offset。若为 None 且 auto_offset=True，则内部自动生成。
        - mask: 外部 modulation mask。若为 None 且 modulated=True，则内部自动生成。
        """
        if offset is None and self.offset_generator is not None:
            offset = self.offset_generator(x)
        if mask is None and self.mask_generator is not None:
            mask = torch.sigmoid(self.mask_generator(x))

        input_for_conv = x
        padding = self.padding
        if self.padding_mode != "zeros":
            input_for_conv = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,) * self.ndim

        return _deform_op(self.ndim)(
            input_for_conv,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        fields = [
            f"{self.in_channels}",
            f"{self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
        ]
        if self.padding != (0,) * self.ndim:
            fields.append(f"padding={self.padding}")
        if self.dilation != (1,) * self.ndim:
            fields.append(f"dilation={self.dilation}")
        if self.groups != 1:
            fields.append(f"groups={self.groups}")
        if self.bias is None:
            fields.append("bias=False")
        if self.padding_mode != "zeros":
            fields.append(f"padding_mode={self.padding_mode}")
        if self.offset_groups != 1:
            fields.append(f"offset_groups={self.offset_groups}")
        if self.modulated:
            fields.append("modulated=True")
        if not self.auto_offset:
            fields.append("auto_offset=False")
        return ", ".join(fields)


class DeformConv1d(DeformConvNd):
    """
    1D deformable convolution for N,C,L tensors.

    中文说明：用于真正的一维输入，例如序列特征 N,C,L。它不是用来处理
    N,C,T,H,W 视频特征中的时域线核；视频时域线核仍应使用 DeformConv3d。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            1, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, **kwargs
        )


class DeformConv2d(DeformConvNd):
    """
    2D deformable convolution for N,C,H,W tensors.

    中文说明：用于二维图像/特征图。如果输入已经是 N,C,T,H,W，则即使卷积核
    只覆盖 H,W 平面，也建议使用 DeformConv3d(kernel_size=(1,k,k))。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            2, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, **kwargs
        )


class DeformConv3d(DeformConvNd):
    """
    3D deformable convolution for N,C,D,H,W tensors.

    中文说明：
    - 用于 3D 体数据或视频/时空特征。
    - 若输入是 N,C,T,H,W，请把 T 视为 PyTorch Conv3d 的 D 维。
    - 时域线核：kernel_size=(k,1,1)，padding=(k//2,0,0)。
    - 空间平面核：kernel_size=(1,k,k)，padding=(0,k//2,k//2)。
    - 完整时空核：kernel_size=(kt,kh,kw)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        **kwargs,
    ) -> None:
        super().__init__(
            3, in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, **kwargs
        )


class SnakeDeformConv3d(DeformConv3d):
    """
    Temporal dynamic snake deformable convolution for N,C,D,H,W tensors.

    中文说明：
    - 该层只实现时域蛇形约束，要求 kernel_size=(k,1,1)，其中 D 维对应时间 T。
    - 时间轴位置由标准 Conv3d 的 D 方向核位置给出；可学习偏移只作用在 H,W
      两个正交方向，并按 DSConv 的中心向两侧递推公式累积。
    - raw offset 经过 BatchNorm3d 和 tanh 限制到 [-1,1]，再乘 extend_scope。
    - 输出 offset 会被组装成 tvdcn deform_conv3d 需要的
      N, 3 * offset_groups * k, out_d, out_h, out_w 格式，其中 dz 恒为 0。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        offset_kernel_size: Optional[_Size] = None,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        offset_groups: int = 1,
        modulated: bool = False,
        mask_groups: int = 1,
        offset_bias: bool = True,
        mask_bias: bool = True,
        extend_scope: float = 1.0,
        mask_scale: float = 2.0,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        if kernel_size_[1:] != (1, 1):
            raise ValueError("SnakeDeformConv3d only supports temporal kernels: kernel_size=(k, 1, 1)")
        if kernel_size_[0] % 2 == 0:
            raise ValueError("SnakeDeformConv3d requires an odd temporal kernel size")
        if extend_scope <= 0:
            raise ValueError("extend_scope must be positive")
        if mask_scale <= 0:
            raise ValueError("mask_scale must be positive")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            offset_groups=offset_groups,
            modulated=False,
            auto_offset=False,
            device=device,
            dtype=dtype,
        )

        if in_channels % mask_groups != 0:
            raise ValueError("in_channels must be divisible by mask_groups")
        if out_channels % mask_groups != 0:
            raise ValueError("out_channels must be divisible by mask_groups")

        factory_kwargs = {"device": device, "dtype": dtype}
        if offset_kernel_size is None:
            self.offset_kernel_size = self.kernel_size
            self.offset_padding = self.padding
        else:
            self.offset_kernel_size = _triple(offset_kernel_size)
            self.offset_padding = _same_padding(self.offset_kernel_size, self.dilation)

        factory_kwargs = {"device": device, "dtype": dtype}
        temporal_kernel = self.kernel_size[0]
        offset_channels = 2 * offset_groups * temporal_kernel
        self.offset_generator = nn.Conv3d(
            in_channels,
            offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=offset_groups,
            bias=offset_bias,
            **factory_kwargs,
        )
        # self.offset_generator = nn.Conv3d(
        #             in_channels,
        #             offset_channels,
        #             kernel_size=3, # 保证5*5*5卷积核
        #             stride=1,
        #             padding=1,
        #             dilation=self.dilation,
        #             groups=offset_groups,
        #             bias=offset_bias,
        #             **factory_kwargs,
        #         )
        self.offset_norm = nn.BatchNorm3d(offset_channels, **factory_kwargs)

        self.modulated = modulated
        self.mask_groups = mask_groups
        self.extend_scope = float(extend_scope)
        self.mask_scale = float(mask_scale)
        self.auto_offset = True

        if modulated:
            self.mask_generator = nn.Conv3d(
                in_channels,
                mask_groups * temporal_kernel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=mask_groups,
                bias=mask_bias,
                **factory_kwargs,
            )
        else:
            self.mask_generator = None

        self.reset_parameters()

    @staticmethod
    def _accumulate_from_center(delta: Tensor) -> Tensor:
        center = delta.size(2) // 2
        zero = torch.zeros_like(delta[:, :, center])

        lower_offsets = []
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center - step]
            lower_offsets.append(acc)
        lower_offsets.reverse()

        upper_offsets = [zero]
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center + step]
            upper_offsets.append(acc)

        return torch.stack(lower_offsets + upper_offsets, dim=2)

    def _make_temporal_snake_offset(self, x: Tensor) -> Tensor:
        raw_offset = self.offset_generator(x)
        raw_offset = torch.tanh(self.offset_norm(raw_offset)) * self.extend_scope # 这里不符合DCN的默认设置，默认是直接传回raw而不做任何约束
        # raw_offset = torch.tanh(raw_offset) * self.extend_scope # 这里不符合DCN的默认设置，默认是直接传回raw而不做任何约束

        n, _, out_d, out_h, out_w = raw_offset.shape
        temporal_kernel = self.kernel_size[0]
        raw_offset = raw_offset.view(
            n,
            self.offset_groups,
            temporal_kernel,
            2,
            out_d,
            out_h,
            out_w,
        )

        delta_h = raw_offset[:, :, :, 0]
        delta_w = raw_offset[:, :, :, 1]
        snake_h = self._accumulate_from_center(delta_h)
        snake_w = self._accumulate_from_center(delta_w)

        offset = raw_offset.new_zeros(
            n,
            self.offset_groups,
            temporal_kernel,
            1,
            1,
            3,
            out_d,
            out_h,
            out_w,
        )
        offset[:, :, :, 0, 0, 1] = snake_h
        offset[:, :, :, 0, 0, 2] = snake_w
        return offset.reshape(n, 3 * self.offset_groups * temporal_kernel, out_d, out_h, out_w)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.offset_generator is not None:
            nn.init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                nn.init.zeros_(self.offset_generator.bias)
        if hasattr(self, "offset_norm"):
            nn.init.ones_(self.offset_norm.weight)
            nn.init.zeros_(self.offset_norm.bias)
        if self.mask_generator is not None:
            nn.init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                nn.init.zeros_(self.mask_generator.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset = self._make_temporal_snake_offset(x)
        if self.mask_generator is not None:
            mask = torch.sigmoid(self.mask_generator(x)) * self.mask_scale
        else:
            mask = None
        return super().forward(x, offset=offset, mask=mask)

    def extra_repr(self) -> str:
        fields = [super().extra_repr(), f"extend_scope={self.extend_scope}"]
        if self.mask_scale != 2.0:
            fields.append(f"mask_scale={self.mask_scale}")
        return ", ".join(field for field in fields if field)


class SnakeUnrestDeformConv3d(DeformConv3d):
        
    """Temporal dynamic snake deformable convolution for N,C,D,H,W tensors.

    中文说明：
    - 该层只实现时域蛇形约束，要求 kernel_size=(k,1,1)，其中 D 维对应时间 T。
    - 时间轴位置由标准 Conv3d 的 D 方向核位置给出；可学习偏移只作用在 H,W
      两个正交方向，并按 DSConv 的中心向两侧递推公式累积。
    - raw offset不做处理。
    - 输出 offset 会被组装成 tvdcn deform_conv3d 需要的
      N, 3 * offset_groups * k, out_d, out_h, out_w 格式，其中 dz 恒为 0。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        offset_kernel_size: Optional[_Size] = None,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        offset_groups: int = 1,
        modulated: bool = False,
        mask_groups: int = 1,
        offset_bias: bool = True,
        mask_bias: bool = True,
        extend_scope: float = 1.0,
        mask_scale: float = 2.0,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        if kernel_size_[1:] != (1, 1):
            raise ValueError("SnakeDeformConv3d only supports temporal kernels: kernel_size=(k, 1, 1)")
        if kernel_size_[0] % 2 == 0:
            raise ValueError("SnakeDeformConv3d requires an odd temporal kernel size")
        if extend_scope <= 0:
            raise ValueError("extend_scope must be positive")
        if mask_scale <= 0:
            raise ValueError("mask_scale must be positive")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            offset_groups=offset_groups,
            modulated=False,
            auto_offset=False,
            device=device,
            dtype=dtype,
        )

        if in_channels % mask_groups != 0:
            raise ValueError("in_channels must be divisible by mask_groups")
        if out_channels % mask_groups != 0:
            raise ValueError("out_channels must be divisible by mask_groups")

        factory_kwargs = {"device": device, "dtype": dtype}
        if offset_kernel_size is None:
            self.offset_kernel_size = self.kernel_size
            self.offset_padding = self.padding
        else:
            self.offset_kernel_size = _triple(offset_kernel_size)
            self.offset_padding = _same_padding(self.offset_kernel_size, self.dilation)

        factory_kwargs = {"device": device, "dtype": dtype}
        temporal_kernel = self.kernel_size[0]
        offset_channels = 2 * offset_groups * temporal_kernel
        self.offset_generator = nn.Conv3d(
            in_channels,
            offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=offset_groups,
            bias=offset_bias,
            **factory_kwargs,
        )

        self.modulated = modulated
        self.mask_groups = mask_groups
        self.extend_scope = float(extend_scope)
        self.mask_scale = float(mask_scale)
        self.auto_offset = True

        if modulated:
            self.mask_generator = nn.Conv3d(
                in_channels,
                mask_groups * temporal_kernel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=mask_groups,
                bias=mask_bias,
                **factory_kwargs,
            )
        else:
            self.mask_generator = None

        self.reset_parameters()

    @staticmethod
    def _accumulate_from_center(delta: Tensor) -> Tensor:
        center = delta.size(2) // 2
        zero = torch.zeros_like(delta[:, :, center])

        lower_offsets = []
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center - step]
            lower_offsets.append(acc)
        lower_offsets.reverse()

        upper_offsets = [zero]
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center + step]
            upper_offsets.append(acc)

        return torch.stack(lower_offsets + upper_offsets, dim=2)

    def _make_temporal_snake_offset(self, x: Tensor) -> Tensor:
        raw_offset = self.offset_generator(x)

        n, _, out_d, out_h, out_w = raw_offset.shape
        temporal_kernel = self.kernel_size[0]
        raw_offset = raw_offset.view(
            n,
            self.offset_groups,
            temporal_kernel,
            2,
            out_d,
            out_h,
            out_w,
        )

        delta_h = raw_offset[:, :, :, 0]
        delta_w = raw_offset[:, :, :, 1]
        snake_h = self._accumulate_from_center(delta_h)
        snake_w = self._accumulate_from_center(delta_w)

        offset = raw_offset.new_zeros(
            n,
            self.offset_groups,
            temporal_kernel,
            1,
            1,
            3,
            out_d,
            out_h,
            out_w,
        )
        offset[:, :, :, 0, 0, 1] = snake_h
        offset[:, :, :, 0, 0, 2] = snake_w
        return offset.reshape(n, 3 * self.offset_groups * temporal_kernel, out_d, out_h, out_w)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.offset_generator is not None:
            nn.init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                nn.init.zeros_(self.offset_generator.bias)
        if hasattr(self, "offset_norm"):
            nn.init.ones_(self.offset_norm.weight)
            nn.init.zeros_(self.offset_norm.bias)
        if self.mask_generator is not None:
            nn.init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                nn.init.zeros_(self.mask_generator.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset = self._make_temporal_snake_offset(x)
        if self.mask_generator is not None:
            mask = torch.sigmoid(self.mask_generator(x)) * self.mask_scale
        else:
            mask = None
        return super().forward(x, offset=offset, mask=mask)

    def extra_repr(self) -> str:
        fields = [super().extra_repr(), f"extend_scope={self.extend_scope}"]
        if self.mask_scale != 2.0:
            fields.append(f"mask_scale={self.mask_scale}")
        return ", ".join(field for field in fields if field)


class VelocitySnakeDeformConv3d(DeformConv3d):
    """Temporal snake DCN with a shared constant-velocity motion prior.

    For each offset group, the offset generator predicts a 2-D velocity and
    unconstrained 2-D residual increments. The velocity provides the common
    linear trajectory, while residuals use the center-outward snake rule.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Size,
        offset_kernel_size: Optional[_Size] = None,
        stride: _Size = 1,
        padding: Union[str, _Size] = 0,
        dilation: _Size = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        offset_groups: int = 1,
        modulated: bool = False,
        mask_groups: int = 1,
        offset_bias: bool = True,
        mask_bias: bool = True,
        extend_scope: float = 1.0,
        mask_scale: float = 2.0,
        use_residual: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        if kernel_size_[1:] != (1, 1):
            raise ValueError("VelocitySnakeDeformConv3d only supports temporal kernels: kernel_size=(k, 1, 1)")
        if kernel_size_[0] % 2 == 0:
            raise ValueError("VelocitySnakeDeformConv3d requires an odd temporal kernel size")
        if extend_scope <= 0:
            raise ValueError("extend_scope must be positive")
        if mask_scale <= 0:
            raise ValueError("mask_scale must be positive")

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            offset_groups=offset_groups,
            modulated=False,
            auto_offset=False,
            device=device,
            dtype=dtype,
        )

        if in_channels % mask_groups != 0:
            raise ValueError("in_channels must be divisible by mask_groups")
        if out_channels % mask_groups != 0:
            raise ValueError("out_channels must be divisible by mask_groups")

        factory_kwargs = {"device": device, "dtype": dtype}
        if offset_kernel_size is None:
            self.offset_kernel_size = self.kernel_size
            self.offset_padding = self.padding
        else:
            self.offset_kernel_size = _triple(offset_kernel_size)
            self.offset_padding = _same_padding(self.offset_kernel_size, self.dilation)

        factory_kwargs = {"device": device, "dtype": dtype}
        temporal_kernel = self.kernel_size[0]
        offset_channels = 2 * offset_groups * temporal_kernel
        self.offset_generator = nn.Conv3d(
            in_channels,
            offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=offset_groups,
            bias=offset_bias,
            **factory_kwargs,
        )
        self.velocity_norm = nn.BatchNorm3d(2 * offset_groups, **factory_kwargs)

        self.modulated = modulated
        self.mask_groups = mask_groups
        self.extend_scope = float(extend_scope)
        self.mask_scale = float(mask_scale)
        self.use_residual = bool(use_residual)
        self.auto_offset = True

        if modulated:
            self.mask_generator = nn.Conv3d(
                in_channels,
                mask_groups * temporal_kernel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=mask_groups,
                bias=mask_bias,
                **factory_kwargs,
            )
        else:
            self.mask_generator = None

        self.reset_parameters()

    @staticmethod
    def _accumulate_from_center(delta: Tensor) -> Tensor:
        center = delta.size(2) // 2
        zero = torch.zeros_like(delta[:, :, center])

        lower_offsets = []
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center - step]
            lower_offsets.append(acc)
        lower_offsets.reverse()

        upper_offsets = [zero]
        acc = zero
        for step in range(1, center + 1):
            acc = acc + delta[:, :, center + step]
            upper_offsets.append(acc)

        return torch.stack(lower_offsets + upper_offsets, dim=2)

    def _make_temporal_snake_offset(self, x: Tensor) -> Tensor:
        raw_offset = self.offset_generator(x)
        n, _, out_d, out_h, out_w = raw_offset.shape
        temporal_kernel = self.kernel_size[0]
        center = temporal_kernel // 2

        raw_offset = raw_offset.view(
            n,
            self.offset_groups,
            2 + 2 * (temporal_kernel - 1),
            out_d,
            out_h,
            out_w,
        )
        raw_velocity = raw_offset[:, :, :2]
        raw_velocity = raw_velocity.reshape(n, 2 * self.offset_groups, out_d, out_h, out_w)
        velocity = torch.tanh(self.velocity_norm(raw_velocity)) * self.extend_scope
        velocity = velocity.view(n, self.offset_groups, 2, out_d, out_h, out_w)
        residual = None
        if self.use_residual:
            residual = raw_offset[:, :, 2:].view(
                n,
                self.offset_groups,
                temporal_kernel - 1,
                2,
                out_d,
                out_h,
                out_w,
            )

        delta = raw_offset.new_zeros(
            n,
            self.offset_groups,
            temporal_kernel,
            2,
            out_d,
            out_h,
            out_w,
        )
        residual_index = 0
        for index in range(temporal_kernel):
            if index == center:
                continue
            direction = -1.0 if index < center else 1.0
            delta[:, :, index] = direction * velocity
            if residual is not None:
                delta[:, :, index] = delta[:, :, index] + residual[:, :, residual_index]
            residual_index += 1

        snake_h = self._accumulate_from_center(delta[:, :, :, 0])
        snake_w = self._accumulate_from_center(delta[:, :, :, 1])

        offset = raw_offset.new_zeros(
            n,
            self.offset_groups,
            temporal_kernel,
            1,
            1,
            3,
            out_d,
            out_h,
            out_w,
        )
        offset[:, :, :, 0, 0, 1] = snake_h
        offset[:, :, :, 0, 0, 2] = snake_w
        return offset.reshape(n, 3 * self.offset_groups * temporal_kernel, out_d, out_h, out_w)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "offset_generator") and self.offset_generator is not None:
            nn.init.zeros_(self.offset_generator.weight)
            if self.offset_generator.bias is not None:
                nn.init.zeros_(self.offset_generator.bias)
        if hasattr(self, "velocity_norm"):
            nn.init.ones_(self.velocity_norm.weight)
            nn.init.zeros_(self.velocity_norm.bias)
        if hasattr(self, "mask_generator") and self.mask_generator is not None:
            nn.init.zeros_(self.mask_generator.weight)
            if self.mask_generator.bias is not None:
                nn.init.zeros_(self.mask_generator.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset = self._make_temporal_snake_offset(x)
        if self.mask_generator is not None:
            mask = torch.sigmoid(self.mask_generator(x)) * self.mask_scale
        else:
            mask = None
        return super().forward(x, offset=offset, mask=mask)

    def extra_repr(self) -> str:
        fields = [super().extra_repr(), f"extend_scope={self.extend_scope}"]
        if self.mask_scale != 2.0:
            fields.append(f"mask_scale={self.mask_scale}")
        if not self.use_residual:
            fields.append("use_residual=False")
        return ", ".join(field for field in fields if field)


DCN1d = DeformConv1d
DCN2d = DeformConv2d
DCN3d = DeformConv3d
SnakeDCN3d = SnakeDeformConv3d
VelocitySnakeDCN3d = VelocitySnakeDeformConv3d
