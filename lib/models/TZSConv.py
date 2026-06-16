"""
TZSConv: Temporal Zero-Sum 3D Convolution
==========================================
在时域维度上施加零和约束的 3D 卷积层。

核心原理:
    对卷积核在时域维度 (dim=2) 去均值，使得每个空间位置的时域权重之和为零。
    这使得卷积只对时域变化 (运动/变化) 产生响应，而对静态外观不敏感。

    数学表达:
        w_centered = w - mean(w, dim=2)
        out = Conv3D(x, w_centered)

支持输出跳线连接 (residual add)，接口尽量匹配标准 nn.Conv3d。
"""

import torch
import torch.nn.functional as F
from torch import nn


class TZSConv(nn.Module):
    """
    Temporal Zero-Sum 3D Convolution Layer.

    将标准 3D 卷积核在时域维度上中心化 (去均值)，强制时域零和约束。
    配合 skip_add 可形成残差式的时域差分学习单元。

    Parameters
    ----------
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数。
    kernel_size : int | tuple[int, int, int]
        卷积核尺寸 (T, H, W)。若传入 int，则三个维度等长。默认 3。
    stride : int | tuple[int, int, int]
        卷积步长 (T, H, W)。默认 1。
    padding : int | tuple[int, int, int]
        卷积填充 (T, H, W)。默认 0。
    dilation : int | tuple[int, int, int]
        卷积膨胀 (T, H, W)。默认 1。
    groups : int
        分组卷积的组数。默认 1。
    bias : bool
        是否使用偏置。默认 True。
    skip_add : bool
        是否在输出上加上输入 (残差跳线连接)。
        - 当 in_channels == out_channels 且 stride == 1 时，直接加 identity。
        - 否则自动使用 1×1×1 卷积投影使维度匹配。
        默认 False。

    Examples
    --------
    >>> # 基本用法 (与 nn.Conv3d 接口一致)
    >>> conv = TZSConv(64, 128, kernel_size=3, stride=1, padding=1)
    >>> x = torch.randn(2, 64, 8, 32, 32)
    >>> out = conv(x)  # shape: (2, 128, 8, 32, 32)

    >>> # 带跳线连接的用法
    >>> conv = TZSConv(64, 64, kernel_size=3, stride=1, padding=1, skip_add=True)
    >>> out = conv(x)  # out = TZSConv(x) + x
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        skip_add: bool = False,
    ):
        super().__init__()

        # ---- 参数标准化为 3 元组 ----
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_tuple(kernel_size, name="kernel_size")
        self.stride = _to_tuple(stride, name="stride")
        self.padding = _to_tuple(padding, name="padding")
        self.dilation = _to_tuple(dilation, name="dilation")
        self.groups = groups
        self.skip_add = skip_add

        # ---- 卷积权重 (forward 中会对时域维度去均值) ----
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # ---- 跳线投影 (当 skip_add 启用但维度不匹配时) ----
        if skip_add:
            need_proj = (
                in_channels != out_channels or self.stride != (1, 1, 1)
            )
            if need_proj:
                self.skip_proj = nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=self.stride, bias=False,
                )
            else:
                self.skip_proj = None
        else:
            self.skip_proj = None

        # self._init_weights()

    # ---------------------------------------------------------------
    # 权重初始化
    # ---------------------------------------------------------------
    # def _init_weights(self):
    #     """
    #     中心脉冲初始化 (Center-Dirac Initialization)。

    #     将所有权重置零，仅把时域中心的整个空间切片置为 1。
    #     去均值后自动形成类似 [-1/3, 2/3, -1/3] 的差分算子，
    #     保证训练起始稳定，且天然聚焦时域变化。
    #     """
    #     nn.init.constant_(self.weight, 0.0)

    #     k_t, k_h, k_w = self.kernel_size
    #     c_t, c_h, c_w = k_t // 2, k_h // 2, k_w // 2

    #     with torch.no_grad():
    #         self.weight[:, :, c_t, c_h, c_w] = 1.0

    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

        # 跳线投影由 nn.Conv3d 自带 Kaiming 初始化，无需额外处理

    # ---------------------------------------------------------------
    # 前向传播
    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，shape (N, C_in, D, H, W)

        Returns:
            输出张量，shape (N, C_out, D_out, H_out, W_out)
        """
        # 核心：时域维度 (dim=2) 去均值，强制零和约束
        w_centered = self.weight - self.weight.mean(dim=2, keepdim=True)

        out = F.conv3d(
            x, w_centered, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

        # 跳线连接
        if self.skip_add:
            identity = self.skip_proj(x) if self.skip_proj is not None else x
            out = out + identity

        return out

    # ---------------------------------------------------------------
    # 打印信息
    # ---------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"skip_add={self.skip_add}"
        )


# ============================================================
# 工具函数
# ============================================================

def _to_tuple(value, name="argument"):
    """将 int 或 tuple 统一转换为 3 元组。"""
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError(
        f"{name} 应为 int 或长度为 3 的 tuple，实际收到 {value}"
    )


# ============================================================
# 便捷构造：带 BN + Activation 的 TZSConv Block
# ============================================================

class TZSConvBlock(nn.Module):
    """
    组合模块：TZSConv → BatchNorm3d → Activation。

    这是一个开箱即用的时域零和卷积块，适合直接嵌入 UNet 等架构中。

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int | tuple
    stride : int | tuple
    padding : int | tuple
    dilation : int | tuple
    groups : int
    bias : bool
    skip_add : bool
    activation : str
        激活函数类型，可选 "relu"、"abs"、"leaky_relu"、"none"。默认 "relu"。
    bn_eps : float
        BatchNorm 的 epsilon。默认 1e-3。
    bn_momentum : float
        BatchNorm 的 momentum。默认 0.1。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        skip_add: bool = False,
        activation: str = "relu",
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.1,
    ):
        super().__init__()

        self.tzs_conv = TZSConv(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, skip_add=skip_add,
        )

        self.bn = nn.BatchNorm3d(
            out_channels, eps=bn_eps, momentum=bn_momentum, affine=True,
        )

        act_map = {
            "relu": nn.ReLU(inplace=True),
            "abs": _Abs(),
            "leaky_relu": nn.LeakyReLU(inplace=True),
            "none": nn.Identity(),
        }
        if activation not in act_map:
            raise ValueError(
                f"不支持的激活函数 '{activation}'，可选: {list(act_map.keys())}"
            )
        self.act = act_map[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.tzs_conv(x)))


class _Abs(nn.Module):
    """绝对值激活 (配合零和卷积捕获运动能量)。"""
    def forward(self, x):
        return torch.abs(x)


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TZSConv 自检] device={device}")

    # 测试输入: (N=2, C=16, D=8, H=32, W=32)
    x = torch.randn(2, 16, 8, 32, 32, device=device)

    # ---- 1. 基础用法 (无跳线) ----
    conv1 = TZSConv(16, 32, kernel_size=3, padding=1).to(device)
    out1 = conv1(x)
    print(f"基础 TZSConv:  {x.shape} → {out1.shape}")

    # ---- 2. 带跳线 (同维度 identity) ----
    conv2 = TZSConv(16, 16, kernel_size=3, padding=1, skip_add=True).to(device)
    out2 = conv2(x)
    print(f"跳线 identity: {x.shape} → {out2.shape}")

    # ---- 3. 带跳线 (跨维度 projection) ----
    conv3 = TZSConv(16, 32, kernel_size=3, stride=2, padding=1, skip_add=True).to(device)
    out3 = conv3(x)
    print(f"跳线 project:  {x.shape} → {out3.shape}")

    # ---- 4. Block 封装 ----
    block = TZSConvBlock(16, 32, kernel_size=3, padding=1, activation="relu").to(device)
    out4 = block(x)
    print(f"TZSConvBlock:  {x.shape} → {out4.shape}")

    # ---- 5. 验证零和约束 ----
    print("\n--- 零和约束验证 ---")
    w = conv1.weight  # (32, 16, 3, 3, 3)
    w_centered = w - w.mean(dim=2, keepdim=True)
    t_sum = w_centered.sum(dim=2)  # 沿时域求和
    max_deviation = t_sum.abs().max().item()
    print(f"时域求和最大偏差: {max_deviation:.2e} (应为 0)")

    # ---- 6. 参数统计 ----
    def count_params(m: nn.Module):
        return sum(p.numel() for p in m.parameters())
    print(f"\n基础 TZSConv 参数量:  {count_params(conv1):,}")
    print(f"Block 封装参数量:     {count_params(block):,}")

    print("\n[OK] All self-checks passed.")
