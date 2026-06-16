import torch
import torch.nn.functional as F
from torch import nn


def _to_tuple(value, name="argument"):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError(f"{name} should be int or a tuple/list of length 3, got {value}")


class TOSConv_once_conv_masked(nn.Module):
    """Temporal-spatial sum-one convolution for per-frame background estimation.

    This is the TKK extension of the original T11 TOSConv.

    Main idea:
        Learn one source-frame/spatial-offset distribution:
            weight: [O, I_g, T, K, K]

        For each target frame t:
            remove the whole source frame s=t by temporal mask;
            renormalize the remaining (T-1)*K*K candidates;
            estimate background from other frames' KxK neighborhoods.

    K=1:
        Falls back to the original TOSConv_once_conv_masked logic.

    Input:
        x: [B, C_in, T, H, W]

    Output:
        out: [B, C_out, T, H_out, W_out]
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

        if self.stride != (1, 1, 1):
            raise ValueError("TOSConvTKK_shared only supports stride=1")

        if self.dilation != (1, 1, 1):
            raise ValueError("TOSConvTKK_shared only supports dilation=1")

        if self.padding[0] != 0:
            raise ValueError("TOSConvTKK_shared only supports temporal padding=0")

        if self.kernel_size[0] <= 1:
            raise ValueError("TOSConvTKK_shared requires at least 2 frames")

        temporal_size = self.kernel_size[0]
        kh, kw = self.kernel_size[1:]
        ph, pw = self.padding[1:]

        if kh <= 0 or kw <= 0:
            raise ValueError("TOSConvTKK_shared expects positive spatial kernel sizes")

        if kh != kw:
            raise ValueError("TOSConvTKK_shared only supports square spatial kernels KxK")

        if (kh, kw) == (1, 1) and (ph, pw) != (0, 0):
            raise ValueError("TOSConvTKK_shared with K=1 only supports spatial padding=0")

        self.temporal_size = temporal_size
        self.kh = kh
        self.kw = kw
        self.spatial_padding = (ph, pw)
        self.use_spatial_neighborhood = (kh, kw) != (1, 1)

        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        if not self.use_spatial_neighborhood:
            # K=1: exactly the same shape as original T11 TOSConv.
            # weight: [O, I_g, T]
            self.weight = nn.Parameter(
                torch.empty(
                    out_channels,
                    self.in_channels_per_group,
                    temporal_size,
                )
            )
        else:
            # K>1: shared source-frame/spatial-offset logits.
            # weight: [O, I_g, T_source, K, K]
            #
            # Note:
            # This is NOT [O, I_g, T_target, T_source, K, K].
            # Target-frame-specific weights are obtained by masking source=t
            # and renormalizing, same as the original T11 logic.
            self.weight = nn.Parameter(
                torch.empty(
                    out_channels,
                    self.in_channels_per_group,
                    temporal_size,
                    kh,
                    kw,
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
            self.skip_proj = (
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
                if need_proj
                else None
            )
        else:
            self.skip_proj = None

        # Zero logits:
        #   K=1 -> mean of other T-1 frames.
        #   K>1 -> mean of other (T-1)*K*K candidates.
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _forward_k1(self, x):
        """Original T x 1 x 1 path. Same logic as TOSConv_once_conv_masked."""
        n, _, t, h, w = x.shape

        base_weights = F.softmax(self.weight, dim=2)

        masked_weights = (
            base_weights.unsqueeze(2)
            * self.temporal_exclusion_mask.unsqueeze(0).unsqueeze(0)
        )

        denom = masked_weights.sum(dim=3, keepdim=True).clamp_min(
            torch.finfo(masked_weights.dtype).eps
        )

        normalized_weights = masked_weights / denom

        x_grouped = x.view(
            n,
            self.groups,
            self.in_channels_per_group,
            t,
            h,
            w,
        )

        grouped_weights = normalized_weights.view(
            self.groups,
            self.out_channels_per_group,
            self.in_channels_per_group,
            t,
            t,
        )

        out = torch.einsum(
            "goits,ngishw->ngothw",
            grouped_weights,
            x_grouped,
        )

        out = out.reshape(n, self.out_channels, t, h, w)

        return out

    def _forward_tkk_shared(self, x):
        """Strict (T-1) x K x K background estimation with shared source weights.

        This is mathematically equivalent to:
            1. softmax over T*K*K source candidates;
            2. expand to target frame dimension;
            3. mask the diagonal source frame s=t;
            4. renormalize;
            5. weighted sum.

        For speed and memory, it uses an equivalent:
            total contribution - current source-frame contribution
        formulation, avoiding explicit [T_target, T_source, K*K] weights.
        """
        n, c, t, h, w = x.shape
        kh, kw = self.kh, self.kw
        ph, pw = self.spatial_padding
        kk = kh * kw

        out_h = h + 2 * ph - kh + 1
        out_w = w + 2 * pw - kw + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid output size ({out_h}, {out_w}). "
                f"Check input size {(h, w)}, kernel size {(kh, kw)}, padding {(ph, pw)}."
            )

        # ------------------------------------------------------------
        # 1. Base distribution over source frame and KxK offset
        # ------------------------------------------------------------
        # self.weight: [O, I_g, T, K, K]
        # base_weights: [O, I_g, T, K*K]
        base_weights = F.softmax(
            self.weight.view(
                self.out_channels,
                self.in_channels_per_group,
                t * kk,
            ),
            dim=2,
        ).view(
            self.out_channels,
            self.in_channels_per_group,
            t,
            kk,
        )

        # Grouped base weights:
        # [O, I_g, T, K*K]
        # -> [G, O_g, I_g, T, K*K]
        grouped_weights = base_weights.view(
            self.groups,
            self.out_channels_per_group,
            self.in_channels_per_group,
            t,
            kk,
        )

        # Frame mass:
        # mass[g,o,i,s] = sum_k alpha[g,o,i,s,k]
        #
        # For target frame t, the denominator after excluding source=t is:
        # denom[..., t] = 1 - mass[..., t]
        frame_mass = grouped_weights.sum(dim=-1)

        total_mass = frame_mass.sum(dim=-1, keepdim=True)
        denom = (total_mass - frame_mass).clamp_min(
            torch.finfo(grouped_weights.dtype).eps
        )

        # denom: [G, O_g, I_g, T]
        denom = denom.view(
            1,
            self.groups,
            self.out_channels_per_group,
            self.in_channels_per_group,
            t,
            1,
            1,
        )

        # ------------------------------------------------------------
        # 2. Extract KxK spatial neighborhoods from each source frame
        # ------------------------------------------------------------
        # x: [B, C, T, H, W]
        # -> [B*T, C, H, W]
        x_2d = x.transpose(1, 2).contiguous().view(n * t, c, h, w)

        # patches: [B*T, C*K*K, H_out*W_out]
        patches = F.unfold(
            x_2d,
            kernel_size=(kh, kw),
            padding=(ph, pw),
            stride=1,
            dilation=1,
        )

        # [B*T, C*K*K, H_out*W_out]
        # -> [B, T, G, I_g, K*K, H_out, W_out]
        patches = patches.view(
            n,
            t,
            self.groups,
            self.in_channels_per_group,
            kk,
            out_h,
            out_w,
        )

        # -> [B, G, I_g, T_source, K*K, H_out, W_out]
        patches = patches.permute(0, 2, 3, 1, 4, 5, 6)

        # ------------------------------------------------------------
        # 3. Optimized masked-renormalized background estimation
        # ------------------------------------------------------------
        # Conceptually:
        #   B_t = sum_{s!=t,k} alpha[s,k] * patch[s,k]
        #         -----------------------------------------
        #             sum_{s!=t,k} alpha[s,k]
        #
        # Instead of explicitly forming [T_target, T_source, K*K],
        # compute:
        #   all_contrib      = sum_{s,k} alpha[s,k] * patch[s,k]
        #   current_contrib  = sum_k alpha[t,k] * patch[t,k]
        #   B_t = (all_contrib - current_contrib[t]) / (1 - mass[t])
        #
        # raw_contrib:
        #   [B, G, O_g, I_g, T_source, H_out, W_out]
        raw_contrib = torch.einsum(
            "goisk,ngiskhw->ngoishw",
            grouped_weights,
            patches,
        )

        # all_contrib:
        #   [B, G, O_g, I_g, 1, H_out, W_out]
        all_contrib = raw_contrib.sum(dim=4, keepdim=True)

        # Remove source frame equal to target frame, then renormalize.
        # result before summing I_g:
        #   [B, G, O_g, I_g, T_target, H_out, W_out]
        out = (all_contrib - raw_contrib) / denom

        # Sum over input channels inside each group:
        #   [B, G, O_g, T, H_out, W_out]
        out = out.sum(dim=3)

        out = out.reshape(n, self.out_channels, t, out_h, out_w)

        return out

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                f"TOSConvTKK_shared expects a 5D tensor, got shape {tuple(x.shape)}"
            )

        identity_input = x

        n, _, t, h, w = x.shape
        if t != self.temporal_size:
            raise ValueError(
                f"TOSConvTKK_shared expects {self.temporal_size} frames, but received {t}"
            )

        if self.use_spatial_neighborhood:
            out = self._forward_tkk_shared(x)
        else:
            out = self._forward_k1(x)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        if self.skip_add:
            identity = (
                self.skip_proj(identity_input)
                if self.skip_proj is not None
                else identity_input
            )

            if identity.shape != out.shape:
                raise ValueError(
                    f"skip_add requires identity and background to have the same shape, "
                    f"but got identity={tuple(identity.shape)}, background={tuple(out.shape)}. "
                    f"For K>1, use padding=(0, K//2, K//2) to keep H,W unchanged."
                )

            out = identity - out

        return out

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}, "
            f"skip_add={self.skip_add}, "
            f"use_spatial_neighborhood={self.use_spatial_neighborhood}"
        )