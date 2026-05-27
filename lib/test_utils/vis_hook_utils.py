"""
vis_hook_utils.py
─────────────────────────────────────────────────────────────────────────────
通过 PyTorch forward-hook 机制捕获指定层的输出特征图，并保存为可视化 PNG。

设计目标
  - 完全不侵入模型前向逻辑
  - 支持稀疏卷积输出（spconv SparseConvTensor）自动转 dense
  - 支持返回裸 features Tensor（如 SparseSymmetricCosineAttention）：
      利用输入的 SparseConvTensor 的 indices/spatial_shape 重建再 dense
      对于返回 (enhanced_features, max_scores) 的层，取第 [0] 项（enhanced_features）
  - 支持三种聚合模式：mean / max / 指定通道下标
  - 每段视频每层独立保存前 N 张，层之间互不干扰
  - 归一化策略：同一层同一 batch 内所有帧共享统一 v_max，
      保证 conv1 与 shortcut1 等层的响应强度可直接比较

公开 API
  VisHookManager(model, vis_layers, vis_mode, save_vis_path_upper,
                 max_vis_frames=10)
      .register(video_folder_name)
      .save_batch(patch_ims)
      .remove()
"""

from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
import cv2
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 工具：把层名转成合法的文件名片段
# ─────────────────────────────────────────────────────────────────────────────
def _layer_name_to_safe(layer_name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', layer_name)


# ─────────────────────────────────────────────────────────────────────────────
# 工具：从 hook 的 (input, output) 中提取 dense numpy 数组
#
#   处理优先级：
#   1. output 是 SparseConvTensor → 直接 .dense()
#   2. output 是 tuple/list → 取第一个 SparseConvTensor 元素 dense()；
#      若没有 SparseConvTensor，则检查 input 中是否有 SparseConvTensor
#      用于重建（适用于 shortcut1 类：output 是裸 features Tensor）
#   3. output 是普通 dense Tensor（≥3维）→ 直接转
#   4. output 是 2D Tensor [N, C]（稀疏 features）→ 需要 input 重建
# ─────────────────────────────────────────────────────────────────────────────
def _find_sparse_conv_tensor(obj):
    """递归在 tuple/list 中找第一个 SparseConvTensor"""
    if hasattr(obj, 'dense') and hasattr(obj, 'indices') and hasattr(obj, 'spatial_shape'):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = _find_sparse_conv_tensor(item)
            if result is not None:
                return result
    return None


def _sparse_to_dense(sparse_tensor):
    """SparseConvTensor → dense numpy [B, C, D, H, W] 或 [B, C, H, W]"""
    try:
        dense = sparse_tensor.dense()  # torch.Tensor
        return dense.detach().float().cpu().numpy()
    except Exception as e:
        print(f'[vis_hook_utils] .dense() 失败: {e}')
        return None


def _rebuild_from_features(features_tensor, input_sparse):
    """
    用输入 SparseConvTensor 的 indices/spatial_shape 重建，
    再 dense 转 numpy。
    features_tensor: [N, C] Tensor（新的 features）
    input_sparse: 原始输入 SparseConvTensor（提供 indices, spatial_shape, batch_size）
    """
    try:
        from lib.models.spconv_utils import spconv
        new_sparse = input_sparse.replace_feature(features_tensor)
        return _sparse_to_dense(new_sparse)
    except Exception:
        # 手动 scatter 到 dense
        try:
            indices = input_sparse.indices  # [N, 1+ndim]
            spatial_shape = input_sparse.spatial_shape  # e.g. [T, H, W]
            batch_size = input_sparse.batch_size
            C = features_tensor.shape[1]
            # 构造 dense: [B, C, *spatial_shape]
            dense_shape = [batch_size, C] + list(spatial_shape)
            dense = torch.zeros(dense_shape, dtype=torch.float32,
                                device=features_tensor.device)
            b_idx = indices[:, 0].long()
            spatial_idx = [indices[:, i+1].long() for i in range(len(spatial_shape))]
            dense[tuple([b_idx, slice(None)] + spatial_idx)] = features_tensor.float()
            return dense.detach().cpu().numpy()
        except Exception as e2:
            print(f'[vis_hook_utils] 重建稀疏特征失败: {e2}')
            return None


def _extract_numpy(hook_input, hook_output):
    """
    从 hook 的 input tuple 和 output 中提取 dense numpy。

    返回值：(arr, fixed_v_max)
      - arr:         dense numpy，已处理为非负
      - fixed_v_max: float 表示该层应使用固定 v_max（如 sigmoid 输出固定为 1.0），
                     None 表示由调用方动态计算 v_max

    处理优先级：
    1. output 是 SparseConvTensor → 直接 .dense()，v_max 动态
    2. output 是 tuple/list：
       a. 含 SparseConvTensor → 取第一个 dense()，v_max 动态
       b. 全部是 [N,C] 裸 Tensor（如 shortcut1 返回 (enhanced_features, max_scores)）
          → 取第 [0] 项 enhanced_features [N,C]，v_max 动态
    3. output 是 [N,C] 2D Tensor → 借助输入 SparseConvTensor 重建，v_max 动态
    4. output 是 ≥3 维 dense Tensor：
       - 全非负 → 直接返回，v_max 动态
       - 含负值（logit，如 final_conv）→ 做 sigmoid，v_max 固定为 1.0
         sigmoid 输出即目标概率，量纲固定，0.5 恒为绿色，无目标帧自然偏蓝
    """
    # ── 策略1：output 直接是 SparseConvTensor ──────────────────────────────
    if hasattr(hook_output, 'dense') and hasattr(hook_output, 'indices'):
        return _sparse_to_dense(hook_output), None

    # ── 策略2：output 是 tuple/list ─────────────────────────────────────────
    if isinstance(hook_output, (tuple, list)):
        # 2a：含 SparseConvTensor，直接 dense
        sp = _find_sparse_conv_tensor(hook_output)
        if sp is not None:
            return _sparse_to_dense(sp), None

        # 2b：全部是普通 Tensor（如 shortcut1 返回 (enhanced_features [N,C], max_scores [N,1])）
        #     取第一项 enhanced_features [N,C]（增强后的特征，与 conv1 空间结构相同但幅值更高）
        #     注意：用户要求"保存特征图时取[0]项"，即 enhanced_features
        tensors_2d = [item for item in hook_output
                      if isinstance(item, torch.Tensor) and item.ndim == 2]
        if not tensors_2d:
            return None, None
        # 取第一个 Tensor，即 enhanced_features [N, C]
        vis_tensor = tensors_2d[0]
        in_sparse = None
        if isinstance(hook_input, (tuple, list)):
            in_sparse = _find_sparse_conv_tensor(hook_input)
        elif hasattr(hook_input, 'indices'):
            in_sparse = hook_input
        if in_sparse is not None:
            return _rebuild_from_features(vis_tensor, in_sparse), None
        return None, None

    # ── 此时 hook_output 应为单个普通 Tensor ────────────────────────────────
    if not isinstance(hook_output, torch.Tensor):
        print(f'[vis_hook_utils] 不支持的输出类型: {type(hook_output)}')
        return None, None

    arr = hook_output.detach().float().cpu().numpy()

    # ── 策略3：≥3 维 dense Tensor ────────────────────────────────────────────
    if arr.ndim >= 3:
        if arr.min() < 0:
            # 含负值说明该层未经激活（如 final_conv 的 logit 输出）。
            # 使用 sigmoid 映射到 (0, 1)：物理含义是目标存在概率，量纲固定。
            # v_max 硬编码为 1.0，不参与跨层动态 v_max 计算，
            # 从而不会因为概率图的尺度问题压制其他特征层的颜色。
            arr = 1.0 / (1.0 + np.exp(-arr.astype(np.float64))).astype(np.float32)
            return arr, 1.0   # fixed_v_max = 1.0
        return arr, None      # 全非负，v_max 动态

    # ── 策略4：2D [N,C] 稀疏 features，借助输入重建 ──────────────────────────
    if arr.ndim == 2:
        in_sparse = None
        if isinstance(hook_input, (tuple, list)):
            in_sparse = _find_sparse_conv_tensor(hook_input)
        elif hasattr(hook_input, 'indices'):
            in_sparse = hook_input
        if in_sparse is not None:
            return _rebuild_from_features(hook_output, in_sparse), None
        print(f'[vis_hook_utils] 输出为 [N,C] 裸 features 但找不到输入 SparseConvTensor')
        return None, None

    print(f'[vis_hook_utils] 输出为 1D Tensor，无法可视化')
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 工具：按 vis_mode 对 [C, H, W] 做通道聚合，返回 [H, W]
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(feat_map: np.ndarray, vis_mode: str) -> np.ndarray:
    if feat_map.ndim == 2:
        return feat_map.astype(np.float32)
    if feat_map.ndim != 3:
        feat_map = feat_map.reshape(-1, feat_map.shape[-2], feat_map.shape[-1])
    if vis_mode == 'mean':
        return feat_map.mean(axis=0).astype(np.float32)
    elif vis_mode == 'max':
        return feat_map.max(axis=0).astype(np.float32)
    else:
        try:
            ch_idx = int(vis_mode)
        except ValueError:
            print(f'[vis_hook_utils] 未知 vis_mode="{vis_mode}"，回退到 mean')
            return feat_map.mean(axis=0).astype(np.float32)
        if ch_idx < 0 or ch_idx >= feat_map.shape[0]:
            print(f'[vis_hook_utils] 通道下标 {ch_idx} 超出范围 [0,{feat_map.shape[0]})，回退到 mean')
            return feat_map.mean(axis=0).astype(np.float32)
        return feat_map[ch_idx].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 工具：保存伪彩色 PNG
#
# 归一化策略：[0, v_max] → [0, 255]
#   - 所有特征图 ReLU 后均为非负值，0 固定映射为 0（JET蓝/背景）
#   - v_max 由调用方传入：同一层同一 batch 的所有帧共享同一 v_max，
#     保证同帧/跨帧之间的响应强度可直接比较
#   - 若 v_max 未传入（None），则退化为逐帧各自归一化（仅用于调试）
# ─────────────────────────────────────────────────────────────────────────────
def _save_gray_png(arr2d: np.ndarray, save_path: str, v_max: float = None):
    if v_max is None:
        v_max = float(arr2d.max())
    if v_max < 1e-8:
        img8 = np.zeros_like(arr2d, dtype=np.uint8)
    else:
        img8 = np.clip(arr2d / v_max * 255, 0, 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_color)


# ─────────────────────────────────────────────────────────────────────────────
# 工具：定位子模块
# ─────────────────────────────────────────────────────────────────────────────
def _get_submodule(model: torch.nn.Module, layer_name: str):
    if layer_name.startswith('model.'):
        layer_name = layer_name[len('model.'):]
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


# ─────────────────────────────────────────────────────────────────────────────
# 工具：将 dense numpy 按时间/batch 维拆分为 (frame_idx, [C,H,W]) 列表
# ─────────────────────────────────────────────────────────────────────────────
def _split_to_frames(arr: np.ndarray, patch_len: int):
    """
    arr 为 dense numpy，形状可能是：
      [B, C, H, W]      → B 是 batch，通常 =1；按 B 拆（一般只有1帧）
      [B, C, T, H, W]   → 时空体，B=1，按 T 拆帧
      [C, H, W]         → 单帧
      [H, W]            → 单帧单通道

    patch_len 用于辅助判断：若 B 维度 == patch_len，说明 B 实际是时间维
    返回 list of (frame_idx: int, feat: ndarray [C,H,W] or [H,W])
    """
    ndim = arr.ndim

    if ndim == 2:
        return [(0, arr)]

    if ndim == 3:
        return [(0, arr)]

    if ndim == 4:
        B, C, H, W = arr.shape
        # 若 B == patch_len，认为第0维是时间帧，逐帧拆
        if B == patch_len:
            return [(t, arr[t]) for t in range(B)]
        else:
            # B 是真正的 batch（通常=1），取 B=0 作为单帧
            return [(0, arr[0])]

    if ndim == 5:
        # [B, C, T, H, W]，取 B=0，按 T 拆帧
        arr_b0 = arr[0]          # [C, T, H, W]
        T = arr_b0.shape[1]
        return [(t, arr_b0[:, t, :, :]) for t in range(T)]

    # 更高维度：展平前缀，当单帧处理
    arr_flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    return [(0, arr_flat)]


# ─────────────────────────────────────────────────────────────────────────────
# 核心类：VisHookManager
# ─────────────────────────────────────────────────────────────────────────────
class VisHookManager:

    def __init__(
        self,
        model: torch.nn.Module,
        vis_layers: list,
        vis_mode: str,
        save_vis_path_upper: str,
        max_vis_frames: int = 10,
        patch_len: int = 1,
    ):
        self.model = model
        self.vis_layers = vis_layers
        self.vis_mode = vis_mode
        self.save_vis_path_upper = save_vis_path_upper
        self.max_vis_frames = max_vis_frames
        self.patch_len = patch_len  # 用于区分 [T,C,H,W] 和 [B,C,H,W]

        self._hooks = []
        # {lname: [(input_tuple, output), ...]}  每次 forward 追加
        self._captured = {}
        self._video_folder = None
        self._frames_saved_per_layer = {}

        self._valid_layers = []
        for lname in vis_layers:
            mod = _get_submodule(model, lname)
            if mod is None:
                print(f'[vis_hook_utils] 警告：找不到层 "{lname}"，将跳过')
            else:
                self._valid_layers.append(lname)
                print(f'[vis_hook_utils] 已定位层: "{lname}" → {type(mod).__name__}')

    def register(self, video_folder_name: str):
        self.remove()
        self._video_folder = video_folder_name
        self._captured = {lname: [] for lname in self._valid_layers}
        self._frames_saved_per_layer = {lname: 0 for lname in self._valid_layers}

        for lname in self._valid_layers:
            mod = _get_submodule(self.model, lname)

            def _make_hook(name):
                def _hook(module, inp, out):
                    # 同时保存 input 和 output，用于稀疏 features 重建
                    self._captured[name].append((inp, out))
                return _hook

            handle = mod.register_forward_hook(_make_hook(lname))
            self._hooks.append(handle)

    def save_batch(self, patch_ims: list):
        if not self._captured:
            return
        if all(self._frames_saved_per_layer.get(ln, 0) >= self.max_vis_frames
               for ln in self._valid_layers):
            self._captured = {lname: [] for lname in self._valid_layers}
            return

        vis_folder = os.path.join(self.save_vis_path_upper, self._video_folder)
        os.makedirs(vis_folder, exist_ok=True)

        # ── 第一步：提取所有层的聚合图 ──────────────────────────────────────
        # layer_frames:  {lname: [(fi, agg_2d), ...]}
        # layer_fixed_vmax: {lname: float or None}
        #   float  → 该层已做 sigmoid，量纲固定，不参与跨层 v_max 计算
        #   None   → 普通特征层，参与同帧跨层动态 v_max 计算
        layer_frames:     dict = {}
        layer_fixed_vmax: dict = {}
        for lname in self._valid_layers:
            captures = self._captured.get(lname, [])
            if not captures:
                continue
            hook_input, hook_output = captures[-1]
            arr, fixed_vmax = _extract_numpy(hook_input, hook_output)
            if arr is None:
                continue
            frames_to_save = _split_to_frames(arr, self.patch_len)
            agg_list = [(fi, _aggregate(feat, self.vis_mode))
                        for fi, feat in frames_to_save]
            layer_frames[lname]     = agg_list
            layer_fixed_vmax[lname] = fixed_vmax

        if not layer_frames:
            self._captured = {lname: [] for lname in self._valid_layers}
            return

        # ── 第二步：按帧索引遍历，计算动态层的跨层统一 v_max ─────────────────
        all_fi = sorted({fi for agg_list in layer_frames.values()
                            for fi, _ in agg_list})

        for fi in all_fi:
            # 收集该帧所有层的聚合图
            fi_aggs: dict = {}
            for lname, agg_list in layer_frames.items():
                for frame_idx, agg in agg_list:
                    if frame_idx == fi:
                        fi_aggs[lname] = agg
                        break
            if not fi_aggs:
                continue

            # 动态层（fixed_vmax is None）共享同帧的全局 v_max，
            # 保证 conv1/shortcut1 等同量纲层的颜色可直接比较。
            # 固定层（final_conv sigmoid 输出）使用自身的 fixed_vmax，
            # 不参与动态计算，量纲独立且跨帧一致。
            dynamic_vals = [agg.max() for lname, agg in fi_aggs.items()
                            if layer_fixed_vmax.get(lname) is None]
            dynamic_v_max = float(max(dynamic_vals)) if dynamic_vals else 0.0

            # 命名：fi 对应 patch_ims 中的帧索引
            if fi < len(patch_ims):
                img_stem = os.path.splitext(patch_ims[fi])[0]
            else:
                img_stem = f'frame{fi:04d}'

            # ── 第三步：保存各层特征图 ──────────────────────────────────────
            for lname, agg in fi_aggs.items():
                if self._frames_saved_per_layer[lname] >= self.max_vis_frames:
                    continue
                fixed = layer_fixed_vmax.get(lname)
                v_max = fixed if fixed is not None else dynamic_v_max
                safe_name = _layer_name_to_safe(lname)
                save_path = os.path.join(vis_folder, f'{img_stem}__{safe_name}.png')
                _save_gray_png(agg, save_path, v_max=v_max)
                self._frames_saved_per_layer[lname] += 1

        self._captured = {lname: [] for lname in self._valid_layers}

    def remove(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._captured.clear()
