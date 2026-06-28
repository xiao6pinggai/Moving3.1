import math
from collections import defaultdict

import torch
import torch.nn as nn

try:
    from lib.models.spconv_utils import spconv
    SPARSE_CONV_TYPES = (spconv.conv.SparseConvolution,)
except Exception:
    spconv = None
    SPARSE_CONV_TYPES = ()


def count_parameters(model, trainable_only=False):
    seen = set()
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        total += int(param.numel())
    return total


def extract_feature_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if hasattr(obj, "features") and isinstance(obj.features, torch.Tensor):
        return obj.features
    if isinstance(obj, (tuple, list)):
        for item in obj:
            tensor = extract_feature_tensor(item)
            if tensor is not None:
                return tensor
    return None


def estimate_sigmoid_flops(tensor):
    return 4.0 * float(tensor.numel())


def _kernel_volume(module):
    kernel_size = getattr(module, "kernel_size", 1)
    if isinstance(kernel_size, int):
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "dim") and weight.dim() >= 3:
            return kernel_size ** max(weight.dim() - 2, 1)
        return int(kernel_size)
    return int(math.prod(kernel_size))


def estimate_sparse_pair_count(module, inputs, output):
    indice_key = getattr(module, "indice_key", None)
    if indice_key is None:
        return None

    sparse_tensors = []
    if isinstance(output, (tuple, list)):
        sparse_tensors.extend(item for item in output if hasattr(item, "find_indice_pair"))
    elif hasattr(output, "find_indice_pair"):
        sparse_tensors.append(output)

    for item in inputs:
        if hasattr(item, "find_indice_pair"):
            sparse_tensors.append(item)

    for sparse_tensor in sparse_tensors:
        try:
            indice_data = sparse_tensor.find_indice_pair(indice_key)
        except Exception:
            indice_data = None
        if indice_data is None:
            continue
        if hasattr(indice_data, "indice_pair_num") and torch.is_tensor(indice_data.indice_pair_num):
            return int(indice_data.indice_pair_num.sum().item())
        if hasattr(indice_data, "pair_fwd") and torch.is_tensor(indice_data.pair_fwd):
            return int((indice_data.pair_fwd >= 0).sum().item())
    return None


def estimate_conv_flops(module, inputs, output):
    out_tensor = extract_feature_tensor(output)
    if out_tensor is None:
        return 0.0
    if not hasattr(module, "in_channels") or not hasattr(module, "out_channels"):
        return 0.0

    groups = max(int(getattr(module, "groups", 1)), 1)
    out_channels = max(int(getattr(module, "out_channels", 1)), 1)
    in_channels = int(getattr(module, "in_channels", 1))
    bias_flops = 0.0

    if hasattr(output, "features"):
        active_outputs = int(out_tensor.shape[0])
        pair_count = estimate_sparse_pair_count(module, inputs, output)
        if pair_count is None:
            pair_count = active_outputs * _kernel_volume(module)
        if getattr(module, "bias", None) is not None:
            bias_flops = active_outputs * out_channels
        return float(2.0 * pair_count * out_channels * (in_channels / groups) + bias_flops)

    active_outputs = int(out_tensor.numel() // out_channels)
    pair_count = active_outputs
    if not isinstance(module, nn.Linear):
        pair_count *= _kernel_volume(module)
    if getattr(module, "bias", None) is not None:
        bias_flops = active_outputs * out_channels
    return float(2.0 * pair_count * out_channels * (in_channels / groups) + bias_flops)


def estimate_attention_flops(module, inputs):
    if not inputs:
        return 0.0
    x = inputs[0]
    if not hasattr(x, "features"):
        return 0.0
    features = x.features
    if features.dim() != 2:
        return 0.0
    num_points, channels = features.shape
    kernel_size = int(getattr(module, "k", 1))
    k2 = kernel_size * kernel_size
    flops = 0.0
    flops += 4.0 * num_points * channels
    flops += 4.0 * num_points * k2 * channels
    flops += 8.0 * num_points * k2
    flops += max(k2 - 1, 0) * num_points
    flops += 3.0 * num_points * channels
    flops += 6.0 * num_points
    flops += 3.0 * num_points * channels
    return float(flops)


def estimate_module_flops(module, inputs, output):
    class_name = module.__class__.__name__
    if "SparseSymmetricCosineAttention" in class_name:
        return estimate_attention_flops(module, inputs)
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        out_tensor = extract_feature_tensor(output)
        return float(2.0 * out_tensor.numel()) if out_tensor is not None else 0.0
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
        out_tensor = extract_feature_tensor(output)
        return float(out_tensor.numel()) if out_tensor is not None else 0.0
    if isinstance(module, nn.Sigmoid):
        out_tensor = extract_feature_tensor(output)
        return estimate_sigmoid_flops(out_tensor) if out_tensor is not None else 0.0
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return estimate_conv_flops(module, inputs, output)
    if SPARSE_CONV_TYPES and isinstance(module, SPARSE_CONV_TYPES):
        return estimate_conv_flops(module, inputs, output)
    if "Conv" in class_name and hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        return estimate_conv_flops(module, inputs, output)
    return 0.0


def _should_profile_module(module):
    if "SparseSymmetricCosineAttention" in module.__class__.__name__:
        return True
    return len(list(module.children())) == 0


def profile_model(model, inputs, module_classifier=None, module_filter=None):
    flops_by_group = defaultdict(float)
    total_flops = 0.0
    handles = []

    def _make_hook(module_name):
        def _hook(module, module_inputs, module_output):
            nonlocal total_flops
            flops = estimate_module_flops(module, module_inputs, module_output)
            if flops <= 0:
                return
            total_flops += float(flops)
            if module_classifier is None:
                group_name = module_name
            else:
                group_name = module_classifier(module_name, module)
                if group_name is None:
                    return
            flops_by_group[group_name] += float(flops)
        return _hook

    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        if not _should_profile_module(module):
            continue
        if module_filter is not None and not module_filter(module_name, module):
            continue
        handles.append(module.register_forward_hook(_make_hook(module_name)))

    was_training = model.training
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return float(total_flops), count_parameters(model), dict(flops_by_group)
