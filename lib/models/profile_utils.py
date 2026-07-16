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


def _iter_sparse_tensors(obj):
    if hasattr(obj, "features") and hasattr(obj, "indices"):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_sparse_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_sparse_tensors(item)


def _count_valid_pair_tensor(tensor, allow_2d=True):
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None
    # spconv indice_pairs is usually [2, kernel_volume, max_pairs].
    # Count one direction only; counting both directions doubles the MACs.
    if tensor.dim() >= 3 and int(tensor.shape[0]) == 2:
        return int((tensor[0] >= 0).sum().item())
    # pair_fwd / pair_bwd are usually [kernel_volume, max_pairs].
    if allow_2d and tensor.dim() >= 2:
        return int((tensor >= 0).sum().item())
    return None


def _extract_indice_pair_count(indice_data):
    if indice_data is None:
        return None
    if isinstance(indice_data, dict):
        for key in ("indice_pair_num", "pair_fwd", "indice_pairs", "pair_bwd"):
            count = _extract_indice_pair_count(indice_data.get(key))
            if count is not None:
                return count
        return None
    if hasattr(indice_data, "indice_pair_num") and torch.is_tensor(indice_data.indice_pair_num):
        return int(indice_data.indice_pair_num.clamp_min(0).sum().item())
    for attr in ("pair_fwd", "indice_pairs", "pair_bwd"):
        value = getattr(indice_data, attr, None)
        count = _count_valid_pair_tensor(value)
        if count is not None:
            return count
    if isinstance(indice_data, (tuple, list)):
        # Prefer explicit pair tensors over 1-D pair_num tensors to avoid mistaking coords for pairs.
        for item in indice_data:
            count = _count_valid_pair_tensor(item, allow_2d=False)
            if count is not None:
                return count
        for item in indice_data:
            if torch.is_tensor(item) and item.dim() == 1:
                return int(item.clamp_min(0).sum().item())
    return _count_valid_pair_tensor(indice_data)


def estimate_sparse_pair_count(module, inputs, output):
    indice_key = getattr(module, "indice_key", None)
    sparse_tensors = list(_iter_sparse_tensors(output))
    for item in inputs:
        sparse_tensors.extend(_iter_sparse_tensors(item))

    for sparse_tensor in sparse_tensors:
        indice_data = None
        if indice_key is not None and hasattr(sparse_tensor, "find_indice_pair"):
            try:
                indice_data = sparse_tensor.find_indice_pair(indice_key)
            except Exception:
                indice_data = None
        count = _extract_indice_pair_count(indice_data)
        if count is not None:
            return count

        indice_dict = getattr(sparse_tensor, "indice_dict", None)
        if isinstance(indice_dict, dict):
            candidates = []
            if indice_key is not None and indice_key in indice_dict:
                candidates.append(indice_dict[indice_key])
            elif indice_key is None and len(indice_dict) == 1:
                candidates.extend(indice_dict.values())
            for candidate in candidates:
                count = _extract_indice_pair_count(candidate)
                if count is not None:
                    return count
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


def _attention_score_valid_count(output, default_count):
    score = None
    if isinstance(output, (tuple, list)) and len(output) > 1 and torch.is_tensor(output[1]):
        score = output[1]
    if score is None:
        return int(default_count)
    try:
        return int((score.detach() > 0).sum().item())
    except Exception:
        return int(default_count)


def _linear_flops(in_features, out_features, active_outputs, bias=False):
    flops = 2.0 * float(active_outputs) * float(in_features) * float(out_features)
    if bias:
        flops += float(active_outputs) * float(out_features)
    return flops


def estimate_triplet_motion_attention_flops(module, inputs, output=None):
    if not inputs:
        return 0.0
    x = inputs[0]
    if not hasattr(x, "features"):
        return 0.0
    features = x.features
    if features.dim() != 2:
        return 0.0
    num_points, channels = features.shape
    num_points = int(num_points)
    channels = int(channels)
    if num_points == 0 or channels == 0:
        return 0.0

    valid_points = int(getattr(module, "_profile_last_valid_points", -1))
    if valid_points < 0:
        valid_points = _attention_score_valid_count(output, num_points)
    valid_points = max(valid_points, 0)

    topk = max(int(getattr(module, "topk", getattr(module, "k", 1))), 1)
    pair_slots = valid_points * topk * topk
    one_side_slots = valid_points * topk
    window_offsets = getattr(module, "window_offsets", None)
    window_size = int(window_offsets.shape[0]) if torch.is_tensor(window_offsets) else topk

    flops = 0.0
    # LayerNorm over all active sparse points and feature-distance top-k over sparse windows.
    flops += 5.0 * num_points * channels
    flops += 2.0 * num_points * window_size * (3.0 * channels + 1.0)

    # Attention projections; invalid padded top-k slots are still projected in current forward code.
    flops += _linear_flops(channels, channels, valid_points, bias=False)  # q
    flops += _linear_flops(channels, channels, valid_points, bias=False)  # cur value
    flops += _linear_flops(channels, channels, one_side_slots, bias=False)  # prev value
    flops += _linear_flops(channels, channels, one_side_slots, bias=False)  # next value

    if getattr(module, "k_laiyuan", "coords") == "features":
        flops += _linear_flops(channels, channels, valid_points, bias=False)
        flops += _linear_flops(channels, channels, one_side_slots, bias=False)
        flops += _linear_flops(channels, channels, one_side_slots, bias=False)
        flops += 2.0 * pair_slots * channels
    else:
        traj_mlp = getattr(module, "traj_mlp", None)
        if isinstance(traj_mlp, nn.Sequential) and len(traj_mlp) >= 3:
            hidden = int(getattr(traj_mlp[0], "out_features", channels))
            out_features = int(getattr(traj_mlp[-1], "out_features", channels))
            flops += _linear_flops(6, hidden, pair_slots, bias=getattr(traj_mlp[0], "bias", None) is not None)
            flops += float(pair_slots) * hidden
            flops += _linear_flops(hidden, out_features, pair_slots, bias=getattr(traj_mlp[-1], "bias", None) is not None)
        else:
            flops += _linear_flops(6, channels, pair_slots, bias=True)
        flops += 2.0 * pair_slots * channels

    # Masked softmax, marginalization, value aggregation, output projection, FFN residual.
    flops += 8.0 * pair_slots
    flops += 2.0 * pair_slots
    if getattr(module, "wbianyuanhua", True):
        flops += 4.0 * one_side_slots * channels
    else:
        flops += 4.0 * pair_slots * channels
    flops += _linear_flops(channels, channels, valid_points, bias=False)
    if getattr(module, "useFFN", True):
        ffn = getattr(module, "ffn", None)
        hidden = None
        if isinstance(ffn, nn.Sequential) and len(ffn) >= 3:
            hidden = int(getattr(ffn[0], "out_features", 0))
        if hidden is None or hidden <= 0:
            hidden = max(4, channels * 4)
        flops += 5.0 * valid_points * channels
        flops += _linear_flops(channels, hidden, valid_points, bias=True)
        flops += float(valid_points) * hidden
        flops += _linear_flops(hidden, channels, valid_points, bias=True)
    return float(flops)


def estimate_sparse_symmetric_attention_flops(module, inputs, output=None):
    if not inputs:
        return 0.0
    x = inputs[0]
    if not hasattr(x, "features"):
        return 0.0
    features = x.features
    if features.dim() != 2:
        return 0.0
    num_points, channels = features.shape
    num_points = int(num_points)
    channels = int(channels)
    if num_points == 0 or channels == 0:
        return 0.0
    valid_points = _attention_score_valid_count(output, num_points)
    offsets = getattr(module, "spatial_offsets", None)
    if torch.is_tensor(offsets):
        k2 = int(offsets.numel())
    else:
        kernel_size = int(getattr(module, "k", 1))
        k2 = kernel_size * kernel_size
    flops = 0.0
    flops += 2.0 * num_points * k2  # search/mask bookkeeping
    flops += 4.0 * num_points * channels  # normalization/intensity statistics
    flops += 4.0 * valid_points * k2 * channels  # cosine on real sparse hits (estimated from nonzero scores)
    flops += 8.0 * num_points * k2
    flops += max(k2 - 1, 0) * num_points
    flops += 6.0 * num_points
    flops += 3.0 * num_points * channels
    return float(flops)


def estimate_attention_flops(module, inputs, output=None):
    class_name = module.__class__.__name__
    if "TripletMotionConsistency" in class_name:
        return estimate_triplet_motion_attention_flops(module, inputs, output=output)
    return estimate_sparse_symmetric_attention_flops(module, inputs, output=output)


def estimate_module_flops(module, inputs, output):
    class_name = module.__class__.__name__
    if "SparseSymmetricCosineAttention" in class_name or "TripletMotionConsistency" in class_name:
        return estimate_attention_flops(module, inputs, output=output)
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


def _is_attention_profile_module(module):
    class_name = module.__class__.__name__
    return "SparseSymmetricCosineAttention" in class_name or "TripletMotionConsistency" in class_name


def _should_profile_module(module):
    if _is_attention_profile_module(module):
        return True
    return len(list(module.children())) == 0


def _is_attention_descendant_to_skip(module_name, attention_prefixes):
    for prefix in attention_prefixes:
        if not prefix or not module_name.startswith(prefix + "."):
            continue
        suffix = module_name[len(prefix) + 1:]
        return not (suffix == "conv" or suffix.startswith("conv."))
    return False


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

    attention_prefixes = [
        module_name for module_name, module in model.named_modules()
        if _is_attention_profile_module(module)
    ]

    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        if _is_attention_descendant_to_skip(module_name, attention_prefixes):
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
