from functools import partial
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_cur = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_cur, 'path_setup.py')):
    _cur = os.path.dirname(_cur)
if _cur not in sys.path:
    sys.path.insert(0, _cur)

from lib.models.spconv_backbone import post_act_block


class SparseSymmetricCosineAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=5, alpha=0.5, stride=1,
                 indice_key="subm0", use_qkv=False, use_biqkv=False,
                 use_maxpool=False, temporal_dilation=2, conv=None,
                 exclude_center=True):
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
        self.alpha = alpha
        self.temporal_dilation = temporal_dilation
        self.exclude_center = exclude_center
        self.conv = conv if conv is not None else post_act_block(
            in_channels, in_channels, (1, 3, 3),
            norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            padding=(0, 1, 1),
            indice_key=indice_key,
        )

        self.scale_y = 4096
        self.scale_t = 4096 * 4096
        self.scale_b = 4096 * 4096 * 100

        dy = torch.arange(-self.r, self.r + 1)
        dx = torch.arange(-self.r, self.r + 1)
        mesh_y, mesh_x = torch.meshgrid(dy, dx, indexing='ij')
        offsets_y = mesh_y.reshape(-1)
        offsets_x = mesh_x.reshape(-1)
        if self.exclude_center:
            valid_offsets = (offsets_y != 0) | (offsets_x != 0)
            offsets_y = offsets_y[valid_offsets]
            offsets_x = offsets_x[valid_offsets]

        spatial_offsets = offsets_y * self.scale_y + offsets_x
        norm = max(float(self.r), 1.0)
        offsets_y_f = offsets_y.to(torch.float32) / norm
        offsets_x_f = offsets_x.to(torch.float32) / norm
        offsets_d_f = torch.sqrt(offsets_y_f.pow(2) + offsets_x_f.pow(2))

        self.register_buffer('spatial_offsets', spatial_offsets)
        self.register_buffer('offsets_y_f', offsets_y_f)
        self.register_buffer('offsets_x_f', offsets_x_f)
        self.register_buffer('offsets_d_f', offsets_d_f)

        edge_hidden = max(8, min(32, in_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, edge_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(edge_hidden, 1),
        )
        self.value_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.update_proj = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        self.gate_proj = nn.Linear(in_channels * 2, in_channels)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _collect_edges(self, sorted_keys, sorted_features, sorted_features_norm,
                       features_norm, center_rows, ptr, mask_flat, direction):
        if not mask_flat.any():
            return None

        valid = mask_flat.nonzero(as_tuple=False).squeeze(1)
        row_idx = center_rows[valid]
        offset_idx = valid % self.spatial_offsets.numel()
        neighbor_features = sorted_features[ptr[valid]]
        neighbor_features_norm = sorted_features_norm[ptr[valid]]
        sim = (features_norm[row_idx] * neighbor_features_norm).sum(dim=1).clamp_min_(0)

        direction_col = sim.new_full((valid.numel(), 1), float(direction))
        edge_attr = torch.stack([
            sim,
            self.offsets_y_f[offset_idx].to(sim.dtype),
            self.offsets_x_f[offset_idx].to(sim.dtype),
            self.offsets_d_f[offset_idx].to(sim.dtype),
            direction_col.squeeze(1),
        ], dim=1)
        return row_idx, neighbor_features, edge_attr, sim

    def forward(self, x):
        features_ori = x.features
        x = self.conv(x)
        indices, features = x.indices, x.features
        n_points, channels = features.shape
        k2 = self.spatial_offsets.numel()

        b = indices[:, 0].long()
        t = indices[:, 1].long()
        y = indices[:, 2].long()
        x_coord = indices[:, 3].long()
        current_keys = b * self.scale_b + t * self.scale_t + y * self.scale_y + x_coord
        sorted_keys, sort_idx = torch.sort(current_keys)

        features_norm = F.normalize(features, p=2, dim=1)
        sorted_features = features[sort_idx]
        sorted_features_norm = features_norm[sort_idx]

        time_offset = self.temporal_dilation * self.scale_t
        keys_col = current_keys.unsqueeze(1)
        keys_prev = keys_col - time_offset - self.spatial_offsets.unsqueeze(0)
        keys_next = keys_col + time_offset + self.spatial_offsets.unsqueeze(0)

        keys_prev_flat = keys_prev.reshape(-1)
        keys_next_flat = keys_next.reshape(-1)
        ptr_prev = torch.searchsorted(sorted_keys, keys_prev_flat).clamp(max=n_points - 1)
        ptr_next = torch.searchsorted(sorted_keys, keys_next_flat).clamp(max=n_points - 1)
        mask_prev_flat = sorted_keys[ptr_prev] == keys_prev_flat
        mask_next_flat = sorted_keys[ptr_next] == keys_next_flat

        center_rows = torch.arange(n_points, device=features.device).unsqueeze(1).expand(n_points, k2).reshape(-1)
        prev_edges = self._collect_edges(
            sorted_keys, sorted_features, sorted_features_norm,
            features_norm, center_rows, ptr_prev, mask_prev_flat, -1,
        )
        next_edges = self._collect_edges(
            sorted_keys, sorted_features, sorted_features_norm,
            features_norm, center_rows, ptr_next, mask_next_flat, 1,
        )

        edge_parts = [edge for edge in (prev_edges, next_edges) if edge is not None]
        context = features.new_zeros((n_points, channels))
        score = features.new_zeros((n_points, 1))
        has_neighbor = features.new_zeros((n_points, 1))

        if edge_parts:
            row_idx = torch.cat([edge[0] for edge in edge_parts], dim=0)
            neighbor_features = torch.cat([edge[1] for edge in edge_parts], dim=0)
            edge_attr = torch.cat([edge[2] for edge in edge_parts], dim=0)

            edge_weight = torch.sigmoid(self.edge_mlp(edge_attr))
            value = self.value_proj(neighbor_features)

            weighted_sum = features.new_zeros((n_points, channels))
            weight_sum = features.new_zeros((n_points, 1))
            edge_count = features.new_zeros((n_points, 1))
            weighted_sum.index_add_(0, row_idx, value * edge_weight)
            weight_sum.index_add_(0, row_idx, edge_weight)
            edge_count.index_add_(0, row_idx, torch.ones_like(edge_weight))

            context = weighted_sum / weight_sum.clamp_min(1e-6)
            has_neighbor = (edge_count > 0).to(features.dtype)
            score = (weight_sum / edge_count.clamp_min(1.0)).clamp(0.0, 1.0)

        update_input = torch.cat([features, context], dim=1)
        gate = torch.sigmoid(self.gate_proj(update_input)) * has_neighbor
        update = self.update_proj(update_input)
        enhanced_features = features_ori + self.alpha * gate * update
        return enhanced_features, score * has_neighbor
