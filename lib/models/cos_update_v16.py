import torch
import torch.nn as nn


class TripletMotionConsistencySparseConv(nn.Module):
    """Triplet motion consistency update for sparse spatio-temporal points.

    The module keeps sparse coordinates unchanged and updates only point
    features. Neighbor search is full-frame top-k within the same batch and
    adjacent temporal frame. Points without both previous and next neighbors
    receive an exact identity output.
    """

    def __init__(
        self,
        in_channels,
        kernel_size=5,
        alpha=0.5,
        temporal_dilation=1,
        conv=None,
        topk=3,
        hidden_ratio=0.5,
        pos_hidden=16,
        pos_scale=16.0,
        chunk_size=1024,
        valid_norm=True,
        **kwargs,
    ):
        super().__init__()
        del kernel_size, kwargs
        self.in_channels = in_channels
        self.alpha = alpha
        self.temporal_dilation = int(temporal_dilation)
        self.topk = max(1, int(topk))
        self.pos_scale = float(pos_scale)
        self.chunk_size = max(1, int(chunk_size))
        self.valid_norm = bool(valid_norm)
        self.conv = conv if conv is not None else nn.Identity()

        hidden_channels = max(4, int(round(in_channels * float(hidden_ratio))))
        pos_hidden = max(4, int(pos_hidden))

        self.prev_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.cur_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.next_proj = nn.Linear(in_channels, in_channels, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pos_hidden, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.bn = nn.BatchNorm1d(in_channels, eps=1e-3, momentum=0.01)
        self.act = nn.ReLU(inplace=True)

        # Start with neutral relative-position modulation: 2 * sigmoid(0) = 1.
        nn.init.zeros_(self.pos_mlp[-1].weight)
        nn.init.zeros_(self.pos_mlp[-1].bias)

    def _build_frame_groups(self, indices):
        b = indices[:, 0].long()
        t = indices[:, 1].long()
        frame_stride = int(t.max().item()) + self.temporal_dilation + 2
        frame_keys = b * frame_stride + t
        unique_keys = torch.unique(frame_keys, sorted=False)
        groups = {}
        for key in unique_keys.detach().cpu().tolist():
            key_tensor = frame_keys.new_tensor(key)
            groups[int(key)] = (frame_keys == key_tensor).nonzero(as_tuple=False).squeeze(1)
        return groups, frame_stride

    def _fill_topk(self, query_idx, cand_idx, spatial, out_idx, out_mask):
        if query_idx.numel() == 0 or cand_idx.numel() == 0:
            return

        k_eff = min(self.topk, int(cand_idx.numel()))
        query_xy = spatial[query_idx].float()
        cand_xy = spatial[cand_idx].float()

        for start in range(0, query_idx.numel(), self.chunk_size):
            end = min(start + self.chunk_size, query_idx.numel())
            dist = torch.cdist(query_xy[start:end], cand_xy, p=2)
            _, top_pos = torch.topk(dist, k=k_eff, dim=1, largest=False)
            rows = query_idx[start:end]
            out_idx[rows, :k_eff] = cand_idx[top_pos]
            out_mask[rows, :k_eff] = True

    def _find_triplet_neighbors(self, indices):
        n_points = indices.shape[0]
        device = indices.device
        idx_prev = torch.zeros(n_points, self.topk, dtype=torch.long, device=device)
        idx_next = torch.zeros(n_points, self.topk, dtype=torch.long, device=device)
        mask_prev = torch.zeros(n_points, self.topk, dtype=torch.bool, device=device)
        mask_next = torch.zeros(n_points, self.topk, dtype=torch.bool, device=device)

        groups, _ = self._build_frame_groups(indices)
        spatial = indices[:, 2:4]

        for key, query_idx in groups.items():
            prev_idx = groups.get(key - self.temporal_dilation)
            next_idx = groups.get(key + self.temporal_dilation)
            if prev_idx is not None:
                self._fill_topk(query_idx, prev_idx, spatial, idx_prev, mask_prev)
            if next_idx is not None:
                self._fill_topk(query_idx, next_idx, spatial, idx_next, mask_next)

        return idx_prev, idx_next, mask_prev, mask_next

    def forward(self, x):
        input_features = x.features
        if input_features.numel() == 0:
            return input_features, input_features.new_zeros((input_features.shape[0], 1))

        x_conv = self.conv(x)
        indices = x_conv.indices
        features = x_conv.features
        n_points, channels = features.shape

        idx_prev, idx_next, mask_prev, mask_next = self._find_triplet_neighbors(indices)
        mask_pair = mask_prev.unsqueeze(2) & mask_next.unsqueeze(1)
        has_pair = mask_pair.flatten(1).any(dim=1)
        if not has_pair.any():
            return input_features, input_features.new_zeros((n_points, 1))

        prev_feat = self.prev_proj(features)[idx_prev]
        cur_feat = self.cur_proj(features).view(n_points, 1, 1, channels)
        next_feat = self.next_proj(features)[idx_next]
        triplet_feat = prev_feat.unsqueeze(2) + cur_feat + next_feat.unsqueeze(1)

        triplet_msg = self.ffn(triplet_feat.reshape(-1, channels)).view(
            n_points, self.topk, self.topk, channels
        )

        spatial = indices[:, 2:4].to(features.dtype)
        s_cur = spatial.view(n_points, 1, 1, 2)
        s_prev = spatial[idx_prev].unsqueeze(2)
        s_next = spatial[idx_next].unsqueeze(1)
        s_minus = (s_cur - s_prev).expand(-1, -1, self.topk, -1)
        s_plus = (s_next - s_cur).expand(-1, self.topk, -1, -1)
        accel = s_plus - s_minus
        motion_input = torch.cat([s_minus, s_plus, accel], dim=-1) / max(self.pos_scale, 1e-6)

        pos_score = self.pos_mlp(motion_input.reshape(-1, 6)).view(n_points, self.topk, self.topk, 1)
        pos_gate = 2.0 * torch.sigmoid(pos_score)
        pos_gate = pos_gate * mask_pair.unsqueeze(-1).to(pos_gate.dtype)

        mod_msg = triplet_msg * pos_gate
        msg_sum = mod_msg.sum(dim=2).sum(dim=1)
        if self.valid_norm:
            denom = mask_pair.flatten(1).sum(dim=1).to(features.dtype).clamp_min(1.0).unsqueeze(1)
        else:
            denom = features.new_full((n_points, 1), float(self.topk * self.topk))
        update = msg_sum / denom
        update = self.act(self.bn(update))
        update = update * has_pair.to(update.dtype).unsqueeze(1)

        score = pos_gate.sum(dim=2).sum(dim=1) / denom.to(pos_gate.dtype)
        score = score * has_pair.to(score.dtype).unsqueeze(1)
        return input_features + self.alpha * update, score
