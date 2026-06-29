from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import csv
import os
import time
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from types import MethodType

import numpy as np
import torch

from lib.dataset.dataset_factory import get_dataset
from lib.models.stNet import get_det_net, load_model
from lib.models.cos_update_v16 import TripletMotionConsistencySparseConv
from lib.test_utils.process_img_dets import preprocess, post_process
from lib.utils1.decode import ctdet_decode
from lib.utils1.opts import opts


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value):
        self.total += float(value)
        self.count += 1

    @property
    def avg(self):
        return self.total / self.count if self.count else 0.0


class Profiler:
    def __init__(self, device):
        self.device = device
        self.meters = defaultdict(AverageMeter)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def time_block(self, name):
        return _TimedBlock(self, name)

    def add(self, name, seconds):
        self.meters[name].add(seconds)

    def rows(self, denominator_name="patch_total"):
        denom = self.meters[denominator_name].avg
        rows = []
        for name in sorted(self.meters):
            avg = self.meters[name].avg
            pct = (avg / denom * 100.0) if denom > 0 else 0.0
            rows.append((name, avg, avg / 10.0, pct, self.meters[name].count))
        return rows


class _TimedBlock:
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler.sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.profiler.sync()
        self.profiler.add(self.name, time.perf_counter() - self.start)


def parse_opt_txt(path):
    values = {}
    with open(path, "r") as f:
        in_opt = False
        for raw_line in f:
            line = raw_line.rstrip()
            if line.strip() == "==> Opt:":
                in_opt = True
                continue
            if not in_opt or not line.startswith("  "):
                continue
            key, sep, value = line.strip().partition(":")
            if sep:
                values[key] = value.strip()
    return values


def opt_args_from_txt(opt_txt, load_model):
    values = parse_opt_txt(opt_txt)
    # Avoid argparse bool fields: passing the string "False" would become True.
    # List defaults such as feat_channels are left to opts.py as well.
    keys = [
        "discribe", "task", "exp_name", "layers", "model_name", "gpus",
        "num_workers", "seed", "lr_step", "num_epochs", "seqLen", "thresh",
        "K", "save_dir", "data_mode", "datasetname", "sup_mode",
        "unsup_iter", "conf_filtered", "hm_weight", "wh_weight",
        "off_weight", "hm_large_heatmap_weight", "stage1_epochs",
        "groups", "downsample_mode", "net1name", "use_tzsconv", "MFE",
        "MFE_skip", "tmc_topk", "tmc_window_size", "tmc_hidden_ratio",
        "tmc_pos_hidden", "tmc_pos_scale", "tmc_chunk_size", "vis_mode",
        "vis_max_frames",
    ]
    args = []
    for key in keys:
        if key in values:
            value = values[key]
            if key in ("gpus", "lr_step"):
                value = value.strip().strip("[]").replace(" ", "")
            args.extend(["--" + key, value])
    args.extend(["--load_model", load_model])
    return args


def patch_full_frame_topk(profiler):
    def full_frame_fill_topk(self, query_idx, cand_idx, spatial, out_idx, out_mask):
        if query_idx.numel() == 0 or cand_idx.numel() == 0:
            return

        k_eff = min(self.topk, int(cand_idx.numel()))
        query_xy = spatial[query_idx].float()
        cand_xy = spatial[cand_idx].float()
        if self.chunk_size > 0:
            step = self.chunk_size
        else:
            step = max(1, self.auto_pair_limit // max(1, int(cand_idx.numel())))
            step = min(step, int(query_idx.numel()))
        cand_y = cand_xy[:, 0].unsqueeze(0)
        cand_x = cand_xy[:, 1].unsqueeze(0)

        with profiler.time_block("mfe_full_frame_topk_search"):
            for start in range(0, query_idx.numel(), step):
                end = min(start + step, query_idx.numel())
                q_xy = query_xy[start:end]
                dy = q_xy[:, 0].unsqueeze(1) - cand_y
                dx = q_xy[:, 1].unsqueeze(1) - cand_x
                dist2 = dy.square() + dx.square()
                _, top_pos = torch.topk(dist2, k=k_eff, dim=1, largest=False)
                rows = query_idx[start:end]
                out_idx[rows, :k_eff] = cand_idx[top_pos]
                out_mask[rows, :k_eff] = True

    TripletMotionConsistencySparseConv._fill_topk = full_frame_fill_topk


def patch_mfe_detail_forward(profiler):
    def timed_forward(self, x):
        with profiler.time_block("mfe3_input_features"):
            input_features = x.features
            if input_features.numel() == 0:
                return input_features, input_features.new_zeros((input_features.shape[0], 1))

        with profiler.time_block("mfe3_pre_conv"):
            x_conv = self.conv(x)
            indices = x_conv.indices
            features = x_conv.features
            n_points, channels = features.shape

        with profiler.time_block("mfe3_neighbor_search_total"):
            idx_prev, idx_next, mask_prev, mask_next = self._find_triplet_neighbors(indices)

        with profiler.time_block("mfe3_pair_mask"):
            mask_pair = mask_prev.unsqueeze(2) & mask_next.unsqueeze(1)
            has_pair = mask_pair.flatten(1).any(dim=1)
            if not has_pair.any():
                return input_features, input_features.new_zeros((n_points, 1))

        with profiler.time_block("mfe3_projection_gather"):
            prev_feat = self.prev_proj(features)[idx_prev]
            cur_feat = self.cur_proj(features).view(n_points, 1, 1, channels)
            next_feat = self.next_proj(features)[idx_next]

        with profiler.time_block("mfe3_triplet_feature_build"):
            triplet_feat = prev_feat.unsqueeze(2) + cur_feat + next_feat.unsqueeze(1)

        with profiler.time_block("mfe3_triplet_ffn"):
            triplet_msg = self.ffn(triplet_feat.reshape(-1, channels)).view(
                n_points, self.topk, self.topk, channels
            )

        with profiler.time_block("mfe3_motion_input_build"):
            spatial = indices[:, 2:4].to(features.dtype)
            s_cur = spatial.view(n_points, 1, 1, 2)
            s_prev = spatial[idx_prev].unsqueeze(2)
            s_next = spatial[idx_next].unsqueeze(1)
            s_minus = (s_cur - s_prev).expand(-1, -1, self.topk, -1)
            s_plus = (s_next - s_cur).expand(-1, self.topk, -1, -1)
            accel = s_plus - s_minus
            motion_input = torch.cat([s_minus, s_plus, accel], dim=-1) / max(self.pos_scale, 1e-6)

        with profiler.time_block("mfe3_pos_mlp_gate"):
            pos_score = self.pos_mlp(motion_input.reshape(-1, 6)).view(n_points, self.topk, self.topk, 1)
            pos_gate = 2.0 * torch.sigmoid(pos_score)
            pos_gate = pos_gate * mask_pair.unsqueeze(-1).to(pos_gate.dtype)

        with profiler.time_block("mfe3_message_weight_sum"):
            mod_msg = triplet_msg * pos_gate
            msg_sum = mod_msg.sum(dim=2).sum(dim=1)

        with profiler.time_block("mfe3_denom_norm"):
            if self.valid_norm:
                denom = mask_pair.flatten(1).sum(dim=1).to(features.dtype).clamp_min(1.0).unsqueeze(1)
            else:
                denom = features.new_full((n_points, 1), float(self.topk * self.topk))
            update = msg_sum / denom
            update = self.act(self.bn(update))
            update = update * has_pair.to(update.dtype).unsqueeze(1)

        with profiler.time_block("mfe3_score_and_output"):
            score = pos_gate.sum(dim=2).sum(dim=1) / denom.to(pos_gate.dtype)
            score = score * has_pair.to(score.dtype).unsqueeze(1)
            return input_features + self.alpha * update, score

    TripletMotionConsistencySparseConv.forward = timed_forward


def add_forward_hooks(model, profiler):
    handles = []

    def timed_hook(name):
        def pre_hook(module, inputs):
            profiler.sync()
            module._profile_start = time.perf_counter()

        def post_hook(module, inputs, output):
            profiler.sync()
            start = getattr(module, "_profile_start", None)
            if start is not None:
                profiler.add(name, time.perf_counter() - start)

        return pre_hook, post_hook

    for name, module in [
        ("model_i2pnet", model.I2PNet),
        ("model_sparse_backbone", model.sp_backbone),
    ]:
        pre_hook, post_hook = timed_hook(name)
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    for head in model.heads:
        pre_hook, post_hook = timed_hook("model_detection_heads")
        head_module = getattr(model, head)
        handles.append(head_module.register_forward_pre_hook(pre_hook))
        handles.append(head_module.register_forward_hook(post_hook))

    for module_name, module in model.named_modules():
        if isinstance(module, TripletMotionConsistencySparseConv):
            pre_hook, post_hook = timed_hook("mfe_" + module_name.split(".")[-1])
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))

    return handles


def process_profiled(model, input_batch, opt, profiler):
    with torch.no_grad():
        with profiler.time_block("model_forward_total"):
            output = model(input_batch)[-1]

        hm = output["hm"]
        wh = output["wh"]
        reg = output["reg"] if opt.off_flag else None

        with profiler.time_block("decode_topk"):
            if reg is not None:
                dets = ctdet_decode(
                    hm[0].transpose(0, 1),
                    wh[0].transpose(0, 1),
                    reg=reg[0].transpose(0, 1),
                    K=opt.K,
                )
            else:
                dets = ctdet_decode(
                    hm[0].transpose(0, 1),
                    wh[0].transpose(0, 1),
                    reg=None,
                    K=opt.K,
                )
    return output, dets


def get_test_root(opt):
    if opt.datasetname in ("aircraft", "sdm_car", "mir"):
        return os.path.join(opt.data_dir, "images/test")
    if opt.datasetname in ("rs_car", "rs_car_new"):
        return os.path.join(opt.data_dir, "images/test1024")
    raise ValueError("Unsupported datasetname: {}".format(opt.datasetname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--opt_txt", default="")
    parser.add_argument("--max_patches", type=int, default=0)
    parser.add_argument("--warmup_patches", type=int, default=1)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--full_frame_topk", action="store_true")
    args = parser.parse_args(ORIGINAL_ARGV[1:])

    opt_txt = args.opt_txt or os.path.join(os.path.dirname(args.weight), "opt.txt")
    opt_argv = opt_args_from_txt(opt_txt, args.weight)
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        opt = opts().parse(opt_argv)
    finally:
        sys.argv = saved_argv
    opt.device = torch.device("cuda" if opt.gpus[0] >= 0 and torch.cuda.is_available() else "cpu")
    if opt.device.type == "cuda":
        torch.cuda.set_device(opt.gpus[0])

    dataset_cls = get_dataset(opt)
    data_val = dataset_cls(opt, "test")
    heads = {"hm": data_val.num_classes, "wh": 2, "reg": 2} if opt.off_flag else {
        "hm": data_val.num_classes,
        "wh": 2,
    }
    model = get_det_net(heads, opt.model_name, data_val.resolution, opt.seqLen, opt, thresh=opt.thresh)
    model = load_model(model, args.weight).to(opt.device).eval()

    profiler = Profiler(opt.device)
    if args.full_frame_topk:
        patch_full_frame_topk(profiler)
    patch_mfe_detail_forward(profiler)
    handles = add_forward_hooks(model, profiler)

    test_root = get_test_root(opt)
    folders = [
        f for f in os.listdir(test_root)
        if not f.startswith(".") and os.path.isdir(os.path.join(test_root, f))
    ]
    folders = [f for f in folders if "." not in f]
    folders.sort()

    measured = 0
    seen = 0
    patch_len = opt.seqLen
    for folder in folders:
        img_dir = os.path.join(test_root, folder, "img1")
        img_list = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        img_list.sort()
        if len(img_list) % patch_len == 0:
            patch_num = len(img_list) // patch_len
            overlap_flag = False
        else:
            patch_num = len(img_list) // patch_len + 1
            overlap_flag = True

        for pk in range(patch_num):
            if overlap_flag and pk == patch_num - 1:
                patch_ims = img_list[len(img_list) - patch_len:len(img_list)]
            else:
                patch_ims = img_list[pk * patch_len:(pk + 1) * patch_len]
            patch_paths = [os.path.join(img_dir, f) for f in patch_ims]
            xml_paths = [p.replace("img1", opt.xmlname).rsplit(".", 1)[0] + ".xml" for p in patch_paths]

            with profiler.time_block("patch_total"):
                with profiler.time_block("preprocess_cpu"):
                    _, meta, _, input_batch = preprocess(patch_paths, data_val, xml_paths)

                with profiler.time_block("host_to_device"):
                    for key in input_batch:
                        if key != "batch_size":
                            input_batch[key] = torch.from_numpy(input_batch[key]).to(opt.device)

                output, dets = process_profiled(model, input_batch, opt, profiler)

                with profiler.time_block("postprocess_cpu"):
                    post_process(dets, meta, data_val.num_classes, max_per_image=opt.K)

            seen += 1
            if seen <= args.warmup_patches:
                for meter in profiler.meters.values():
                    meter.total = 0.0
                    meter.count = 0
                continue
            measured += 1
            if measured % 10 == 0:
                print("profiled measured patches: {}".format(measured))
            if args.max_patches and measured >= args.max_patches:
                break
        if args.max_patches and measured >= args.max_patches:
            break

    for handle in handles:
        handle.remove()

    rows = profiler.rows("patch_total")
    print("\nTiming breakdown: avg per patch, per frame assumes seqLen={}".format(opt.seqLen))
    print("{:<32s} {:>12s} {:>12s} {:>10s} {:>8s}".format(
        "step", "ms/patch", "ms/frame", "pct", "n"
    ))
    for name, avg, avg_frame, pct, count in rows:
        print("{:<32s} {:>12.3f} {:>12.3f} {:>9.2f}% {:>8d}".format(
            name, avg * 1000.0, avg_frame * 1000.0, pct, count
        ))

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "ms_per_patch", "ms_per_frame", "pct_of_patch_total", "count"])
            for name, avg, avg_frame, pct, count in rows:
                writer.writerow([name, avg * 1000.0, avg_frame * 1000.0, pct, count])
        print("saved csv: {}".format(args.output_csv))


if __name__ == "__main__":
    main()
