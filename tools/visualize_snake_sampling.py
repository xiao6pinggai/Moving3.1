from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from lib.dataset.dataset_factory import get_dataset
from lib.models.stNet import get_det_net, load_model
from lib.test_utils.process_img_dets import preprocess
from lib.utils1.opts import opts


LAYER_SPECS = {
    "skip1": {"color": (0, 220, 0), "shape": "circle"},
    "skip2": {"color": (0, 220, 220), "shape": "triangle"},
    "skip3": {"color": (255, 80, 0), "shape": "square"},
}
MISS_COLOR = (0, 0, 255)
ANCHOR_COLOR = (245, 245, 245)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize UNet3D temporal snake sampling points projected onto center frames."
    )
    parser.add_argument("--load_model", required=True, help="checkpoint .pth path")
    parser.add_argument("--datasetname", default="aircraft")
    parser.add_argument("--model_name", default="UNet3D")
    parser.add_argument("--seqLen", type=int, default=10)
    parser.add_argument("--feat_channels", default="[16,32,64,128]")
    parser.add_argument("--T_pooling", default=None)
    parser.add_argument("--downsample_mode", default="stride")
    parser.add_argument("--upsample_mode", default="deconv")
    parser.add_argument("--UNet3D_skip", default="snack", choices=["snack", "snack_unrest", "v_snack", "tconv", "tdcn"])
    parser.add_argument("--Snack_skip", default="[1,1,1]")
    parser.add_argument("--Snack_max_offset", default="[5,5,5]")
    parser.add_argument("--TKernel", default="[[5,1,1],[5,1,1],[5,1,1]]")
    parser.add_argument("--OffsetKernel", default=None)
    parser.add_argument("--VSnack_residual", type=int, default=1, choices=[0, 1])
    parser.add_argument("--TZSConv_skip", default="[0,0,0]")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--frame_interval", type=int, default=30)
    parser.add_argument("--output_dir", default=None, help="default: checkpoint directory / vis_sample")
    parser.add_argument("--layers", nargs="+", default=["skip1", "skip2", "skip3"],
                        help="layers to draw, e.g. skip1 skip3 or [skip1,skip3]")
    parser.add_argument("--point_radius", type=int, default=4)
    parser.add_argument("--anchor_point_percent", type=float, default=0.0,
                        help="central area percent of each anchor box to use as base sampling points; 0=center only, 100=full box")
    parser.add_argument("--show_neighbor_boxes", action="store_true",
                        help="draw same-track GT boxes from neighboring kernel frames")
    parser.add_argument("--hit_mode", default="corresponding",
                        choices=["corresponding", "anchor", "any_projected"],
                        help="which GT box decides whether a sampling point is inside")
    parser.add_argument("--line_thickness", type=int, default=1)
    parser.add_argument("--box_thickness", type=int, default=1)
    parser.add_argument("--anchor_thickness", type=int, default=1)
    parser.add_argument("--batch_patches", type=int, default=1)
    parser.add_argument("--max_videos", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--videos", nargs="*", default=None, help="optional video names to visualize")
    parser.add_argument("--draw_labels", action="store_true")
    parser.add_argument("--vis_sample", action="store_true")
    parser.add_argument("--vis_features", action="store_true")
    args = parser.parse_args(ORIGINAL_ARGV[1:])
    args.layers = normalize_layers(args.layers)
    return args


def normalize_layers(value):
    if isinstance(value, str):
        value = [value]
    if len(value) == 1:
        text = value[0].strip()
        if text.startswith("[") or text.startswith("("):
            value = ast.literal_eval(text)
    layers = [str(item).strip() for item in value if str(item).strip()]
    invalid = [layer for layer in layers if layer not in LAYER_SPECS]
    if invalid:
        raise ValueError("Invalid --layers values: {}. Choose from {}.".format(
            invalid, sorted(LAYER_SPECS)))
    return layers


def build_opt(args):
    opt_args = [
        "--datasetname", args.datasetname,
        "--model_name", args.model_name,
        "--load_model", args.load_model,
        "--seqLen", str(args.seqLen),
        "--feat_channels", args.feat_channels,
        "--downsample_mode", args.downsample_mode,
        "--upsample_mode", args.upsample_mode,
        "--UNet3D_skip", args.UNet3D_skip,
        "--Snack_skip", args.Snack_skip,
        "--Snack_max_offset", args.Snack_max_offset,
        "--TKernel", args.TKernel,
        "--VSnack_residual", str(args.VSnack_residual),
        "--TZSConv_skip", args.TZSConv_skip,
        "--gpus", args.gpus,
    ]
    if args.OffsetKernel is not None:
        opt_args.extend(["--OffsetKernel", args.OffsetKernel])
    if args.T_pooling is not None:
        opt_args.extend(["--T_pooling", args.T_pooling])
    opt = opts().parse(opt_args)
    opt.device = torch.device("cuda" if opt.gpus[0] >= 0 and torch.cuda.is_available() else "cpu")
    return opt


def image_video_name(image_info):
    file_name = image_info.get("file_name", "").replace("\\", "/").lstrip("./")
    for prefix in ("images/train/", "images/test/", "images/train1024/", "images/test1024/"):
        if file_name.startswith(prefix):
            file_name = file_name[len(prefix):]
            break
    parts = file_name.split("/")
    return parts[0] if parts and parts[0] else str(image_info["id"])


def xywh_to_xyxy(box):
    x, y, w, h = box[:4]
    return [float(x), float(y), float(x + w), float(y + h)]


def box_center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def point_in_box(x, y, box):
    return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def color_for_track(track_id):
    hue = int((int(track_id) * 37) % 180)
    hsv = np.uint8([[[hue, 180, 245]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_marker(img, x, y, color, shape, radius, thickness=-1):
    x_i, y_i = int(round(x)), int(round(y))
    if radius <= 0:
        if 0 <= y_i < img.shape[0] and 0 <= x_i < img.shape[1]:
            img[y_i, x_i] = color
        return
    if shape == "circle":
        cv2.circle(img, (x_i, y_i), radius, color, thickness, lineType=cv2.LINE_8)
    elif shape == "square":
        cv2.rectangle(img, (x_i - radius, y_i - radius), (x_i + radius, y_i + radius),
                      color, thickness, lineType=cv2.LINE_8)
    elif shape == "triangle":
        pts = np.array([
            [x_i, y_i - radius - 1],
            [x_i - radius - 1, y_i + radius],
            [x_i + radius + 1, y_i + radius],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_8)
    else:
        cv2.circle(img, (x_i, y_i), radius, color, thickness, lineType=cv2.LINE_8)


def draw_box(img, box, color, thickness, label=None):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_8)
    if label:
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)


def make_image_path(data_dir, file_name):
    return os.path.join(data_dir, file_name)


def build_annotation_index(dataset):
    coco_gt = dataset.coco
    images = [dict(img) for img in coco_gt.dataset["images"]]
    images_by_id = {int(img["id"]): img for img in images}
    video_images = defaultdict(list)
    anns_by_image = defaultdict(list)
    track_box = {}

    for img in images:
        img_id = int(img["id"])
        video_name = image_video_name(img)
        img["video_name"] = video_name
        video_images[video_name].append(img)

    for video_name in video_images:
        video_images[video_name].sort(key=lambda item: int(item.get("video_frame_id", item["id"])))

    for ann in coco_gt.dataset["annotations"]:
        image_id = int(ann["image_id"])
        img = images_by_id[image_id]
        track_id = int(ann.get("track_id", ann["id"]))
        frame_id = int(img.get("video_frame_id", image_id))
        box = xywh_to_xyxy(ann["bbox"])
        item = {
            "id": int(ann["id"]),
            "image_id": image_id,
            "track_id": track_id,
            "frame_id": frame_id,
            "video_name": img["video_name"],
            "box": box,
        }
        anns_by_image[image_id].append(item)
        track_box[(img["video_name"], track_id, frame_id)] = box

    return images_by_id, video_images, anns_by_image, track_box


def register_skip_hooks(model, requested_layers, capture_samples=False, capture_features=False):
    captures = {}
    handles = []
    sample_layers = []

    def unwrap(module):
        return module.module if hasattr(module, "module") else module

    root = unwrap(model)
    backbone = getattr(root, "backbone", None)
    if backbone is None:
        raise AttributeError("Model does not have backbone; cannot find UNet3D skip layers.")

    def make_snake_hook(layer_name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            with torch.no_grad():
                offset = module._make_temporal_snake_offset(x).detach()
            captures[layer_name] = {
                "input_shape": tuple(x.shape),
                "offset": offset.cpu(),
                "kernel": int(module.kernel_size[0]),
                "kernel_size": tuple(module.kernel_size),
                "offset_groups": int(module.offset_groups),
            }
        return hook

    def make_dcn_hook(layer_name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            with torch.no_grad():
                offset = module.offset_generator(x).detach()
            captures[layer_name] = {
                "input_shape": tuple(x.shape),
                "offset": offset.cpu(),
                "kernel": int(module.kernel_size[0]),
                "kernel_size": tuple(module.kernel_size),
                "offset_groups": int(module.offset_groups),
            }
        return hook

    def make_feature_hook(layer_name):
        def hook(module, inputs, output):
            capture = captures.setdefault(layer_name, {})
            capture["feature_before"] = inputs[0].detach().cpu()
            capture["feature_after"] = output.detach().cpu()
        return hook

    for layer_name in requested_layers:
        if layer_name not in LAYER_SPECS:
            print("Skip unknown layer name: {}".format(layer_name))
            continue
        skip = getattr(backbone, layer_name, None)
        if skip is None or not hasattr(skip, "block") or len(skip.block) == 0:
            print("Layer {} is not present in this model.".format(layer_name))
            continue
        layer = skip.block[0]
        if capture_samples:
            if hasattr(layer, "_make_temporal_snake_offset"):
                handles.append(layer.register_forward_hook(make_snake_hook(layer_name)))
                sample_layers.append(layer_name)
            elif getattr(layer, "offset_generator", None) is not None:
                handles.append(layer.register_forward_hook(make_dcn_hook(layer_name)))
                sample_layers.append(layer_name)
            else:
                print("Layer {} has no deformable-convolution offset generator.".format(layer_name))
        if capture_features:
            handles.append(skip.register_forward_hook(make_feature_hook(layer_name)))

    return captures, handles, sample_layers


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def patch_for_center(video_frames, center_pos, seq_len):
    center_local = seq_len // 2
    start = center_pos - center_local
    start = max(0, min(start, len(video_frames) - seq_len))
    end = start + seq_len
    if start < 0 or end > len(video_frames):
        return None, None
    return video_frames[start:end], center_pos - start


def prepare_patch(dataset, opt, frames):
    img_paths = [make_image_path(opt.data_dir, img["file_name"]) for img in frames]
    xml_paths = [
        path.replace("/img1/", "/" + opt.xmlname + "/").rsplit(".", 1)[0] + ".xml"
        for path in img_paths
    ]
    _, _, _, input_batch = preprocess(img_paths, dataset, xml_paths)
    return img_paths, input_batch


def concat_batches(patches):
    merged = {}
    keys = [key for key in patches[0].keys() if key != "batch_size"]
    for key in keys:
        merged[key] = np.concatenate([patch[key] for patch in patches], axis=0)
    return merged


def to_torch_batch(input_batch, device):
    batch = {}
    for key, value in input_batch.items():
        if key == "batch_size":
            continue
        batch[key] = torch.from_numpy(value).to(device)
    return batch


def offset_to_points(layer_capture, batch_index, local_t, center_xy, image_w, image_h):
    offset = layer_capture["offset"]
    b, _, out_d, out_h, out_w = offset.shape
    if batch_index >= b:
        return []

    kernel_size = tuple(layer_capture.get("kernel_size", (layer_capture["kernel"], 1, 1)))
    groups = layer_capture["offset_groups"]
    center_d, center_h, center_w = [size // 2 for size in kernel_size]
    local_d = int(round(local_t * (out_d - 1) / max(1, layer_capture["input_shape"][2] - 1)))
    local_d = max(0, min(out_d - 1, local_d))
    stride_x = float(image_w) / float(out_w)
    stride_y = float(image_h) / float(out_h)
    feat_x = int(round(center_xy[0] / stride_x))
    feat_y = int(round(center_xy[1] / stride_y))
    feat_x = max(0, min(out_w - 1, feat_x))
    feat_y = max(0, min(out_h - 1, feat_y))

    view = offset.view(b, groups, kernel_size[0], kernel_size[1], kernel_size[2], 3, out_d, out_h, out_w)
    layer_offset = view[batch_index, 0, :, :, :, :, local_d, feat_y, feat_x]
    points = []
    for d_idx in range(kernel_size[0]):
        for h_idx in range(kernel_size[1]):
            for w_idx in range(kernel_size[2]):
                dz = float(layer_offset[d_idx, h_idx, w_idx, 0])
                dh = float(layer_offset[d_idx, h_idx, w_idx, 1])
                dw = float(layer_offset[d_idx, h_idx, w_idx, 2])
                points.append({
                    "k_idx": d_idx,
                    "time_delta": d_idx - center_d,
                    "x": (feat_x + w_idx - center_w + dw) * stride_x,
                    "y": (feat_y + h_idx - center_h + dh) * stride_y,
                    "dw": dw,
                    "dh": dh,
                    "dz": dz,
                    "base_x": (feat_x + w_idx - center_w) * stride_x,
                    "base_y": (feat_y + h_idx - center_h) * stride_y,
                })
    return points


def target_box_for_sample(hit_mode, anchor_box, track_box, video_name, track_id, frame_id, time_delta, kernel_radius):
    if hit_mode == "anchor":
        return anchor_box
    if hit_mode == "corresponding":
        if time_delta == 0:
            return anchor_box
        return track_box.get((video_name, track_id, frame_id + time_delta))
    if hit_mode == "any_projected":
        boxes = []
        for dt in range(-kernel_radius, kernel_radius + 1):
            box = anchor_box if dt == 0 else track_box.get((video_name, track_id, frame_id + dt))
            if box is not None:
                boxes.append(box)
        return boxes
    raise ValueError("unsupported hit_mode: {}".format(hit_mode))


def is_single_box(target):
    return (
        isinstance(target, (list, tuple))
        and len(target) == 4
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in target)
    )


def point_hits_target(x, y, target):
    if is_single_box(target):
        return point_in_box(x, y, target)
    if isinstance(target, (list, tuple)):
        return any(point_in_box(x, y, box) for box in target)
    return point_in_box(x, y, target)


def format_offset_sequence(layer_name, points):
    values = []
    for point in points:
        values.append("{:+d}:({:+.1f},{:+.1f})".format(
            int(point["time_delta"]), float(point["dh"]), float(point["dw"])))
    return "{} dh,dw {}".format(layer_name, " ".join(values))


def anchor_base_points(layer_capture, anchor_box, image_w, image_h, percent):
    pct = max(0.0, min(100.0, float(percent)))
    if pct <= 0.0:
        return [box_center(anchor_box)]

    offset = layer_capture["offset"]
    _, _, _, out_h, out_w = offset.shape
    stride_x = float(image_w) / float(out_w)
    stride_y = float(image_h) / float(out_h)

    cx, cy = box_center(anchor_box)
    scale = np.sqrt(pct / 100.0)
    half_w = max(0.0, (anchor_box[2] - anchor_box[0]) * 0.5 * scale)
    half_h = max(0.0, (anchor_box[3] - anchor_box[1]) * 0.5 * scale)
    x1, x2 = cx - half_w, cx + half_w
    y1, y2 = cy - half_h, cy + half_h

    fx1 = max(0, int(np.ceil(x1 / stride_x)))
    fx2 = min(out_w - 1, int(np.floor(x2 / stride_x)))
    fy1 = max(0, int(np.ceil(y1 / stride_y)))
    fy2 = min(out_h - 1, int(np.floor(y2 / stride_y)))

    points = []
    if fx1 <= fx2 and fy1 <= fy2:
        for fy in range(fy1, fy2 + 1):
            for fx in range(fx1, fx2 + 1):
                points.append((fx * stride_x, fy * stride_y))
    if not points:
        points.append((cx, cy))
    return points


def visualize_task(task, captures, args, opt, anns_by_image, track_box):
    center_img = task["center_img"]
    center_image_id = int(center_img["id"])
    video_name = center_img["video_name"]
    frame_id = int(center_img.get("video_frame_id", center_image_id))
    img_path = make_image_path(opt.data_dir, center_img["file_name"])
    canvas = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if canvas is None:
        raise ValueError("Cannot read center image: {}".format(img_path))
    image_h, image_w = canvas.shape[:2]
    overlay = canvas.copy()

    center_anns = anns_by_image.get(center_image_id, [])
    if not center_anns:
        return None

    # Draw projected GT boxes for each center-frame track.
    kernel_radius = 0
    for capture in captures.values():
        kernel_radius = max(kernel_radius, int(capture["kernel"]) // 2)
    if kernel_radius <= 0:
        kernel_radius = 2

    for ann in center_anns:
        track_id = ann["track_id"]
        track_color = color_for_track(track_id)
        for dt in range(-kernel_radius, kernel_radius + 1):
            projected = track_box.get((video_name, track_id, frame_id + dt))
            if projected is None:
                continue
            is_center = dt == 0
            if not is_center and not args.show_neighbor_boxes:
                continue
            label = "id{} t{:+d}".format(track_id, dt) if args.draw_labels else None
            draw_box(
                overlay,
                projected,
                ANCHOR_COLOR if is_center else track_color,
                args.anchor_thickness if is_center else args.box_thickness,
                label=label if is_center else None,
            )

    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0.0, canvas)

    layer_counts = {name: {"inside": 0, "outside": 0} for name in args.layers}
    offset_texts = []

    # Draw sampling points for every center-frame GT anchor.
    for ann in center_anns:
        track_id = ann["track_id"]
        for layer_name in args.layers:
            if layer_name not in captures:
                continue
            spec = LAYER_SPECS[layer_name]
            base_points = anchor_base_points(
                captures[layer_name], ann["box"], image_w, image_h, args.anchor_point_percent)
            for base_xy in base_points:
                points = offset_to_points(
                    captures[layer_name],
                    task["batch_index"],
                    task["local_t"],
                    base_xy,
                    image_w,
                    image_h,
                )
                if len(offset_texts) < len(args.layers) and not any(text.startswith(layer_name + " ") for text in offset_texts):
                    offset_texts.append(format_offset_sequence(layer_name, points))
                prev = None
                for point in points:
                    target_box = target_box_for_sample(
                        args.hit_mode, ann["box"], track_box, video_name, track_id,
                        frame_id, point["time_delta"], kernel_radius)
                    inside = point_hits_target(point["x"], point["y"], target_box)
                    layer_counts[layer_name]["inside" if inside else "outside"] += 1
                    color = spec["color"] if inside else MISS_COLOR
                    if prev is not None and args.line_thickness > 0:
                        cv2.line(canvas,
                                 (int(round(prev["x"])), int(round(prev["y"]))),
                                 (int(round(point["x"])), int(round(point["y"]))),
                                 color, args.line_thickness, lineType=cv2.LINE_8)
                    draw_marker(canvas, point["x"], point["y"], color, spec["shape"], args.point_radius)
                    if args.draw_labels:
                        cv2.putText(
                            canvas,
                            "{}{:+d}".format(layer_name[-1], point["time_delta"]),
                            (int(round(point["x"])) + 3, int(round(point["y"])) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    prev = point

    cv2.putText(canvas, "{} frame {}".format(video_name, frame_id),
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    offset_line = " | ".join(offset_texts)
    if offset_line:
        max_chars = max(24, int(image_w / 7))
        words = offset_line.split(" ")
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        text_y = max(18, image_h - 12 - 16 * len(lines))
        for line in lines:
            cv2.putText(canvas, line, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 255, 255), 1, cv2.LINE_AA)
            text_y += 16
    cv2.putText(canvas, "hit_mode={} | skip1 green, skip2 yellow, skip3 blue, red=out".format(args.hit_mode),
                (8, image_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    video_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    out_name = "{:06d}_img{}.png".format(frame_id, center_image_id)
    out_path = os.path.join(video_dir, out_name)
    cv2.imwrite(out_path, canvas)
    return out_path, layer_counts


def feature_map_to_image(feature, batch_index, local_t, sequence_length, image_w, image_h, annotations):
    if batch_index >= feature.size(0):
        return None
    feature_d = int(round(local_t * (feature.size(2) - 1) / max(1, sequence_length - 1)))
    feature_d = max(0, min(feature.size(2) - 1, feature_d))
    feature_map = feature[batch_index, :, feature_d].mean(dim=0).numpy()
    max_value = float(feature_map.max())
    if max_value > 0.0:
        feature_map = feature_map / max_value
    else:
        feature_map = np.zeros_like(feature_map)
    feature_map = np.clip(feature_map, 0.0, 1.0)
    feature_map = cv2.resize(feature_map, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.applyColorMap(np.round(feature_map * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    for ann in annotations:
        draw_box(image, ann["box"], ANCHOR_COLOR, 1)
    return image


def visualize_feature_task(task, captures, args, opt, anns_by_image):
    if not args.vis_features:
        return []

    center_img = task["center_img"]
    center_image_id = int(center_img["id"])
    image_path = make_image_path(opt.data_dir, center_img["file_name"])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read center image: {}".format(image_path))
    image_h, image_w = image.shape[:2]
    annotations = anns_by_image.get(center_image_id, [])
    video_dir = os.path.join(args.feature_output_dir, center_img["video_name"])

    saved_paths = []
    frame_id = int(center_img.get("video_frame_id", center_image_id))
    for layer_name in args.layers:
        capture = captures.get(layer_name)
        if capture is None or "feature_before" not in capture or "feature_after" not in capture:
            continue
        for stage in ("before", "after"):
            feature_image = feature_map_to_image(
                capture["feature_" + stage],
                task["batch_index"],
                task["local_t"],
                args.seqLen,
                image_w,
                image_h,
                annotations,
            )
            if feature_image is None:
                continue
            layer_dir = os.path.join(video_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            out_path = os.path.join(
                layer_dir,
                "{:06d}_img{}_{}.png".format(frame_id, center_image_id, stage),
            )
            cv2.imwrite(out_path, feature_image)
            saved_paths.append(out_path)
    return saved_paths


def selected_frame_positions(video_frames, interval):
    positions = []
    for pos, img in enumerate(video_frames):
        frame_id = int(img.get("video_frame_id", pos + 1))
        if frame_id % interval == 1 or (frame_id == 1 and not positions):
            positions.append(pos)
    if not positions and video_frames:
        positions.append(0)
    return positions


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.load_model)), "vis_sample")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_features:
        args.feature_output_dir = os.path.join(os.path.dirname(os.path.abspath(args.load_model)), "vis_feature")
        os.makedirs(args.feature_output_dir, exist_ok=True)

    opt = build_opt(args)
    dataset_cls = get_dataset(opt)
    dataset = dataset_cls(opt, args.split)
    images_by_id, video_images, anns_by_image, track_box = build_annotation_index(dataset)

    if opt.off_flag:
        heads = {"hm": dataset.num_classes, "wh": 2, "reg": 2}
    else:
        heads = {"hm": dataset.num_classes, "wh": 2}
    model = get_det_net(heads, opt.model_name, dataset.resolution, opt.seqLen, opt, thresh=opt.thresh)
    model = load_model(model, args.load_model)
    model = model.to(opt.device).eval()

    captures, handles, sample_layers = register_skip_hooks(
        model, args.layers, capture_samples=args.vis_sample, capture_features=args.vis_features
    )
    if not handles:
        raise RuntimeError("No hooks registered. Enable --vis_sample and/or --vis_features.")
    if args.vis_sample and not sample_layers:
        raise RuntimeError("No deformable-convolution sampling hooks registered. Check --layers and UNet3D skip settings.")

    wanted_videos = set(args.videos) if args.videos else None
    videos = sorted(video_images)
    if wanted_videos is not None:
        videos = [video for video in videos if video in wanted_videos]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    pending_inputs = []
    pending_tasks = []
    saved = 0
    total_counts = {name: {"inside": 0, "outside": 0} for name in args.layers}

    def flush_pending():
        nonlocal pending_inputs, pending_tasks, saved
        if not pending_inputs:
            return
        merged = concat_batches(pending_inputs)
        torch_batch = to_torch_batch(merged, opt.device)
        captures.clear()
        with torch.no_grad():
            _ = model(torch_batch)
        for task_index, task in enumerate(pending_tasks):
            task["batch_index"] = task_index
            result = visualize_task(task, captures, args, opt, anns_by_image, track_box) if args.vis_sample else None
            if result:
                out_path, layer_counts = result
                saved += 1
                for layer_name, counts in layer_counts.items():
                    total_counts[layer_name]["inside"] += counts["inside"]
                    total_counts[layer_name]["outside"] += counts["outside"]
                print("saved {}".format(out_path))
            for feature_path in visualize_feature_task(task, captures, args, opt, anns_by_image):
                print("saved {}".format(feature_path))
        pending_inputs = []
        pending_tasks = []

    try:
        for video_idx, video_name in enumerate(videos):
            frames = video_images[video_name]
            positions = selected_frame_positions(frames, args.frame_interval)
            for pos in positions:
                if args.max_frames > 0 and saved + len(pending_tasks) >= args.max_frames:
                    break
                patch_frames, local_t = patch_for_center(frames, pos, opt.seqLen)
                if patch_frames is None:
                    continue
                center_img = frames[pos]
                if not anns_by_image.get(int(center_img["id"])):
                    continue
                _, input_batch = prepare_patch(dataset, opt, patch_frames)
                pending_inputs.append(input_batch)
                pending_tasks.append({
                    "video_name": video_name,
                    "center_img": center_img,
                    "local_t": local_t,
                })
                if len(pending_inputs) >= max(1, args.batch_patches):
                    flush_pending()
            print("processed video {}/{}: {}".format(video_idx + 1, len(videos), video_name))
            if args.max_frames > 0 and saved + len(pending_tasks) >= args.max_frames:
                break
        flush_pending()
    finally:
        remove_hooks(handles)

    for layer_name in args.layers:
        counts = total_counts[layer_name]
        total = counts["inside"] + counts["outside"]
        rate = counts["inside"] / total * 100.0 if total else 0.0
        print("{}: inside {} outside {} inside_rate {:.2f}%".format(
            layer_name, counts["inside"], counts["outside"], rate))
    print("done. saved {} visualizations to {}".format(saved, args.output_dir))


if __name__ == "__main__":
    main()
