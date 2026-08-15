from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import OrderedDict

import torch


RENAME_PAIRS = (
    ("backbone.snake_skip1", "backbone.skip1"),
    ("backbone.snake_skip2", "backbone.skip2"),
    ("backbone.snake_skip3", "backbone.skip3"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename old UNet3D snake skip checkpoint keys to current skip keys."
    )
    parser.add_argument("--pth", default="//root/autodl-tmp/SGNet/weights/aircraft_multi/UNet3D/UNet3D_tdcn111555_seglen10_2026_08_06_18_27_23/model_best_ap50.pth", help="input .pth checkpoint path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output .pth path, default: <input>_renamed.pth",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite input checkpoint. Ignored when --output is set.",
    )
    return parser.parse_args()


def default_output_path(pth):
    root, ext = os.path.splitext(pth)
    if not ext:
        ext = ".pth"
    return root + "_renamed" + ext


def rename_key(key):
    for old, new in RENAME_PAIRS:
        if key.startswith(old + "."):
            return new + key[len(old):]
    return key


def rename_state_dict(state_dict):
    renamed = OrderedDict()
    changed = []

    for key, value in state_dict.items():
        new_key = rename_key(key)
        if new_key in renamed:
            raise KeyError("Renaming would create duplicate key: {}".format(new_key))
        renamed[new_key] = value
        if new_key != key:
            changed.append((key, new_key))

    return renamed, changed


def main():
    args = parse_args()
    output = args.output
    if output is None:
        output = args.pth if args.in_place else default_output_path(args.pth)

    checkpoint = torch.load(args.pth, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        new_state_dict, changed = rename_state_dict(checkpoint["state_dict"])
        checkpoint["state_dict"] = new_state_dict
        save_obj = checkpoint
    else:
        save_obj, changed = rename_state_dict(checkpoint)

    torch.save(save_obj, output)

    print("input:  {}".format(args.pth))
    print("output: {}".format(output))
    print("renamed keys: {}".format(len(changed)))
    for old, new in changed[:20]:
        print("{} -> {}".format(old, new))
    if len(changed) > 20:
        print("... {} more".format(len(changed) - 20))


if __name__ == "__main__":
    main()
