import importlib
import os
import sys


def load_triplet_topk_exact(topk_relu):
    cuda_name = {
        "fanghui": "triplet_v18_cuda",
        "bufanghui": "triplet_v18_1_cuda",
    }[str(topk_relu).lower()]

    cuda_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cuda_name))
    if cuda_dir in sys.path:
        sys.path.remove(cuda_dir)
    sys.path.insert(0, cuda_dir)

    sys.modules.pop("triplet_topk_cuda", None)
    sys.modules.pop("triplet_topk_cuda_ext", None)
    return importlib.import_module("triplet_topk_cuda").triplet_topk_exact
