import importlib
import os
import sys


def load_feature_topk_v23_exact():
    cuda_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feature_topk_v23_cuda"))
    if cuda_dir in sys.path:
        sys.path.remove(cuda_dir)
    sys.path.insert(0, cuda_dir)

    sys.modules.pop("feature_topk_v23_cuda_ext", None)
    return importlib.import_module("feature_topk_v23_cuda").feature_topk_v23_exact
