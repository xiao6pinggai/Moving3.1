# Triplet V18 CUDA exact pair top-k

## Files

- `triplet_topk_ext.cpp`: PyBind11/C++ entry.
- `triplet_topk_ext_kernel.cu`: CUDA kernels.
- `triplet_topk_cuda.py`: small Python wrapper around the compiled extension.
- `triplet_motion_v18.py`: full V18 module. Trainable parameter names are kept compatible with the previous module.
- `setup.py`: build script.

## Requirements

Use a PyTorch installation with CUDA support and an installed `nvcc` that is compatible with your PyTorch CUDA runtime.

Check:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
PY
nvcc --version
```

## Build

From this folder:

```bash
python setup.py build_ext --inplace
```

or:

```bash
pip install -v -e .
```

Optional: specify architectures explicitly, for example:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
python setup.py build_ext --inplace
```

## Smoke test

```bash
python - <<'PY'
import torch
import triplet_topk_cuda_ext
from triplet_motion_v18 import TripletMotionConsistencyMotionPairSparseConv
print('extension import: ok')
print('class import: ok', TripletMotionConsistencyMotionPairSparseConv)
PY
```

## Notes for Codex

1. Work inside this directory.
2. Run `python setup.py build_ext --inplace`.
3. Make sure the generated `.so` file is in the same directory as `triplet_topk_cuda.py` and `triplet_motion_v18.py`.
4. In training code, import:

```python
from triplet_motion_v18 import TripletMotionConsistencyMotionPairSparseConv
```

5. If the data pipeline already has a dense index map, pass it into forward:

```python
out, score = module(x, index_map=index_map)
```

The expected format is:

```python
index_map[b, t, y, x] = sparse row index
index_map[b, t, y, x] = -1  # invalid / empty
```

Use `torch.int32` on CUDA for best speed and memory.

If no `index_map` is passed, V18 will build one from `x_conv.indices`. For best speed, precompute and pass `index_map` from the segmentation/filtering stage.
