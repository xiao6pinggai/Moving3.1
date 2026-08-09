import time

import torch
import tvdcn
from tvdcn import ops


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
    print("tvdcn_version", getattr(tvdcn, "__version__", None))
    print("has_ops", tvdcn.has_ops())
    print("with_cuda", tvdcn.with_cuda())
    print("cuda_version", tvdcn.cuda_version())
    if hasattr(tvdcn, "cuda_arch_list"):
        print("cuda_arch_list", tvdcn.cuda_arch_list())
    print("conv3d_symbols", [x for x in dir(ops) if "Conv3d" in x or "conv3d" in x])

    device = "cuda"
    dtype = torch.float32
    n, cin, cout = 1, 8, 8
    d, h, w = 8, 32, 32
    k = 3
    x = torch.randn(n, cin, d, h, w, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(cout, cin, k, k, k, device=device, dtype=dtype, requires_grad=True)
    offset = torch.zeros(n, 3 * k * k * k, d, h, w, device=device, dtype=dtype, requires_grad=True)

    y = ops.deform_conv3d(x, weight, offset, None, None, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
    loss = y.square().mean()
    loss.backward()
    sync()
    print("forward_shape", tuple(y.shape))
    print("grad_ok", x.grad is not None and weight.grad is not None and offset.grad is not None)

    # Forward latency, modest tensor size for GTX 1650 Ti.
    warmup, iters = 10, 50
    for _ in range(warmup):
        y = ops.deform_conv3d(x, weight, offset, None, None, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        y = ops.deform_conv3d(x, weight, offset, None, None, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
    sync()
    print("forward_ms", (time.perf_counter() - start) * 1000.0 / iters)

    conv = torch.nn.Conv3d(cin, cout, k, padding=1, bias=False).to(device)
    conv.weight.data.copy_(weight.detach())
    with torch.no_grad():
        z = conv(x)
        print("zero_offset_max_abs_diff_vs_conv3d", (y - z).abs().max().item())
    for _ in range(warmup):
        z = conv(x)
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        z = conv(x)
    sync()
    print("conv3d_forward_ms", (time.perf_counter() - start) * 1000.0 / iters)

    bw_iters = 20
    x_b = x.detach().clone().requires_grad_(True)
    w_b = weight.detach().clone().requires_grad_(True)
    o_b = offset.detach().clone().requires_grad_(True)
    for _ in range(5):
        y_b = ops.deform_conv3d(x_b, w_b, o_b, None, None, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
        y_b.mean().backward()
        x_b.grad.zero_()
        w_b.grad.zero_()
        o_b.grad.zero_()
    sync()
    start = time.perf_counter()
    for _ in range(bw_iters):
        y_b = ops.deform_conv3d(x_b, w_b, o_b, None, None, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1)
        y_b.mean().backward()
        x_b.grad.zero_()
        w_b.grad.zero_()
        o_b.grad.zero_()
    sync()
    print("forward_backward_ms", (time.perf_counter() - start) * 1000.0 / bw_iters)


if __name__ == "__main__":
    main()
