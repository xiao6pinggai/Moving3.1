import torch

from dcn_nd import DeformConv1d, DeformConv2d, DeformConv3d


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = DeformConv1d(4, 5, 3, padding=1).to(dev)
    y1 = m1(torch.randn(2, 4, 16, device=dev))
    print("1d", tuple(y1.shape))

    m2 = DeformConv2d(4, 5, (3, 3), padding=1).to(dev)
    y2 = m2(torch.randn(2, 4, 16, 16, device=dev))
    print("2d", tuple(y2.shape))

    m3 = DeformConv3d(4, 5, (1, 3, 3), padding=(0, 1, 1)).to(dev)
    x = torch.randn(2, 4, 4, 16, 16, device=dev, requires_grad=True)
    y3 = m3(x)
    y3.mean().backward()
    print("3d", tuple(y3.shape), x.grad is not None)

    offset = m3.offset_generator(x.detach())
    y4 = m3(x.detach(), offset=offset)
    print("external_offset", tuple(y4.shape))


if __name__ == "__main__":
    main()
