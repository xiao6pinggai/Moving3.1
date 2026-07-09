#include <torch/extension.h>
#include <vector>

constexpr int FEATURE_TOPK_V25_API_VERSION = 2;

std::vector<torch::Tensor> feature_topk_v25_cuda_forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor features,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
);

int api_version() {
    return FEATURE_TOPK_V25_API_VERSION;
}

std::vector<torch::Tensor> forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor features,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(index_map.is_cuda(), "index_map must be a CUDA tensor");
    TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
    TORCH_CHECK(window_offsets.is_cuda(), "window_offsets must be a CUDA tensor");

    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be torch.int32");
    TORCH_CHECK(index_map.scalar_type() == torch::kInt32, "index_map must be torch.int32, with invalid value -1");
    TORCH_CHECK(features.is_floating_point(), "features must be a floating point tensor");
    TORCH_CHECK(window_offsets.scalar_type() == torch::kInt32, "window_offsets must be torch.int32");

    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 4, "indices must have shape [N, 4]");
    TORCH_CHECK(index_map.dim() == 4, "index_map must have shape [B, T, H, W]");
    TORCH_CHECK(features.dim() == 2 && features.size(0) == indices.size(0), "features must have shape [N, C]");
    TORCH_CHECK(window_offsets.dim() == 2 && window_offsets.size(1) == 2, "window_offsets must have shape [O, 2]");
    TORCH_CHECK(topk > 0, "topk must be > 0");
    TORCH_CHECK(temporal_dilation == 1, "feature_topk_v25_cuda_ext is specialized for temporal_dilation=1");

    return feature_topk_v25_cuda_forward(
        indices.contiguous(),
        index_map.contiguous(),
        features.contiguous(),
        window_offsets.contiguous(),
        temporal_dilation,
        topk
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("api_version", &api_version, "feature_topk_v25 CUDA extension API version");
    m.def("forward", &forward, "Feature-distance prev/current/next top-k forward CUDA for cosv25");
}
