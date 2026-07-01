#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> triplet_topk_cuda_forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor pair_prev_offset_id,
    torch::Tensor pair_next_offset_id,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
);

std::vector<torch::Tensor> forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor pair_prev_offset_id,
    torch::Tensor pair_next_offset_id,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(index_map.is_cuda(), "index_map must be a CUDA tensor");
    TORCH_CHECK(pair_prev_offset_id.is_cuda(), "pair_prev_offset_id must be a CUDA tensor");
    TORCH_CHECK(pair_next_offset_id.is_cuda(), "pair_next_offset_id must be a CUDA tensor");
    TORCH_CHECK(window_offsets.is_cuda(), "window_offsets must be a CUDA tensor");

    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be torch.int32");
    TORCH_CHECK(index_map.scalar_type() == torch::kInt32, "index_map must be torch.int32, with invalid value -1");
    TORCH_CHECK(pair_prev_offset_id.scalar_type() == torch::kInt32, "pair_prev_offset_id must be torch.int32");
    TORCH_CHECK(pair_next_offset_id.scalar_type() == torch::kInt32, "pair_next_offset_id must be torch.int32");
    TORCH_CHECK(window_offsets.scalar_type() == torch::kInt32, "window_offsets must be torch.int32");

    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == 4, "indices must have shape [N, 4]");
    TORCH_CHECK(index_map.dim() == 4, "index_map must have shape [B, T, H, W]");
    TORCH_CHECK(pair_prev_offset_id.dim() == 1, "pair_prev_offset_id must have shape [P]");
    TORCH_CHECK(pair_next_offset_id.dim() == 1, "pair_next_offset_id must have shape [P]");
    TORCH_CHECK(pair_prev_offset_id.numel() == pair_next_offset_id.numel(), "pair id tensors must have the same length");
    TORCH_CHECK(window_offsets.dim() == 2 && window_offsets.size(1) == 2, "window_offsets must have shape [O, 2]");
    TORCH_CHECK(topk > 0, "topk must be > 0");
    TORCH_CHECK(temporal_dilation > 0, "temporal_dilation must be > 0");

    return triplet_topk_cuda_forward(
        indices.contiguous(),
        index_map.contiguous(),
        pair_prev_offset_id.contiguous(),
        pair_next_offset_id.contiguous(),
        window_offsets.contiguous(),
        temporal_dilation,
        topk
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Exact triplet motion pair top-k forward CUDA");
}
