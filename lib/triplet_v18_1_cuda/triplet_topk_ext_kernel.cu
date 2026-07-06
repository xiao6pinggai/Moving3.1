#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

namespace {

constexpr int INF_RANK = 2147483647;

__device__ __forceinline__ int get_index_map_4d_i32(
    const int* __restrict__ index_map,
    int b,
    int t,
    int y,
    int x,
    int B,
    int T,
    int H,
    int W
) {
    if (b < 0 || b >= B || t < 0 || t >= T || y < 0 || y >= H || x < 0 || x >= W) {
        return -1;
    }
    const int64_t offset = (((static_cast<int64_t>(b) * T + t) * H + y) * W + x);
    return index_map[offset];
}

__device__ __forceinline__ void insert_rank_topk_i32(
    int rank,
    int prev_idx,
    int next_idx,
    int* __restrict__ best_rank,
    int* __restrict__ best_prev,
    int* __restrict__ best_next,
    int K
) {
    int worst = 0;
    int worst_rank = best_rank[0];

    for (int i = 1; i < K; ++i) {
        const int r = best_rank[i];
        if (r > worst_rank) {
            worst = i;
            worst_rank = r;
        }
    }

    if (rank < worst_rank) {
        best_rank[worst] = rank;
        best_prev[worst] = prev_idx;
        best_next[worst] = next_idx;
    }
}

__device__ __forceinline__ void sort_small_topk_i32(
    int* __restrict__ best_rank,
    int* __restrict__ best_prev,
    int* __restrict__ best_next,
    int K
) {
    for (int i = 0; i < K; ++i) {
        int min_j = i;
        int min_rank = best_rank[i];
        for (int j = i + 1; j < K; ++j) {
            if (best_rank[j] < min_rank) {
                min_j = j;
                min_rank = best_rank[j];
            }
        }
        if (min_j != i) {
            const int tr = best_rank[i];
            best_rank[i] = best_rank[min_j];
            best_rank[min_j] = tr;

            const int tp = best_prev[i];
            best_prev[i] = best_prev[min_j];
            best_prev[min_j] = tp;

            const int tn = best_next[i];
            best_next[i] = best_next[min_j];
            best_next[min_j] = tn;
        }
    }
}

__global__ void triplet_topk_parallel_kernel(
    const int* __restrict__ indices,              // [N, 4]
    const int* __restrict__ index_map,            // [B, T, H, W], invalid = -1
    const int* __restrict__ pair_prev_offset_id,  // [P], sorted by motion cost
    const int* __restrict__ pair_next_offset_id,  // [P], sorted by motion cost
    const int* __restrict__ window_offsets,       // [O, 2]
    int64_t* __restrict__ out_prev,               // [N, K]
    int64_t* __restrict__ out_next,               // [N, K]
    bool* __restrict__ out_mask,                  // [N, K]
    int N,
    int B,
    int T,
    int H,
    int W,
    int P,
    int temporal_dilation,
    int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int b = indices[row * 4 + 0];
    const int t = indices[row * 4 + 1];
    const int y = indices[row * 4 + 2];
    const int x = indices[row * 4 + 3];

    extern __shared__ int smem[];

    const int threads = blockDim.x;
    int* sm_rank = smem;
    int* sm_prev = sm_rank + threads * K;
    int* sm_next = sm_prev + threads * K;
    int* final_rank = sm_next + threads * K;
    int* final_prev = final_rank + K;
    int* final_next = final_prev + K;

    const int base = threadIdx.x * K;
    for (int k = 0; k < K; ++k) {
        sm_rank[base + k] = INF_RANK;
        sm_prev[base + k] = 0;
        sm_next[base + k] = 0;
    }

    __syncthreads();

    for (int pid = threadIdx.x; pid < P; pid += blockDim.x) {
        const int po = pair_prev_offset_id[pid];
        const int no = pair_next_offset_id[pid];

        const int op_y = window_offsets[po * 2 + 0];
        const int op_x = window_offsets[po * 2 + 1];
        const int on_y = window_offsets[no * 2 + 0];
        const int on_x = window_offsets[no * 2 + 1];

        const int prev_idx = get_index_map_4d_i32(
            index_map,
            b,
            t - temporal_dilation,
            y + op_y,
            x + op_x,
            B,
            T,
            H,
            W
        );
        if (prev_idx < 0) continue;

        const int next_idx = get_index_map_4d_i32(
            index_map,
            b,
            t + temporal_dilation,
            y + on_y,
            x + on_x,
            B,
            T,
            H,
            W
        );
        if (next_idx < 0) continue;

        // pair_prev_offset_id / pair_next_offset_id are already sorted by exact motion cost.
        // Therefore exact top-k valid pairs are the K smallest pid among valid pairs.
        insert_rank_topk_i32(
            pid,
            prev_idx,
            next_idx,
            sm_rank + base,
            sm_prev + base,
            sm_next + base,
            K
        );
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            final_rank[k] = INF_RANK;
            final_prev[k] = 0;
            final_next[k] = 0;
        }

        for (int th = 0; th < threads; ++th) {
            const int th_base = th * K;
            for (int k = 0; k < K; ++k) {
                const int r = sm_rank[th_base + k];
                if (r < INF_RANK) {
                    insert_rank_topk_i32(
                        r,
                        sm_prev[th_base + k],
                        sm_next[th_base + k],
                        final_rank,
                        final_prev,
                        final_next,
                        K
                    );
                }
            }
        }

        sort_small_topk_i32(final_rank, final_prev, final_next, K);

        for (int k = 0; k < K; ++k) {
            const int64_t out_pos = static_cast<int64_t>(row) * K + k;
            if (final_rank[k] < INF_RANK) {
                out_prev[out_pos] = static_cast<int64_t>(final_prev[k]);
                out_next[out_pos] = static_cast<int64_t>(final_next[k]);
                out_mask[out_pos] = true;
            } else {
                out_prev[out_pos] = 0;
                out_next[out_pos] = 0;
                out_mask[out_pos] = false;
            }
        }
    }
}

__global__ void triplet_topk_serial_kernel(
    const int* __restrict__ indices,
    const int* __restrict__ index_map,
    const int* __restrict__ pair_prev_offset_id,
    const int* __restrict__ pair_next_offset_id,
    const int* __restrict__ window_offsets,
    int64_t* __restrict__ out_prev,
    int64_t* __restrict__ out_next,
    bool* __restrict__ out_mask,
    int N,
    int B,
    int T,
    int H,
    int W,
    int P,
    int temporal_dilation,
    int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int b = indices[row * 4 + 0];
    const int t = indices[row * 4 + 1];
    const int y = indices[row * 4 + 2];
    const int x = indices[row * 4 + 3];

    int filled = 0;

    // Because pair ids are sorted by motion cost, greedily take the first valid
    // pairs without reusing either the previous-frame point or next-frame point.
    for (int pid = 0; pid < P && filled < K; ++pid) {
        const int po = pair_prev_offset_id[pid];
        const int no = pair_next_offset_id[pid];

        const int op_y = window_offsets[po * 2 + 0];
        const int op_x = window_offsets[po * 2 + 1];
        const int on_y = window_offsets[no * 2 + 0];
        const int on_x = window_offsets[no * 2 + 1];

        const int prev_idx = get_index_map_4d_i32(
            index_map,
            b,
            t - temporal_dilation,
            y + op_y,
            x + op_x,
            B,
            T,
            H,
            W
        );
        if (prev_idx < 0) continue;

        const int next_idx = get_index_map_4d_i32(
            index_map,
            b,
            t + temporal_dilation,
            y + on_y,
            x + on_x,
            B,
            T,
            H,
            W
        );
        if (next_idx < 0) continue;

        bool reused = false;
        for (int k = 0; k < filled; ++k) {
            const int64_t used_pos = static_cast<int64_t>(row) * K + k;
            if (
                static_cast<int>(out_prev[used_pos]) == prev_idx ||
                static_cast<int>(out_next[used_pos]) == next_idx
            ) {
                reused = true;
                break;
            }
        }
        if (reused) continue;

        const int64_t out_pos = static_cast<int64_t>(row) * K + filled;
        out_prev[out_pos] = static_cast<int64_t>(prev_idx);
        out_next[out_pos] = static_cast<int64_t>(next_idx);
        out_mask[out_pos] = true;
        ++filled;
    }

    for (int k = filled; k < K; ++k) {
        const int64_t out_pos = static_cast<int64_t>(row) * K + k;
        out_prev[out_pos] = 0;
        out_next[out_pos] = 0;
        out_mask[out_pos] = false;
    }
}

} // namespace

std::vector<torch::Tensor> triplet_topk_cuda_forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor pair_prev_offset_id,
    torch::Tensor pair_next_offset_id,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
) {
    const c10::cuda::CUDAGuard device_guard(indices.device());

    const int64_t N64 = indices.size(0);
    const int64_t B64 = index_map.size(0);
    const int64_t T64 = index_map.size(1);
    const int64_t H64 = index_map.size(2);
    const int64_t W64 = index_map.size(3);
    const int64_t P64 = pair_prev_offset_id.numel();

    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N is too large for int32 kernel indexing");
    TORCH_CHECK(B64 <= std::numeric_limits<int>::max(), "B is too large for int32 kernel indexing");
    TORCH_CHECK(T64 <= std::numeric_limits<int>::max(), "T is too large for int32 kernel indexing");
    TORCH_CHECK(H64 <= std::numeric_limits<int>::max(), "H is too large for int32 kernel indexing");
    TORCH_CHECK(W64 <= std::numeric_limits<int>::max(), "W is too large for int32 kernel indexing");
    TORCH_CHECK(P64 <= std::numeric_limits<int>::max(), "number of offset pairs is too large for int32 kernel indexing");
    TORCH_CHECK(topk <= std::numeric_limits<int>::max(), "topk is too large for int32 kernel indexing");

    const int N = static_cast<int>(N64);
    const int B = static_cast<int>(B64);
    const int T = static_cast<int>(T64);
    const int H = static_cast<int>(H64);
    const int W = static_cast<int>(W64);
    const int P = static_cast<int>(P64);
    const int K = static_cast<int>(topk);
    const int dt = static_cast<int>(temporal_dilation);

    auto long_opts = torch::TensorOptions().device(indices.device()).dtype(torch::kInt64);
    auto bool_opts = torch::TensorOptions().device(indices.device()).dtype(torch::kBool);

    auto out_prev = torch::empty({N64, topk}, long_opts);
    auto out_next = torch::empty({N64, topk}, long_opts);
    auto out_mask = torch::empty({N64, topk}, bool_opts);

    if (N == 0) {
        return {out_prev, out_next, out_mask};
    }

    // The serial early-exit path is faster for small K when valid pairs appear early.
    // Keep the parallel path available for future experiments.
    int threads = 256;
    size_t shared_bytes = 0;
    bool use_parallel = false;
    const bool prefer_serial_early_exit = true;

    while (!prefer_serial_early_exit && threads >= 32) {
        const int64_t shared_ints = static_cast<int64_t>(K) * (static_cast<int64_t>(threads) * 3 + 3);
        shared_bytes = static_cast<size_t>(shared_ints * static_cast<int64_t>(sizeof(int)));
        if (shared_bytes <= 48 * 1024) {
            use_parallel = true;
            break;
        }
        threads /= 2;
    }

    const dim3 blocks(N);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (use_parallel) {
        triplet_topk_parallel_kernel<<<blocks, threads, shared_bytes, stream>>>(
            indices.data_ptr<int>(),
            index_map.data_ptr<int>(),
            pair_prev_offset_id.data_ptr<int>(),
            pair_next_offset_id.data_ptr<int>(),
            window_offsets.data_ptr<int>(),
            out_prev.data_ptr<int64_t>(),
            out_next.data_ptr<int64_t>(),
            out_mask.data_ptr<bool>(),
            N,
            B,
            T,
            H,
            W,
            P,
            dt,
            K
        );
    } else {
        // Supports arbitrary positive topk without shared-memory pressure and exits once K valid pairs are found.
        triplet_topk_serial_kernel<<<blocks, 1, 0, stream>>>(
            indices.data_ptr<int>(),
            index_map.data_ptr<int>(),
            pair_prev_offset_id.data_ptr<int>(),
            pair_next_offset_id.data_ptr<int>(),
            window_offsets.data_ptr<int>(),
            out_prev.data_ptr<int64_t>(),
            out_next.data_ptr<int64_t>(),
            out_mask.data_ptr<bool>(),
            N,
            B,
            T,
            H,
            W,
            P,
            dt,
            K
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_prev, out_next, out_mask};
}
