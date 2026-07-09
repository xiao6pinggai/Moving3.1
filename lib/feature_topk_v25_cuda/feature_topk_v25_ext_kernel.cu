#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

namespace {

constexpr float INF_DIST = 3.4028234663852886e38F;
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

__device__ __forceinline__ bool is_better(float dist, int rank, float worst_dist, int worst_rank) {
    return dist < worst_dist || (dist == worst_dist && rank < worst_rank);
}

__device__ __forceinline__ void insert_dist_topk(
    float dist,
    int rank,
    int idx,
    float* __restrict__ best_dist,
    int* __restrict__ best_rank,
    int* __restrict__ best_idx,
    int K
) {
    int worst = 0;
    float worst_dist = best_dist[0];
    int worst_rank = best_rank[0];

    for (int i = 1; i < K; ++i) {
        const float d = best_dist[i];
        const int r = best_rank[i];
        if (d > worst_dist || (d == worst_dist && r > worst_rank)) {
            worst = i;
            worst_dist = d;
            worst_rank = r;
        }
    }

    if (is_better(dist, rank, worst_dist, worst_rank)) {
        best_dist[worst] = dist;
        best_rank[worst] = rank;
        best_idx[worst] = idx;
    }
}

__device__ __forceinline__ void sort_small_topk(
    float* __restrict__ best_dist,
    int* __restrict__ best_rank,
    int* __restrict__ best_idx,
    int K
) {
    for (int i = 0; i < K; ++i) {
        int min_j = i;
        float min_dist = best_dist[i];
        int min_rank = best_rank[i];
        for (int j = i + 1; j < K; ++j) {
            const float d = best_dist[j];
            const int r = best_rank[j];
            if (d < min_dist || (d == min_dist && r < min_rank)) {
                min_j = j;
                min_dist = d;
                min_rank = r;
            }
        }
        if (min_j != i) {
            const float td = best_dist[i];
            best_dist[i] = best_dist[min_j];
            best_dist[min_j] = td;

            const int tr = best_rank[i];
            best_rank[i] = best_rank[min_j];
            best_rank[min_j] = tr;

            const int ti = best_idx[i];
            best_idx[i] = best_idx[min_j];
            best_idx[min_j] = ti;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ float l2_feature_distance(
    const scalar_t* __restrict__ features,
    int query_idx,
    int cand_idx,
    int C
) {
    float acc = 0.0F;
    const int64_t q_base = static_cast<int64_t>(query_idx) * C;
    const int64_t c_base = static_cast<int64_t>(cand_idx) * C;
    for (int c = 0; c < C; ++c) {
        const float q = static_cast<float>(features[q_base + c]);
        const float v = static_cast<float>(features[c_base + c]);
        const float diff = v - q;
        acc += diff * diff;
    }
    return acc;
}

template <typename scalar_t>
__global__ void feature_topk_v25_kernel(
    const int* __restrict__ indices,
    const int* __restrict__ index_map,
    const scalar_t* __restrict__ features,
    const int* __restrict__ window_offsets,
    int64_t* __restrict__ out_prev,
    int64_t* __restrict__ out_cur,
    int64_t* __restrict__ out_next,
    bool* __restrict__ mask_prev,
    bool* __restrict__ mask_cur,
    bool* __restrict__ mask_next,
    float* __restrict__ dist_prev,
    float* __restrict__ dist_cur,
    float* __restrict__ dist_next,
    int64_t* __restrict__ pos_prev,
    int64_t* __restrict__ pos_cur,
    int64_t* __restrict__ pos_next,
    int N,
    int C,
    int B,
    int T,
    int H,
    int W,
    int O,
    int K
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int curK = K - 1;
    const int b = indices[row * 4 + 0];
    const int t = indices[row * 4 + 1];
    const int y = indices[row * 4 + 2];
    const int x = indices[row * 4 + 3];
    const int center_pos = O / 2;

    extern __shared__ unsigned char smem_raw[];
    float* sm_prev_dist = reinterpret_cast<float*>(smem_raw);
    float* sm_cur_dist = sm_prev_dist + blockDim.x * K;
    float* sm_next_dist = sm_cur_dist + blockDim.x * K;
    int* sm_prev_rank = reinterpret_cast<int*>(sm_next_dist + blockDim.x * K);
    int* sm_cur_rank = sm_prev_rank + blockDim.x * K;
    int* sm_next_rank = sm_cur_rank + blockDim.x * K;
    int* sm_prev_idx = sm_next_rank + blockDim.x * K;
    int* sm_cur_idx = sm_prev_idx + blockDim.x * K;
    int* sm_next_idx = sm_cur_idx + blockDim.x * K;
    float* final_prev_dist = reinterpret_cast<float*>(sm_next_idx + blockDim.x * K);
    float* final_cur_dist = final_prev_dist + K;
    float* final_next_dist = final_cur_dist + K;
    int* final_prev_rank = reinterpret_cast<int*>(final_next_dist + K);
    int* final_cur_rank = final_prev_rank + K;
    int* final_next_rank = final_cur_rank + K;
    int* final_prev_idx = final_next_rank + K;
    int* final_cur_idx = final_prev_idx + K;
    int* final_next_idx = final_cur_idx + K;

    const int base = threadIdx.x * K;
    for (int k = 0; k < K; ++k) {
        sm_prev_dist[base + k] = INF_DIST;
        sm_cur_dist[base + k] = INF_DIST;
        sm_next_dist[base + k] = INF_DIST;
        sm_prev_rank[base + k] = INF_RANK;
        sm_cur_rank[base + k] = INF_RANK;
        sm_next_rank[base + k] = INF_RANK;
        sm_prev_idx[base + k] = 0;
        sm_cur_idx[base + k] = 0;
        sm_next_idx[base + k] = 0;
    }

    __syncthreads();

    const bool has_prev = t > 0;
    const bool has_next = t + 1 < T;
    for (int oid = threadIdx.x; oid < O; oid += blockDim.x) {
        const int off_y = window_offsets[oid * 2 + 0];
        const int off_x = window_offsets[oid * 2 + 1];

        if (has_prev) {
            const int prev_idx = get_index_map_4d_i32(index_map, b, t - 1, y + off_y, x + off_x, B, T, H, W);
            if (prev_idx >= 0) {
                const float dist = l2_feature_distance(features, row, prev_idx, C);
                insert_dist_topk(dist, oid, prev_idx, sm_prev_dist + base, sm_prev_rank + base, sm_prev_idx + base, K);
            }
        }

        if (curK > 0) {
            const int cur_idx = get_index_map_4d_i32(index_map, b, t, y + off_y, x + off_x, B, T, H, W);
            if (cur_idx >= 0 && cur_idx != row) {
                const float dist = l2_feature_distance(features, row, cur_idx, C);
                insert_dist_topk(dist, oid, cur_idx, sm_cur_dist + base, sm_cur_rank + base, sm_cur_idx + base, curK);
            }
        }

        if (has_next) {
            const int next_idx = get_index_map_4d_i32(index_map, b, t + 1, y + off_y, x + off_x, B, T, H, W);
            if (next_idx >= 0) {
                const float dist = l2_feature_distance(features, row, next_idx, C);
                insert_dist_topk(dist, oid, next_idx, sm_next_dist + base, sm_next_rank + base, sm_next_idx + base, K);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            final_prev_dist[k] = INF_DIST;
            final_cur_dist[k] = INF_DIST;
            final_next_dist[k] = INF_DIST;
            final_prev_rank[k] = INF_RANK;
            final_cur_rank[k] = INF_RANK;
            final_next_rank[k] = INF_RANK;
            final_prev_idx[k] = 0;
            final_cur_idx[k] = 0;
            final_next_idx[k] = 0;
        }

        for (int th = 0; th < blockDim.x; ++th) {
            const int th_base = th * K;
            for (int k = 0; k < K; ++k) {
                if (sm_prev_rank[th_base + k] < INF_RANK) {
                    insert_dist_topk(sm_prev_dist[th_base + k], sm_prev_rank[th_base + k], sm_prev_idx[th_base + k], final_prev_dist, final_prev_rank, final_prev_idx, K);
                }
                if (sm_next_rank[th_base + k] < INF_RANK) {
                    insert_dist_topk(sm_next_dist[th_base + k], sm_next_rank[th_base + k], sm_next_idx[th_base + k], final_next_dist, final_next_rank, final_next_idx, K);
                }
                if (curK > 0 && k < curK && sm_cur_rank[th_base + k] < INF_RANK) {
                    insert_dist_topk(sm_cur_dist[th_base + k], sm_cur_rank[th_base + k], sm_cur_idx[th_base + k], final_cur_dist, final_cur_rank, final_cur_idx, curK);
                }
            }
        }

        sort_small_topk(final_prev_dist, final_prev_rank, final_prev_idx, K);
        if (curK > 0) {
            sort_small_topk(final_cur_dist, final_cur_rank, final_cur_idx, curK);
        }
        sort_small_topk(final_next_dist, final_next_rank, final_next_idx, K);

        for (int k = 0; k < K; ++k) {
            const int64_t out_pos = static_cast<int64_t>(row) * K + k;
            if (final_prev_rank[k] < INF_RANK) {
                out_prev[out_pos] = static_cast<int64_t>(final_prev_idx[k]);
                mask_prev[out_pos] = true;
                dist_prev[out_pos] = final_prev_dist[k];
                pos_prev[out_pos] = static_cast<int64_t>(final_prev_rank[k]);
            } else {
                out_prev[out_pos] = 0;
                mask_prev[out_pos] = false;
                dist_prev[out_pos] = INF_DIST;
                pos_prev[out_pos] = 0;
            }

            if (k == 0) {
                out_cur[out_pos] = static_cast<int64_t>(row);
                mask_cur[out_pos] = true;
                dist_cur[out_pos] = 0.0F;
                pos_cur[out_pos] = static_cast<int64_t>(center_pos);
            } else if (final_cur_rank[k - 1] < INF_RANK) {
                out_cur[out_pos] = static_cast<int64_t>(final_cur_idx[k - 1]);
                mask_cur[out_pos] = true;
                dist_cur[out_pos] = final_cur_dist[k - 1];
                pos_cur[out_pos] = static_cast<int64_t>(final_cur_rank[k - 1]);
            } else {
                out_cur[out_pos] = 0;
                mask_cur[out_pos] = false;
                dist_cur[out_pos] = INF_DIST;
                pos_cur[out_pos] = 0;
            }

            if (final_next_rank[k] < INF_RANK) {
                out_next[out_pos] = static_cast<int64_t>(final_next_idx[k]);
                mask_next[out_pos] = true;
                dist_next[out_pos] = final_next_dist[k];
                pos_next[out_pos] = static_cast<int64_t>(final_next_rank[k]);
            } else {
                out_next[out_pos] = 0;
                mask_next[out_pos] = false;
                dist_next[out_pos] = INF_DIST;
                pos_next[out_pos] = 0;
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> feature_topk_v25_cuda_forward(
    torch::Tensor indices,
    torch::Tensor index_map,
    torch::Tensor features,
    torch::Tensor window_offsets,
    int64_t temporal_dilation,
    int64_t topk
) {
    const c10::cuda::CUDAGuard device_guard(indices.device());

    const int64_t N64 = indices.size(0);
    const int64_t C64 = features.size(1);
    const int64_t B64 = index_map.size(0);
    const int64_t T64 = index_map.size(1);
    const int64_t H64 = index_map.size(2);
    const int64_t W64 = index_map.size(3);
    const int64_t O64 = window_offsets.size(0);

    TORCH_CHECK(temporal_dilation == 1, "feature_topk_v25_cuda_ext is specialized for temporal_dilation=1");
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N is too large for int32 kernel indexing");
    TORCH_CHECK(C64 <= std::numeric_limits<int>::max(), "C is too large for int32 kernel indexing");
    TORCH_CHECK(B64 <= std::numeric_limits<int>::max(), "B is too large for int32 kernel indexing");
    TORCH_CHECK(T64 <= std::numeric_limits<int>::max(), "T is too large for int32 kernel indexing");
    TORCH_CHECK(H64 <= std::numeric_limits<int>::max(), "H is too large for int32 kernel indexing");
    TORCH_CHECK(W64 <= std::numeric_limits<int>::max(), "W is too large for int32 kernel indexing");
    TORCH_CHECK(O64 <= std::numeric_limits<int>::max(), "number of window offsets is too large for int32 kernel indexing");
    TORCH_CHECK(topk > 0 && topk <= std::numeric_limits<int>::max(), "topk must be a positive int32 value");

    const int N = static_cast<int>(N64);
    const int C = static_cast<int>(C64);
    const int B = static_cast<int>(B64);
    const int T = static_cast<int>(T64);
    const int H = static_cast<int>(H64);
    const int W = static_cast<int>(W64);
    const int O = static_cast<int>(O64);
    const int K = static_cast<int>(topk);

    auto long_opts = torch::TensorOptions().device(indices.device()).dtype(torch::kInt64);
    auto bool_opts = torch::TensorOptions().device(indices.device()).dtype(torch::kBool);
    auto float_opts = torch::TensorOptions().device(indices.device()).dtype(torch::kFloat32);

    auto out_prev = torch::empty({N64, topk}, long_opts);
    auto out_cur = torch::empty({N64, topk}, long_opts);
    auto out_next = torch::empty({N64, topk}, long_opts);
    auto mask_prev = torch::empty({N64, topk}, bool_opts);
    auto mask_cur = torch::empty({N64, topk}, bool_opts);
    auto mask_next = torch::empty({N64, topk}, bool_opts);
    auto dist_prev = torch::empty({N64, topk}, float_opts);
    auto dist_cur = torch::empty({N64, topk}, float_opts);
    auto dist_next = torch::empty({N64, topk}, float_opts);
    auto pos_prev = torch::empty({N64, topk}, long_opts);
    auto pos_cur = torch::empty({N64, topk}, long_opts);
    auto pos_next = torch::empty({N64, topk}, long_opts);

    if (N == 0) {
        return {out_prev, out_cur, out_next, mask_prev, mask_cur, mask_next, dist_prev, dist_cur, dist_next, pos_prev, pos_cur, pos_next};
    }

    int threads = 128;
    while (threads > 32) {
        const int64_t shared_bytes = static_cast<int64_t>(K) *
            (static_cast<int64_t>(threads) * 9 + 9) *
            static_cast<int64_t>(sizeof(int));
        if (shared_bytes <= 48 * 1024) {
            break;
        }
        threads /= 2;
    }

    const int64_t shared_bytes64 = static_cast<int64_t>(K) *
        (static_cast<int64_t>(threads) * 9 + 9) *
        static_cast<int64_t>(sizeof(int));
    TORCH_CHECK(shared_bytes64 <= std::numeric_limits<int>::max(), "shared memory size is too large");
    const size_t shared_bytes = static_cast<size_t>(shared_bytes64);

    const dim3 blocks(N);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(features.scalar_type(), "feature_topk_v25_cuda_forward", [&] {
        feature_topk_v25_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            indices.data_ptr<int>(),
            index_map.data_ptr<int>(),
            features.data_ptr<scalar_t>(),
            window_offsets.data_ptr<int>(),
            out_prev.data_ptr<int64_t>(),
            out_cur.data_ptr<int64_t>(),
            out_next.data_ptr<int64_t>(),
            mask_prev.data_ptr<bool>(),
            mask_cur.data_ptr<bool>(),
            mask_next.data_ptr<bool>(),
            dist_prev.data_ptr<float>(),
            dist_cur.data_ptr<float>(),
            dist_next.data_ptr<float>(),
            pos_prev.data_ptr<int64_t>(),
            pos_cur.data_ptr<int64_t>(),
            pos_next.data_ptr<int64_t>(),
            N,
            C,
            B,
            T,
            H,
            W,
            O,
            K
        );
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_prev, out_cur, out_next, mask_prev, mask_cur, mask_next, dist_prev, dist_cur, dist_next, pos_prev, pos_cur, pos_next};
}
