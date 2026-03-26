/**
 * Optimized NF4 GEMV kernels for Jetson Orin (sm_87).
 *
 * Includes:
 * 1. Single-projection NF4 GEMV (launch_nf4_gemv_fast)
 * 2. Fused QKV NF4 GEMV — 3 projections in 1 launch (launch_nf4_fused_qkv)
 * 3. Fused gate+up NF4 GEMV — 2 projections in 1 launch (launch_nf4_fused_2)
 *
 * Fusion reduces kernel launches from ~480 to ~320 per token
 * (saves 81 launches × ~10μs = ~0.8ms on Jetson ARM).
 * Also improves GPU scheduling (more blocks to choose from).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// ============================================================================
// Core NF4 dot product: one block processes ROWS_PER_BLOCK output rows
// from a single weight matrix. Used by both single and fused kernels.
// ============================================================================

__device__ __forceinline__ void nf4_gemv_block(
    const float* __restrict__ s_qmap,
    float s_sums[][N_WARPS],
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int first_row,
    int out_dim,
    int in_dim
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row;
             byte_idx += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + byte_idx]);
            int j0 = byte_idx * 2;

            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            float w0 = s_qmap[packed >> 4] * scale;
            float w1 = s_qmap[packed & 0x0F] * scale;

            float x0 = __half2float(__ldg(&input[j0]));
            float x1 = __half2float(__ldg(&input[j0 + 1]));
            sum = __fmaf_rn(w0, x0, sum);
            sum = __fmaf_rn(w1, x1, sum);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane_id == 0) {
            s_sums[r][warp_id] = sum;
        }
    }

    __syncthreads();

    if (warp_id == 0) {
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int row = first_row + r;
            if (row >= out_dim) break;

            float total = (lane_id < N_WARPS) ? s_sums[r][lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            }
            if (lane_id == 0) {
                output[row] = __float2half(total);
            }
        }
    }
}

// ============================================================================
// Single-projection NF4 GEMV kernel
// ============================================================================

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int in_dim,
    int out_dim,
    int block_size
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK][N_WARPS];

    if (threadIdx.x < 16) {
        const float NF4_TABLE[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    int first_row = blockIdx.x * ROWS_PER_BLOCK;
    nf4_gemv_block(s_qmap, s_sums, weight_data, absmax, input, output,
                   first_row, out_dim, in_dim);
}

// ============================================================================
// Fused 2-projection NF4 GEMV (gate + up, or any 2 with same input/in_dim)
// Single kernel launch replaces 2 separate launches.
// ============================================================================

__global__ void nf4_fused_2_kernel(
    // Projection A
    const uint8_t* __restrict__ a_weight, const float* __restrict__ a_absmax,
    half* __restrict__ a_output, int a_out_dim,
    // Projection B
    const uint8_t* __restrict__ b_weight, const float* __restrict__ b_absmax,
    half* __restrict__ b_output, int b_out_dim,
    // Shared input
    const half* __restrict__ input, int in_dim
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK][N_WARPS];

    if (threadIdx.x < 16) {
        const float NF4_TABLE[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    int a_blocks = (a_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int global_block = blockIdx.x;

    if (global_block < a_blocks) {
        // This block processes projection A
        int first_row = global_block * ROWS_PER_BLOCK;
        nf4_gemv_block(s_qmap, s_sums, a_weight, a_absmax, input, a_output,
                       first_row, a_out_dim, in_dim);
    } else {
        // This block processes projection B
        int first_row = (global_block - a_blocks) * ROWS_PER_BLOCK;
        nf4_gemv_block(s_qmap, s_sums, b_weight, b_absmax, input, b_output,
                       first_row, b_out_dim, in_dim);
    }
}

// ============================================================================
// Fused 3-projection NF4 GEMV (Q + K + V with same input/in_dim)
// Single kernel launch replaces 3 separate launches.
// ============================================================================

__global__ void nf4_fused_3_kernel(
    // Projection A (Q)
    const uint8_t* __restrict__ a_weight, const float* __restrict__ a_absmax,
    half* __restrict__ a_output, int a_out_dim,
    // Projection B (K)
    const uint8_t* __restrict__ b_weight, const float* __restrict__ b_absmax,
    half* __restrict__ b_output, int b_out_dim,
    // Projection C (V)
    const uint8_t* __restrict__ c_weight, const float* __restrict__ c_absmax,
    half* __restrict__ c_output, int c_out_dim,
    // Shared input
    const half* __restrict__ input, int in_dim
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK][N_WARPS];

    if (threadIdx.x < 16) {
        const float NF4_TABLE[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    int a_blocks = (a_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int b_blocks = (b_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int global_block = blockIdx.x;

    if (global_block < a_blocks) {
        int first_row = global_block * ROWS_PER_BLOCK;
        nf4_gemv_block(s_qmap, s_sums, a_weight, a_absmax, input, a_output,
                       first_row, a_out_dim, in_dim);
    } else if (global_block < a_blocks + b_blocks) {
        int first_row = (global_block - a_blocks) * ROWS_PER_BLOCK;
        nf4_gemv_block(s_qmap, s_sums, b_weight, b_absmax, input, b_output,
                       first_row, b_out_dim, in_dim);
    } else {
        int first_row = (global_block - a_blocks - b_blocks) * ROWS_PER_BLOCK;
        nf4_gemv_block(s_qmap, s_sums, c_weight, c_absmax, input, c_output,
                       first_row, c_out_dim, in_dim);
    }
}

// ============================================================================
// Launch functions
// ============================================================================

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* packed, const float* absmax,
    const half* input, half* output,
    int out_dim, int in_dim, int block_size,
    cudaStream_t stream
) {
    int n_blocks = (out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        packed, absmax, input, output, in_dim, out_dim, block_size);
}

void launch_nf4_fused_2(
    const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim,
    const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim,
    const half* input, int in_dim,
    cudaStream_t stream
) {
    int total_blocks = (a_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
                     + (b_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<total_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        a_w, a_abs, a_out, a_dim,
        b_w, b_abs, b_out, b_dim,
        input, in_dim);
}

void launch_nf4_fused_3(
    const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim,
    const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim,
    const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim,
    const half* input, int in_dim,
    cudaStream_t stream
) {
    int total_blocks = (a_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
                     + (b_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
                     + (c_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_3_kernel<<<total_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        a_w, a_abs, a_out, a_dim,
        b_w, b_abs, b_out, b_dim,
        c_w, c_abs, c_out, c_dim,
        input, in_dim);
}

} // extern "C"
