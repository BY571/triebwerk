/**
 * NF4 GEMV kernels for Jetson Orin (sm_87) — llama.cpp-inspired v5.
 *
 * Key techniques from llama.cpp Q4_0 kernels:
 * 1. half2 accumulation: __hfma2 does 2 FMAs in 1 instruction (2x throughput)
 * 2. Dequantize to registers (not shared memory): avoids smem bank conflicts
 * 3. Warp-shuffle reduction: no shared memory needed for dot product reduction
 * 4. Vectorized loads where beneficial
 *
 * Target: 40-50 GB/s (llama.cpp achieves this on Orin Nano) vs our current 18-28 GB/s.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// NF4 lookup table in constant memory (broadcast to all threads, no bank conflicts)
__device__ __constant__ float c_nf4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// Core NF4 GEMV: each block computes ROWS_PER_BLOCK output elements
__device__ __forceinline__ void nf4_gemv_block(
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

    // Shared memory only for cross-warp reduction (tiny: 4*8*4 = 128 bytes)
    extern __shared__ float s_sums[];

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        // Accumulate in half2 for 2x FMA throughput
        half2 h2_sum = __float2half2_rn(0.0f);

        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row;
             byte_idx += THREADS_PER_BLOCK) {
            // Load weight byte
            uint8_t packed = __ldg(&weight_data[row_byte_start + byte_idx]);

            // Dequantize to registers using constant memory lookup
            int j0 = byte_idx * 2;
            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            half w0 = __float2half(c_nf4[packed >> 4] * scale);
            half w1 = __float2half(c_nf4[packed & 0x0F] * scale);

            // Load input as half2 (vectorized: 1 load instead of 2)
            half2 x_pair = __ldg(reinterpret_cast<const half2*>(&input[j0]));

            // half2 FMA: 2 multiplies + 2 adds in 1 instruction
            half2 w_pair = __halves2half2(w0, w1);
            h2_sum = __hfma2(w_pair, x_pair, h2_sum);
        }

        // Convert half2 accumulator to float for warp reduction
        float sum = __half2float(h2_sum.x) + __half2float(h2_sum.y);

        // Warp-level reduction via shuffle (no shared memory needed)
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Cross-warp reduction via shared memory (only lane 0 of each warp)
        if (lane_id == 0) {
            s_sums[r * N_WARPS + warp_id] = sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float total = (lane_id < N_WARPS) ? s_sums[r * N_WARPS + lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            }
            if (lane_id == 0) {
                output[row] = __float2half(total);
            }
        }
        __syncthreads(); // ensure s_sums can be reused for next row
    }
}

// ============================================================================
// Single-projection kernel
// ============================================================================

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int in_dim, int out_dim, int block_size
) {
    int first_row = blockIdx.x * ROWS_PER_BLOCK;
    nf4_gemv_block(weight_data, absmax, input, output, first_row, out_dim, in_dim);
}

// ============================================================================
// Fused 2-projection kernel (gate + up)
// ============================================================================

__global__ void nf4_fused_2_kernel(
    const uint8_t* __restrict__ a_weight, const float* __restrict__ a_absmax,
    half* __restrict__ a_output, int a_out_dim,
    const uint8_t* __restrict__ b_weight, const float* __restrict__ b_absmax,
    half* __restrict__ b_output, int b_out_dim,
    const half* __restrict__ input, int in_dim
) {
    int a_blocks = (a_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < a_blocks) {
        nf4_gemv_block(a_weight, a_absmax, input, a_output,
                       blockIdx.x * ROWS_PER_BLOCK, a_out_dim, in_dim);
    } else {
        nf4_gemv_block(b_weight, b_absmax, input, b_output,
                       (blockIdx.x - a_blocks) * ROWS_PER_BLOCK, b_out_dim, in_dim);
    }
}

// ============================================================================
// Fused 3-projection kernel (Q + K + V)
// ============================================================================

__global__ void nf4_fused_3_kernel(
    const uint8_t* __restrict__ a_weight, const float* __restrict__ a_absmax,
    half* __restrict__ a_output, int a_out_dim,
    const uint8_t* __restrict__ b_weight, const float* __restrict__ b_absmax,
    half* __restrict__ b_output, int b_out_dim,
    const uint8_t* __restrict__ c_weight, const float* __restrict__ c_absmax,
    half* __restrict__ c_output, int c_out_dim,
    const half* __restrict__ input, int in_dim
) {
    int a_blocks = (a_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int b_blocks = (b_out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < a_blocks) {
        nf4_gemv_block(a_weight, a_absmax, input, a_output,
                       blockIdx.x * ROWS_PER_BLOCK, a_out_dim, in_dim);
    } else if (blockIdx.x < a_blocks + b_blocks) {
        nf4_gemv_block(b_weight, b_absmax, input, b_output,
                       (blockIdx.x - a_blocks) * ROWS_PER_BLOCK, b_out_dim, in_dim);
    } else {
        nf4_gemv_block(c_weight, c_absmax, input, c_output,
                       (blockIdx.x - a_blocks - b_blocks) * ROWS_PER_BLOCK, c_out_dim, in_dim);
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
    size_t smem = ROWS_PER_BLOCK * N_WARPS * sizeof(float);
    nf4_gemv_fast_kernel<<<n_blocks, THREADS_PER_BLOCK, smem, stream>>>(
        packed, absmax, input, output, in_dim, out_dim, block_size);
}

void launch_nf4_fused_2(
    const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim,
    const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim,
    const half* input, int in_dim,
    cudaStream_t stream
) {
    int total = (a_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
              + (b_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    size_t smem = ROWS_PER_BLOCK * N_WARPS * sizeof(float);
    nf4_fused_2_kernel<<<total, THREADS_PER_BLOCK, smem, stream>>>(
        a_w, a_abs, a_out, a_dim, b_w, b_abs, b_out, b_dim, input, in_dim);
}

void launch_nf4_fused_3(
    const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim,
    const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim,
    const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim,
    const half* input, int in_dim,
    cudaStream_t stream
) {
    int total = (a_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
              + (b_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
              + (c_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    size_t smem = ROWS_PER_BLOCK * N_WARPS * sizeof(float);
    nf4_fused_3_kernel<<<total, THREADS_PER_BLOCK, smem, stream>>>(
        a_w, a_abs, a_out, a_dim, b_w, b_abs, b_out, b_dim,
        c_w, c_abs, c_out, c_dim, input, in_dim);
}

} // extern "C"
