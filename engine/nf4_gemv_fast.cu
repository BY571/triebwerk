/**
 * Optimized NF4 GEMV kernel for Jetson Orin (sm_87).
 *
 * Performance analysis (v4 microbenchmark):
 *   NF4 achieves 18-28 GB/s vs cuBLAS fp16 at 85-100 GB/s.
 *   Root cause: ~18 instructions per byte for NF4 dequantization
 *   (lookup + scale + FMA) vs ~4 instructions for fp16 (half2 FMA).
 *   This is a fundamental limitation of NF4 on CUDA cores.
 *   Instruction-level optimizations (byte tables, half2 input) gave
 *   no improvement or regressed due to shared memory bank conflicts.
 *
 * Architecture: one block processes ROWS_PER_BLOCK output rows.
 * Each warp handles part of the dot product for one row.
 * Block reduction across warps via shared memory.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

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

    const int first_row = blockIdx.x * ROWS_PER_BLOCK;
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

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* packed,
    const float* absmax,
    const half* input,
    half* output,
    int out_dim,
    int in_dim,
    int block_size,
    cudaStream_t stream
) {
    int n_blocks = (out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        packed, absmax, input, output,
        in_dim, out_dim, block_size
    );
}

} // extern "C"
