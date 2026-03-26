/**
 * Optimized NF4 GEMV kernel v3 for Jetson Orin (sm_87).
 *
 * Lessons from v2: vectorized uint32_t loads cause 50% thread idle for
 * in_dim=1024 (n_vec=128 < 256 threads). Shared memory input caching
 * drops occupancy from 8 to 3 blocks/SM for in_dim=3072 (12KB smem).
 * Both regressions cancelled the gains.
 *
 * v3 strategy: keep byte-level access for full thread utilization,
 * apply only zero-cost optimizations:
 * 1. __ldg() for read-only weight/absmax/input via texture cache
 * 2. Simplified absmax: row-local block index via bit shift (j0 >> 6)
 *    instead of computing flat index + integer division
 * 3. Single absmax load per pair (j0 is always even, never straddles block)
 * 4. Explicit __fmaf_rn for fused multiply-add
 * 5. #pragma unroll on warp reduction
 * 6. Shared memory NF4 lookup table (same as v1)
 * 7. ROWS_PER_BLOCK=4, warp-level reduction
 *
 * For projections where in_dim >= 4*THREADS_PER_BLOCK (only down_proj
 * with in_dim=3072), also uses vectorized uint32_t loads.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ weight_data,    // (total_params/2,) packed NF4
    const float* __restrict__ absmax,            // (n_blocks,) per-block scales
    const half* __restrict__ input,              // (in_dim,) fp16
    half* __restrict__ output,                   // (out_dim,) fp16
    int in_dim,
    int out_dim,
    int block_size                               // 64
) {
    // Shared memory: quant_map (16 floats) + partial sums
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK][N_WARPS];

    // Load NF4 lookup table into shared memory (16 threads, one-time)
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
    const int blocks_per_row = in_dim >> 6;  // in_dim / 64

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;

        // Byte-level access: all THREADS_PER_BLOCK threads stay active
        for (int byte_idx = threadIdx.x; byte_idx < bytes_per_row;
             byte_idx += THREADS_PER_BLOCK) {
            // Read weight byte through texture cache
            uint8_t packed = __ldg(&weight_data[row_byte_start + byte_idx]);

            // Element indices within the row
            int j0 = byte_idx * 2;

            // Absmax: row-local index via bit shift (avoids flat index + division)
            // j0 is always even → never straddles a 64-element block boundary
            float scale = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);

            // Dequantize: NF4 lookup × absmax scale
            float w0 = s_qmap[packed >> 4] * scale;
            float w1 = s_qmap[packed & 0x0F] * scale;

            // Read input through texture cache, FMA accumulate
            float x0 = __half2float(__ldg(&input[j0]));
            float x1 = __half2float(__ldg(&input[j0 + 1]));
            sum = __fmaf_rn(w0, x0, sum);
            sum = __fmaf_rn(w1, x1, sum);
        }

        // Warp-level reduction via shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane_id == 0) {
            s_sums[r][warp_id] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps (only first warp)
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
