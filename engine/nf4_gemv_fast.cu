/**
 * 4-bit GEMV kernels for Jetson Orin (sm_87) — v9 (dp4a).
 *
 * Three kernel variants:
 *   1. NF4 GEMV: shared memory lookup, fp32 FMA (baseline)
 *   2. Q4L GEMV: linear dequant, fp32 FMA (no lookup table)
 *   3. Q4L dp4a GEMV: quantize input to int8, use dp4a for 4 MACs/instruction
 *
 * dp4a (dot product of 4 int8 pairs, accumulated in int32) does 4x more work
 * per instruction than fp32 FMA. Adapted from llama.cpp Q4_0 kernel design.
 * Input quantized to int8 on-the-fly (once, shared across all projections).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define ROWS_PER_BLOCK 4
#define THREADS_PER_BLOCK 256
#define N_WARPS (THREADS_PER_BLOCK / 32)

// ============================================================================
// Q4 Linear GEMV: dequant = (nibble - 8) * scale (NO lookup table)
// ============================================================================

__device__ __forceinline__ void q4l_gemv_block(
    float* __restrict__ s_sums,
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ scales,  // one fp32 scale per 64-element block
    const half* __restrict__ input,
    half* __restrict__ output,
    int first_row, int out_dim, int in_dim
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;
    const int vec_per_row = bytes_per_row >> 2;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int scale_row_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += THREADS_PER_BLOCK) {
            uint32_t packed4 = __ldg(&row_u32[vi]);
            int base_j = vi << 3;
            float scale = __ldg(&scales[scale_row_start + (base_j >> 6)]);

            // dp4a-friendly packing: low nibbles = elem[0..3], high nibbles = elem[4..7]
            // Byte k: lo_nib = elem[k], hi_nib = elem[k+4]
            uint8_t b0 = packed4 & 0xFF;
            uint8_t b1 = (packed4 >> 8) & 0xFF;
            uint8_t b2 = (packed4 >> 16) & 0xFF;
            uint8_t b3 = (packed4 >> 24) & 0xFF;

            // Elements 0-3 (low nibbles of bytes 0-3)
            float w0 = ((float)(b0 & 0xF) - 8.0f) * scale;
            float w1 = ((float)(b1 & 0xF) - 8.0f) * scale;
            float w2 = ((float)(b2 & 0xF) - 8.0f) * scale;
            float w3 = ((float)(b3 & 0xF) - 8.0f) * scale;
            sum = __fmaf_rn(w0, __half2float(__ldg(&input[base_j])), sum);
            sum = __fmaf_rn(w1, __half2float(__ldg(&input[base_j + 1])), sum);
            sum = __fmaf_rn(w2, __half2float(__ldg(&input[base_j + 2])), sum);
            sum = __fmaf_rn(w3, __half2float(__ldg(&input[base_j + 3])), sum);

            // Elements 4-7 (high nibbles of bytes 0-3)
            float w4 = ((float)(b0 >> 4) - 8.0f) * scale;
            float w5 = ((float)(b1 >> 4) - 8.0f) * scale;
            float w6 = ((float)(b2 >> 4) - 8.0f) * scale;
            float w7 = ((float)(b3 >> 4) - 8.0f) * scale;
            sum = __fmaf_rn(w4, __half2float(__ldg(&input[base_j + 4])), sum);
            sum = __fmaf_rn(w5, __half2float(__ldg(&input[base_j + 5])), sum);
            sum = __fmaf_rn(w6, __half2float(__ldg(&input[base_j + 6])), sum);
            sum = __fmaf_rn(w7, __half2float(__ldg(&input[base_j + 7])), sum);
        }

        // No remainder needed: all Qwen3 dims are multiples of 8

        // Warp + cross-warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) s_sums[r * N_WARPS + warp_id] = sum;
        __syncthreads();
        if (warp_id == 0) {
            float total = (lane_id < N_WARPS) ? s_sums[r * N_WARPS + lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            if (lane_id == 0) output[row] = __float2half(total);
        }
        __syncthreads();
    }
}

// ============================================================================
// NF4 GEMV: non-linear, shared memory lookup table (kept as fallback)
// ============================================================================

__device__ __forceinline__ void nf4_gemv_block(
    const float* __restrict__ s_qmap,
    float* __restrict__ s_sums,
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ absmax,
    const half* __restrict__ input,
    half* __restrict__ output,
    int first_row, int out_dim, int in_dim
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;
    const int vec_per_row = bytes_per_row >> 2;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int absmax_row_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += THREADS_PER_BLOCK) {
            uint32_t packed4 = __ldg(&row_u32[vi]);
            int base_j = vi << 3;
            float scale = __ldg(&absmax[absmax_row_start + (base_j >> 6)]);

            uint8_t b0 = packed4 & 0xFF;
            sum = __fmaf_rn(s_qmap[b0 >> 4] * scale, __half2float(__ldg(&input[base_j])), sum);
            sum = __fmaf_rn(s_qmap[b0 & 0xF] * scale, __half2float(__ldg(&input[base_j + 1])), sum);

            uint8_t b1 = (packed4 >> 8) & 0xFF;
            sum = __fmaf_rn(s_qmap[b1 >> 4] * scale, __half2float(__ldg(&input[base_j + 2])), sum);
            sum = __fmaf_rn(s_qmap[b1 & 0xF] * scale, __half2float(__ldg(&input[base_j + 3])), sum);

            uint8_t b2 = (packed4 >> 16) & 0xFF;
            sum = __fmaf_rn(s_qmap[b2 >> 4] * scale, __half2float(__ldg(&input[base_j + 4])), sum);
            sum = __fmaf_rn(s_qmap[b2 & 0xF] * scale, __half2float(__ldg(&input[base_j + 5])), sum);

            uint8_t b3 = (packed4 >> 24) & 0xFF;
            sum = __fmaf_rn(s_qmap[b3 >> 4] * scale, __half2float(__ldg(&input[base_j + 6])), sum);
            sum = __fmaf_rn(s_qmap[b3 & 0xF] * scale, __half2float(__ldg(&input[base_j + 7])), sum);
        }

        int rem_start = vec_per_row << 2;
        for (int bi = rem_start + threadIdx.x; bi < bytes_per_row; bi += THREADS_PER_BLOCK) {
            uint8_t packed = __ldg(&weight_data[row_byte_start + bi]);
            int j0 = bi * 2;
            float sc = __ldg(&absmax[absmax_row_start + (j0 >> 6)]);
            sum = __fmaf_rn(s_qmap[packed >> 4] * sc, __half2float(__ldg(&input[j0])), sum);
            sum = __fmaf_rn(s_qmap[packed & 0xF] * sc, __half2float(__ldg(&input[j0 + 1])), sum);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) s_sums[r * N_WARPS + warp_id] = sum;
        __syncthreads();
        if (warp_id == 0) {
            float total = (lane_id < N_WARPS) ? s_sums[r * N_WARPS + lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                total += __shfl_down_sync(0xFFFFFFFF, total, offset);
            if (lane_id == 0) output[row] = __float2half(total);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void init_nf4_table(float* s_qmap) {
    if (threadIdx.x < 16) {
        const float T[16] = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
        };
        s_qmap[threadIdx.x] = T[threadIdx.x];
    }
    __syncthreads();
}

// ============================================================================
// Input quantization: fp16 → int8 (per-64-element blocks)
// ============================================================================

__global__ void quantize_input_q8_kernel(
    const half* __restrict__ input,   // (dim,) fp16
    int8_t* __restrict__ q8_data,     // (dim,) int8 output
    float* __restrict__ q8_scales,    // (dim/64,) per-block scale
    float* __restrict__ q8_sums,      // (dim/64,) per-block sum of q8
    int dim
) {
    const int block_size = 64;
    int blk = blockIdx.x;
    int base = blk * block_size;
    if (base >= dim) return;

    // Phase 1: find max absolute value in this block
    __shared__ float s_max[256];
    float my_max = 0.0f;
    for (int i = threadIdx.x; i < block_size && base + i < dim; i += blockDim.x) {
        float v = fabsf(__half2float(input[base + i]));
        my_max = fmaxf(my_max, v);
    }
    s_max[threadIdx.x] = my_max;
    __syncthreads();
    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        __syncthreads();
    }
    float scale = s_max[0] / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;

    // Phase 2: quantize and compute sum
    __shared__ int s_sum[256];
    int my_sum = 0;
    for (int i = threadIdx.x; i < block_size && base + i < dim; i += blockDim.x) {
        float v = __half2float(input[base + i]);
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        q8_data[base + i] = (int8_t)q;
        my_sum += q;
    }
    s_sum[threadIdx.x] = my_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        q8_scales[blk] = scale;
        q8_sums[blk] = (float)s_sum[0];
    }
}

// ============================================================================
// dp4a Q4L GEMV: integer dot product, 4 MACs per instruction
// ============================================================================

// dp4a GEMV with shared memory input caching.
// The q8 input vector is loaded into shared memory ONCE per block,
// then reused across all ROWS_PER_BLOCK rows. This eliminates
// redundant global loads (each row was re-reading the same input).
// Dynamic shared memory: s_sums[RPB*nwarps] + s_q8[in_dim] + s_q8_sc[in_dim/64]

__device__ __forceinline__ void q4l_dp4a_gemv_block(
    float* __restrict__ s_sums,    // shared: [ROWS_PER_BLOCK * n_warps]
    int8_t* __restrict__ s_q8,     // shared: [in_dim] cached input
    float* __restrict__ s_q8_sc,   // shared: [blocks_per_row] cached scales
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ w_scales,
    const int8_t* __restrict__ q8_data,
    const float* __restrict__ q8_scales,
    const float* __restrict__ q8_sums_arr,
    half* __restrict__ output,
    int first_row, int out_dim, int in_dim
) {
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;
    const int vec_per_row = bytes_per_row >> 2;

    // Cooperatively load q8 input + scales into shared memory (once per block)
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        s_q8[i] = q8_data[i];
    for (int i = threadIdx.x; i < blocks_per_row; i += blockDim.x)
        s_q8_sc[i] = q8_scales[i];
    __syncthreads();

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        float sum = 0.0f;
        const int row_byte_start = row * bytes_per_row;
        const int w_scale_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += (int)blockDim.x) {
            uint32_t packed = __ldg(&row_u32[vi]);
            int base_elem = vi << 3;
            int w_blk = base_elem >> 6;
            float w_sc = __ldg(&w_scales[w_scale_start + w_blk]);
            float x_sc = s_q8_sc[w_blk];  // from shared memory

            int vi_lo = (packed >> 0) & 0x0F0F0F0F;
            int vi_hi = (packed >> 4) & 0x0F0F0F0F;

            // Read q8 from shared memory instead of global
            const int* sq8_i32 = reinterpret_cast<const int*>(&s_q8[base_elem]);
            int u0 = sq8_i32[0];
            int u1 = sq8_i32[1];

            int sumi = 0;
            sumi = __dp4a(vi_lo, u0, sumi);
            sumi = __dp4a(vi_hi, u1, sumi);

            int q8_sum_local = 0;
            q8_sum_local = __dp4a(0x01010101, u0, q8_sum_local);
            q8_sum_local = __dp4a(0x01010101, u1, q8_sum_local);

            sum += w_sc * x_sc * ((float)sumi - 8.0f * (float)q8_sum_local);
        }

        // Warp + cross-warp reduction (dynamic n_warps)
        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
        if (lane_id == 0) s_sums[r * n_warps + warp_id] = sum;
        __syncthreads();
        if (warp_id == 0) {
            float total = (lane_id < n_warps) ? s_sums[r * n_warps + lane_id] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off /= 2)
                total += __shfl_down_sync(0xFFFFFFFF, total, off);
            if (lane_id == 0) output[row] = __float2half(total);
        }
        __syncthreads();
    }
}

// ============================================================================
// Fused W4A16 batch GEMM: read 4-bit weights ONCE, multiply against G inputs
// Eliminates the fp16 dequant buffer (830MB) entirely.
// Each block computes ROWS_PER_BLOCK rows of the output for ALL G columns.
// Weight layout: Q4L (dp4a-friendly: lo nibbles = elem[0..3], hi = elem[4..7])
// Input: fp16 (in_dim, G) column-major
// Output: fp16 (out_dim, G) column-major
// ============================================================================

__global__ __launch_bounds__(256, 4)
void q4l_batch_gemm_kernel(
    const uint8_t* __restrict__ weight_data,
    const float* __restrict__ w_scales,   // per-block scale factors
    const half* __restrict__ input,       // (in_dim, G) column-major fp16
    half* __restrict__ output,            // (out_dim, G) column-major fp16
    int in_dim, int out_dim, int G
) {
    const int n_warps = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int bytes_per_row = in_dim / 2;
    const int blocks_per_row = in_dim >> 6;
    const int vec_per_row = bytes_per_row >> 2;

    // Shared memory: sums[RPB * G * n_warps]
    extern __shared__ float s_sums[];

    int first_row = blockIdx.x * ROWS_PER_BLOCK;

    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int row = first_row + r;
        if (row >= out_dim) break;

        // Accumulate G output values per row
        float sums[8] = {};  // max G=8, registers

        const int row_byte_start = row * bytes_per_row;
        const int w_scale_start = row * blocks_per_row;
        const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(
            weight_data + row_byte_start);

        for (int vi = threadIdx.x; vi < vec_per_row; vi += (int)blockDim.x) {
            uint32_t packed = __ldg(&row_u32[vi]);
            int base_elem = vi << 3;
            int w_blk = base_elem >> 6;
            float w_sc = __ldg(&w_scales[w_scale_start + w_blk]);

            // Dequant 8 weight elements to fp32
            uint8_t b0 = packed & 0xFF;
            uint8_t b1 = (packed >> 8) & 0xFF;
            uint8_t b2 = (packed >> 16) & 0xFF;
            uint8_t b3 = (packed >> 24) & 0xFF;
            float w[8];
            w[0] = ((float)(b0 & 0xF) - 8.0f) * w_sc;
            w[1] = ((float)(b1 & 0xF) - 8.0f) * w_sc;
            w[2] = ((float)(b2 & 0xF) - 8.0f) * w_sc;
            w[3] = ((float)(b3 & 0xF) - 8.0f) * w_sc;
            w[4] = ((float)(b0 >> 4) - 8.0f) * w_sc;
            w[5] = ((float)(b1 >> 4) - 8.0f) * w_sc;
            w[6] = ((float)(b2 >> 4) - 8.0f) * w_sc;
            w[7] = ((float)(b3 >> 4) - 8.0f) * w_sc;

            // Multiply against all G input columns (weight reuse!)
            for (int g = 0; g < G; g++) {
                const half* x_col = input + (size_t)g * in_dim;
                float s = 0.0f;
                s = __fmaf_rn(w[0], __half2float(__ldg(&x_col[base_elem])), s);
                s = __fmaf_rn(w[1], __half2float(__ldg(&x_col[base_elem+1])), s);
                s = __fmaf_rn(w[2], __half2float(__ldg(&x_col[base_elem+2])), s);
                s = __fmaf_rn(w[3], __half2float(__ldg(&x_col[base_elem+3])), s);
                s = __fmaf_rn(w[4], __half2float(__ldg(&x_col[base_elem+4])), s);
                s = __fmaf_rn(w[5], __half2float(__ldg(&x_col[base_elem+5])), s);
                s = __fmaf_rn(w[6], __half2float(__ldg(&x_col[base_elem+6])), s);
                s = __fmaf_rn(w[7], __half2float(__ldg(&x_col[base_elem+7])), s);
                sums[g] += s;
            }
        }

        // Warp + cross-warp reduction for each G column
        for (int g = 0; g < G; g++) {
            float sum = sums[g];
            #pragma unroll
            for (int off = 16; off > 0; off /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
            if (lane_id == 0) s_sums[g * n_warps + warp_id] = sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            for (int g = 0; g < G; g++) {
                float total = (lane_id < n_warps) ? s_sums[g * n_warps + lane_id] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    total += __shfl_down_sync(0xFFFFFFFF, total, off);
                if (lane_id == 0) output[row + (size_t)g * out_dim] = __float2half(total);
            }
        }
        __syncthreads();
    }
}

// dp4a kernel wrapper with dynamic shared memory
// Layout: s_sums[RPB * n_warps] + s_q8[in_dim] + s_q8_sc[in_dim/64]
__global__ __launch_bounds__(256, 6)
void q4l_dp4a_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ w_scales,
    const int8_t* __restrict__ q8, const float* __restrict__ q8_sc,
    const float* __restrict__ q8_sm, half* __restrict__ y,
    int in_dim, int out_dim
) {
    extern __shared__ char smem[];
    int n_warps = blockDim.x / 32;

    float* s_sums = reinterpret_cast<float*>(smem);
    int8_t* s_q8 = reinterpret_cast<int8_t*>(s_sums + ROWS_PER_BLOCK * n_warps);
    float* s_q8_sc = reinterpret_cast<float*>(s_q8 + in_dim);

    q4l_dp4a_gemv_block(s_sums, s_q8, s_q8_sc, w, w_scales, q8, q8_sc, q8_sm, y,
                        blockIdx.x * ROWS_PER_BLOCK, out_dim, in_dim);
}

// ============================================================================
// Kernel wrappers (use Q4L when is_q4l=1, NF4 otherwise)
// ============================================================================

__global__ void nf4_gemv_fast_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ a,
    const half* __restrict__ x, half* __restrict__ y,
    int in_dim, int out_dim, int block_size, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int first_row = blockIdx.x * ROWS_PER_BLOCK;
    if (is_q4l)
        q4l_gemv_block(s_sums, w, a, x, y, first_row, out_dim, in_dim);
    else
        nf4_gemv_block(s_qmap, s_sums, w, a, x, y, first_row, out_dim, in_dim);
}

__global__ void nf4_fused_2_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const half* __restrict__ x, int in_dim, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (blockIdx.x < ab) {
        if (is_q4l) q4l_gemv_block(s_sums, aw, aa, x, ay, blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay, blockIdx.x * ROWS_PER_BLOCK, ad, in_dim);
    } else {
        if (is_q4l) q4l_gemv_block(s_sums, bw, ba, x, by, (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by, (blockIdx.x - ab) * ROWS_PER_BLOCK, bd, in_dim);
    }
}

__global__ void nf4_fused_3_kernel(
    const uint8_t* __restrict__ aw, const float* __restrict__ aa,
    half* __restrict__ ay, int ad,
    const uint8_t* __restrict__ bw, const float* __restrict__ ba,
    half* __restrict__ by, int bd,
    const uint8_t* __restrict__ cw, const float* __restrict__ ca,
    half* __restrict__ cy, int cd,
    const half* __restrict__ x, int in_dim, int is_q4l
) {
    __shared__ float s_qmap[16];
    __shared__ float s_sums[ROWS_PER_BLOCK * N_WARPS];
    if (!is_q4l) init_nf4_table(s_qmap);
    int ab = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int bb = (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int first;
    if (blockIdx.x < ab) {
        first = blockIdx.x * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, aw, aa, x, ay, first, ad, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, aw, aa, x, ay, first, ad, in_dim);
    } else if (blockIdx.x < ab + bb) {
        first = (blockIdx.x - ab) * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, bw, ba, x, by, first, bd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, bw, ba, x, by, first, bd, in_dim);
    } else {
        first = (blockIdx.x - ab - bb) * ROWS_PER_BLOCK;
        if (is_q4l) q4l_gemv_block(s_sums, cw, ca, x, cy, first, cd, in_dim);
        else nf4_gemv_block(s_qmap, s_sums, cw, ca, x, cy, first, cd, in_dim);
    }
}

// ============================================================================
// Launch functions — is_q4l=0 for NF4, is_q4l=1 for Q4 Linear
// ============================================================================

extern "C" {

void launch_nf4_gemv_fast(
    const uint8_t* p, const float* a, const half* x, half* y,
    int od, int id, int bs, cudaStream_t s
) {
    int nb = (od + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_gemv_fast_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(p, a, x, y, id, od, bs, 0);
}

void launch_nf4_fused_2(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, x, id, 0);
}

void launch_q4l_fused_2(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_2_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, x, id, 1);
}

void launch_nf4_fused_3(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const uint8_t* cw, const float* ca, half* cy, int cd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (cd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_3_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, cw, ca, cy, cd, x, id, 0);
}

void launch_q4l_fused_3(
    const uint8_t* aw, const float* aa, half* ay, int ad,
    const uint8_t* bw, const float* ba, half* by, int bd,
    const uint8_t* cw, const float* ca, half* cy, int cd,
    const half* x, int id, cudaStream_t s
) {
    int nb = (ad + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (bd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK
           + (cd + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    nf4_fused_3_kernel<<<nb, THREADS_PER_BLOCK, 0, s>>>(aw, aa, ay, ad, bw, ba, by, bd, cw, ca, cy, cd, x, id, 1);
}

// Quantize fp16 input to int8 for dp4a
void launch_quantize_input_q8(
    const half* input, int8_t* q8_data, float* q8_scales, float* q8_sums,
    int dim, cudaStream_t stream
) {
    int n_blocks = (dim + 63) / 64;
    quantize_input_q8_kernel<<<n_blocks, 64, 0, stream>>>(
        input, q8_data, q8_scales, q8_sums, dim);
}

// dp4a Q4L GEMV — dynamic thread count + shared memory input caching
void launch_q4l_dp4a_gemv(
    const uint8_t* w, const float* w_scales,
    const int8_t* q8, const float* q8_sc, const float* q8_sm,
    half* y, int out_dim, int in_dim, cudaStream_t stream
) {
    int nb = (out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    // Dynamic thread count: 128 for small dims (fix 50% idle), 256 for larger
    int vec_per_row = in_dim / 2 / 4;
    int threads = (vec_per_row <= 128) ? 128 : 256;
    int n_warps = threads / 32;
    int blocks_per_row = in_dim / 64;
    // Shared memory: s_sums + s_q8 input + s_q8_scales
    size_t smem = ROWS_PER_BLOCK * n_warps * sizeof(float)
                + in_dim * sizeof(int8_t)
                + blocks_per_row * sizeof(float);
    q4l_dp4a_kernel<<<nb, threads, smem, stream>>>(
        w, w_scales, q8, q8_sc, q8_sm, y, in_dim, out_dim);
}

// Fused W4A16 batch GEMM: read Q4L weights once, multiply against G fp16 inputs
void launch_q4l_batch_gemm(
    const uint8_t* w, const float* w_scales,
    const half* input, half* output,
    int out_dim, int in_dim, int G, cudaStream_t stream
) {
    int nb = (out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int threads = (in_dim / 2 / 4 <= 128) ? 128 : 256;
    int n_warps = threads / 32;
    size_t smem = G * n_warps * sizeof(float);
    q4l_batch_gemm_kernel<<<nb, threads, smem, stream>>>(
        w, w_scales, input, output, in_dim, out_dim, G);
}

// ============================================================================
// Q4L dequantization: packed uint8 + scales -> fp16 (for batched GEMM path)
// ============================================================================

// Dequant Q4L to fp16 for batched GEMM
__global__ void dequant_q4l_kernel(
    half* __restrict__ out,
    const uint8_t* __restrict__ data,
    const float* __restrict__ scales,
    int out_dim, int in_dim
) {
    int bytes_total = out_dim * in_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes_total) return;

    uint8_t packed = data[idx];
    int group = idx / 4;
    int byte_in_group = idx % 4;
    int elem_lo = group * 8 + byte_in_group;
    int elem_hi = elem_lo + 4;

    int block_idx = elem_lo / 64;
    float scale = scales[block_idx];

    out[elem_lo] = __float2half(((float)(packed & 0x0F) - 8.0f) * scale);
    out[elem_hi] = __float2half(((float)(packed >> 4) - 8.0f) * scale);
}

void launch_dequant_q4l(
    half* out, const uint8_t* data, const float* scales,
    int out_dim, int in_dim, cudaStream_t stream
) {
    int bytes_total = out_dim * in_dim / 2;
    int threads = 256;
    int blocks = (bytes_total + threads - 1) / threads;
    dequant_q4l_kernel<<<blocks, threads, 0, stream>>>(
        out, data, scales, out_dim, in_dim);
}

} // extern "C"
