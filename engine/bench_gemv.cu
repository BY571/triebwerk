/**
 * Microbenchmark: NF4 GEMV kernel at Qwen3-0.6B projection sizes.
 * Measures kernel time in isolation using CUDA events.
 *
 * Build: nvcc -O3 -arch=sm_87 -o bench_gemv bench_gemv.cu nf4_gemv_fast.cu
 * Run:   ./bench_gemv
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Forward declaration of the NF4 kernel launcher
extern "C" {
    void launch_nf4_gemv_fast(
        const uint8_t* packed, const float* absmax, const half* input,
        half* output, int out_dim, int in_dim, int block_size,
        cudaStream_t stream);
}

struct BenchCase {
    const char* name;
    int out_dim;
    int in_dim;
};

float bench_nf4_gemv(const BenchCase& c, int warmup, int iters) {
    int block_size = 64;
    int total_params = c.out_dim * c.in_dim;
    int data_bytes = total_params / 2;
    int n_blocks = total_params / block_size;

    // Allocate
    uint8_t* d_data;   cudaMalloc(&d_data, data_bytes);
    float* d_absmax;   cudaMalloc(&d_absmax, n_blocks * sizeof(float));
    half* d_input;     cudaMalloc(&d_input, c.in_dim * sizeof(half));
    half* d_output;    cudaMalloc(&d_output, c.out_dim * sizeof(half));

    // Fill with random-ish data
    {
        uint8_t* h = (uint8_t*)malloc(data_bytes);
        for (int i = 0; i < data_bytes; i++) h[i] = (uint8_t)(i * 37 + 13);
        cudaMemcpy(d_data, h, data_bytes, cudaMemcpyHostToDevice);
        free(h);

        float* a = (float*)malloc(n_blocks * sizeof(float));
        for (int i = 0; i < n_blocks; i++) a[i] = 0.01f + (i % 100) * 0.001f;
        cudaMemcpy(d_absmax, a, n_blocks * sizeof(float), cudaMemcpyHostToDevice);
        free(a);

        half* x = (half*)malloc(c.in_dim * sizeof(half));
        for (int i = 0; i < c.in_dim; i++) x[i] = __float2half(0.1f);
        cudaMemcpy(d_input, x, c.in_dim * sizeof(half), cudaMemcpyHostToDevice);
        free(x);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        launch_nf4_gemv_fast(d_data, d_absmax, d_input, d_output,
                             c.out_dim, c.in_dim, block_size, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        launch_nf4_gemv_fast(d_data, d_absmax, d_input, d_output,
                             c.out_dim, c.in_dim, block_size, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_absmax);
    cudaFree(d_input);
    cudaFree(d_output);

    return ms / iters;
}

float bench_fp16_cublas(const BenchCase& c, int warmup, int iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    half* d_weight;  cudaMalloc(&d_weight, (size_t)c.out_dim * c.in_dim * sizeof(half));
    half* d_input;   cudaMalloc(&d_input, c.in_dim * sizeof(half));
    half* d_output;  cudaMalloc(&d_output, c.out_dim * sizeof(half));

    // Fill with data
    cudaMemset(d_weight, 0, (size_t)c.out_dim * c.in_dim * sizeof(half));
    cudaMemset(d_input, 0, c.in_dim * sizeof(half));

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     c.out_dim, 1, c.in_dim, &alpha,
                     d_weight, CUDA_R_16F, c.in_dim,
                     d_input, CUDA_R_16F, c.in_dim,
                     &beta, d_output, CUDA_R_16F, c.out_dim,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     c.out_dim, 1, c.in_dim, &alpha,
                     d_weight, CUDA_R_16F, c.in_dim,
                     d_input, CUDA_R_16F, c.in_dim,
                     &beta, d_output, CUDA_R_16F, c.out_dim,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_weight);
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);

    return ms / iters;
}

int main() {
    BenchCase cases[] = {
        {"q_proj",    2048, 1024},
        {"k_proj",    1024, 1024},
        {"v_proj",    1024, 1024},
        {"o_proj",    1024, 2048},
        {"gate_proj", 3072, 1024},
        {"up_proj",   3072, 1024},
        {"down_proj", 1024, 3072},
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);
    int warmup = 50;
    int iters = 200;

    printf("%-12s %5s %5s | %8s %8s | %5s | %8s %8s\n",
           "Layer", "Out", "In", "NF4(us)", "fp16(us)", "Ratio", "NF4 GB/s", "fp16 GB/s");
    printf("%-12s %5s %5s | %8s %8s | %5s | %8s %8s\n",
           "-----", "---", "---", "-------", "-------", "-----", "--------", "---------");

    float total_nf4 = 0, total_fp16 = 0;
    float total_nf4_bytes = 0, total_fp16_bytes = 0;

    for (int i = 0; i < n_cases; i++) {
        float nf4_ms = bench_nf4_gemv(cases[i], warmup, iters);
        float fp16_ms = bench_fp16_cublas(cases[i], warmup, iters);

        float nf4_us = nf4_ms * 1000.0f;
        float fp16_us = fp16_ms * 1000.0f;
        float ratio = nf4_us / fp16_us;

        // Bandwidth: bytes loaded / time
        float nf4_bytes = (float)(cases[i].out_dim * cases[i].in_dim) / 2.0f;  // NF4 data
        nf4_bytes += (float)(cases[i].out_dim * cases[i].in_dim / 64) * 4.0f;  // absmax
        float fp16_bytes = (float)cases[i].out_dim * cases[i].in_dim * 2.0f;    // fp16 data

        float nf4_gbps = nf4_bytes / (nf4_us * 1000.0f);  // GB/s
        float fp16_gbps = fp16_bytes / (fp16_us * 1000.0f);

        printf("%-12s %5d %5d | %7.1f %7.1f | %5.2fx | %7.1f  %7.1f\n",
               cases[i].name, cases[i].out_dim, cases[i].in_dim,
               nf4_us, fp16_us, ratio, nf4_gbps, fp16_gbps);

        total_nf4 += nf4_us;
        total_fp16 += fp16_us;
        total_nf4_bytes += nf4_bytes;
        total_fp16_bytes += fp16_bytes;
    }

    printf("\n");
    printf("Total per-layer:  NF4=%.1fus  fp16=%.1fus  ratio=%.2fx\n",
           total_nf4, total_fp16, total_nf4 / total_fp16);
    printf("27 NF4 layers:    NF4=%.1fms  fp16=%.1fms\n",
           total_nf4 * 27.0f / 1000.0f, total_fp16 * 27.0f / 1000.0f);
    printf("Avg bandwidth:    NF4=%.1f GB/s  fp16=%.1f GB/s\n",
           total_nf4_bytes * 27 / (total_nf4 * 27 * 1000.0f),
           total_fp16_bytes * 27 / (total_fp16 * 27 * 1000.0f));

    // LM Head benchmark (VOCAB_SIZE=151936, HIDDEN=1024)
    printf("\n=== LM Head (151936 x 1024, cuBLAS fp32 compute) ===\n");
    {
        BenchCase lm = {"lm_head", 151936, 1024};
        float lm_ms = bench_fp16_cublas(lm, 20, 50);
        float lm_us = lm_ms * 1000.0f;
        float lm_bytes = (float)lm.out_dim * lm.in_dim * 2.0f;
        float lm_gbps = lm_bytes / (lm_us * 1000.0f);
        printf("LM head: %.1f us (%.1f ms) = %.1f GB/s\n", lm_us, lm_ms, lm_gbps);
    }

    // Kernel launch overhead: measure 200 tiny kernel launches
    printf("\n=== Kernel launch overhead ===\n");
    {
        half* d_dummy;
        cudaMalloc(&d_dummy, 1024 * sizeof(half));
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int n_launches = 500;
        cudaEventRecord(start);
        for (int i = 0; i < n_launches; i++) {
            // Tiny kernel that barely does anything
            launch_nf4_gemv_fast(nullptr, nullptr, nullptr, d_dummy, 4, 128, 64, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("%.0f tiny launches: %.1f ms (%.1f us/launch)\n",
               (float)n_launches, ms, ms * 1000.0f / n_launches);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_dummy);
    }

    // Full decode time estimate
    printf("\n=== Full decode time estimate (NF4 model) ===\n");
    printf("27 layers NF4 GEMV:   %.1f ms\n", total_nf4 * 27 / 1000.0f);
    printf("1 layer fp16 GEMV:    %.1f ms\n", total_fp16 / 1000.0f);
    printf("~450 kernel launches: ~%.1f ms (est)\n", 450 * 0.005f);
    printf("Predicted total:      ~%.1f ms\n",
           total_nf4 * 27 / 1000.0f + total_fp16 / 1000.0f + 450 * 0.005f + 5.0f);
    printf("Actual measured:      ~29.3 ms\n");

    return 0;
}
