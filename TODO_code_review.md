# Code Review Findings (from automated agents, 2026-03-27)

## Applied

- [x] Loss normalization: divide by n_valid not n_samples
- [x] AMP consistency between reference and current log-probs
- [x] Remove CPU round-trip on unified memory (Jetson)
- [x] Remove expensive empty_cache() from inner loop
- [x] Fix surr1/surr2 tensor leak in GRPO path
- [x] Remove debug code (dump_half, dump_float, debug_mode)
- [x] Add CUDA_CHECK / CUBLAS_CHECK macros

## Critical (must fix before release)

- [ ] Apply CUDA_CHECK to all cudaMalloc in allocate_buffers() (~20 calls)
- [ ] Apply CUDA_CHECK to all cudaMalloc in alloc_batch() (~18 calls)
- [ ] Apply CUBLAS_CHECK to all cublasGemmEx calls (~10 calls)
- [ ] Apply CUDA_CHECK to cudaMemcpy in weights.cpp load() lambda
- [ ] Fix alloc_batch() memory leak (TODO comment at line 1089)
- [ ] Validate ModelConfig values (reject -1 / missing keys, don't silently fall back)

## High priority

- [ ] Add NaN/Inf detection in grpo_step (skip sample, log warning)
- [ ] Add NaN/Inf check on grad_norm after clipping
- [ ] Check file existence before opening in load_index() and load_weights()
- [ ] Add error message for missing jetson_engine module ("run cmake first")
- [ ] Check cudaStreamBeginCapture/EndCapture/GraphInstantiate return codes
- [ ] Validate token_id bounds in embedding lookup kernels
- [ ] Make LoRA sync failures visible (return error count from sync())

## Medium priority

- [ ] Fix top-p sampling: sort by probability, not index order (both single + batch)
- [ ] Add bounds check in dequant_q4l_kernel for non-multiple-of-64 dims
- [ ] Add cudaGetLastError() after kernel launches for debug builds
- [ ] Make embedding sharing failure an error (not warning) on Jetson
- [ ] Rename NF4Weight struct to QuantWeight (used for both NF4 and Q4L)
- [ ] Remove disabled CUDA graph code path (or fix it for Q4L)
- [ ] Remove dead NF4 kernel code in kernels.cu (lines 39-179)
