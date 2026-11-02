#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Explicit stream integration, not an LD_PRELOAD interceptor. One CUDA device
// and one ordered main stream per adapter. Caller owns stream/buffers.
void* fuse_cuda_open(void* core, int device, const char* trace_csv, unsigned poll_us);
const char* fuse_cuda_error(void);
int fuse_cuda_runtime_version(void);
int fuse_cuda_reference_abi_version(void); // v5 = 5; guards stale mixed libraries
// stream is a cudaStream_t passed opaquely through the C ABI.
// Boundary host function commits GPC/EMC before subsequent kernels can execute.
// CPU boost happens synchronously before the caller submits this block.
uint64_t fuse_cuda_begin(void* adapter, void* stream, uint64_t block_id,
                        uint64_t cpu_submit_khz, uint64_t cpu_run_khz,
                        uint64_t gpu_hz, uint64_t emc_hz,
                        uint64_t estimated_us, uint64_t switch_us,
                        uint64_t next_gpu_hz, uint64_t next_emc_hz, int lookahead);
int fuse_cuda_end(void* adapter, uint64_t ticket);
int fuse_cuda_drain(void* adapter);
// Must precede fuse_close(core). Drains registered stream, retires events, stops
// worker and releases its core reference, including on deferred CUDA errors.
int fuse_cuda_close(void* adapter);
#ifdef __cplusplus
}
#endif
