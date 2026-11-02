#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// All frequencies are absolute: CPU kHz, GPC/EMC Hz.
// mode: 0 = no sysfs writes, 1 = root-only real hardware, 2 = temporary mock tree.
// Creation acquires ownership/snapshots but does not change clocks.
// GPU-only extension: callback(op=0,set MHz; op=1,reset); returns 0 on success.
// Callback/context must outlive retained CUDA handles; CPU/EMC vector entries must be 0.
typedef int (*fuse_gpu_callback)(void*, unsigned, uint64_t);
int fuse_gpu_backend_abi_version(void);
void* fuse_open_gpu(const char*, int, fuse_gpu_callback, void*);
void* fuse_open(const char* config, int mode);
// Independent reference ABI, not an author-released API version.
int fuse_reference_abi_version(void); // v5 = 5; dual-granularity/timing evidence
const char* fuse_error(void);
int fuse_set(void* handle, uint64_t cpu_khz, uint64_t gpu_hz, uint64_t emc_hz);
int fuse_cpu(void* handle, uint64_t cpu_khz);
int fuse_enqueue(void* handle, uint64_t cpu_khz, uint64_t gpu_hz, uint64_t emc_hz, uint64_t delay_us);
int fuse_cancel(void* handle);
double fuse_last_write_us(void* handle);
// v2 stream adapter primitives. All functions serialize through the same writer.
// begin_submission boosts CPU on the calling submission thread. Ticket ownership
// prevents an older submission from lowering CPU during a newer submission.
uint64_t fuse_submission_begin(void* handle, uint64_t cpu_submit_khz, uint64_t cpu_run_khz);
int fuse_submission_end(void* handle, uint64_t ticket); // 1 = superseded, 0 = applied
// Called from an ordered CUDA boundary host function; makes no CUDA API calls.
int fuse_gpu_boundary(void* handle, uint64_t ticket, uint64_t gpu_hz, uint64_t emc_hz);
// 0 = resource request issued, 1 = obsolete source, 2 = unchanged requested target.
// A no-op is not evidence of a physical write/transition. CPU is never modified.
int fuse_gpu_lookahead(void* handle, uint64_t source_ticket, uint64_t gpu_hz, uint64_t emc_hz);
int fuse_status(void* handle);
int fuse_validate_vector(void* handle, uint64_t cpu_khz, uint64_t gpu_hz, uint64_t emc_hz);
// Adapters/callbacks retain the clock owner so late CUDA errors cannot cause UAF.
void fuse_retain(void* handle);
void fuse_release(void* handle);
// Cancels pending commands and restores all original domains; -1 signals failure.
int fuse_close(void* handle);
#ifdef __cplusplus
}
#endif
