#pragma once
#include "CUDAFUSE.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

// Explicit adapter: ONE enqueue must correspond to ONE declared super-block.
// For a fused end-to-end engine this controls only the whole-engine boundary.
// It cannot infer engine-internal layer boundaries or split a serialized engine.
struct TensorRTFUSEBlock {
    uint64_t id,cpu_submit_khz,cpu_run_khz,gpu_hz,emc_hz;
    uint64_t estimated_us,switch_us,next_gpu_hz,next_emc_hz;
    bool lookahead;
};
inline bool enqueueFUSESuperBlock(void* adapter,nvinfer1::IExecutionContext& context,
                                  cudaStream_t main_stream,const TensorRTFUSEBlock& b){
    auto ticket=fuse_cuda_begin(adapter,main_stream,b.id,b.cpu_submit_khz,b.cpu_run_khz,
                                b.gpu_hz,b.emc_hz,b.estimated_us,b.switch_us,
                                b.next_gpu_hz,b.next_emc_hz,b.lookahead?1:0);
    if(!ticket)throw std::runtime_error(fuse_cuda_error());
    // TensorRT's enqueueV3 joins its auxiliary streams back into main_stream.
    // The end marker is therefore after the enqueue's joined GPU work.
    bool ok=context.enqueueV3(main_stream);
    if(fuse_cuda_end(adapter,ticket)<0)throw std::runtime_error(fuse_cuda_error());
    return ok;
}
