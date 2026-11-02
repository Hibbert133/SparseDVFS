#include "../cpp/TensorRTFUSE.hpp"
// Compile/link this adapter into an application that already has TensorRT
// engines, bindings, buffers and the Qwen cache implementation. No engine or
// weight conversion is implied by this helper.
bool submit_one_superblock(void* adapter,nvinfer1::IExecutionContext& context,
                          cudaStream_t stream,const TensorRTFUSEBlock& block){
    return enqueueFUSESuperBlock(adapter,context,stream,block);
}
