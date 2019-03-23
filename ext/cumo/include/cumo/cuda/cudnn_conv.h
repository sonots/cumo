#ifndef CUMO_CUDA_CUDNN_CONV_H
#define CUMO_CUDA_CUDNN_CONV_H

#include <ruby.h>
#include <cudnn.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

cudnnConvolutionFwdAlgoPerf_t
cumo_cuda_cudnn_FindConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        size_t ndim,
        cudnnDataType_t cudnn_dtype,
        cudnnTensorDescriptor_t x_desc,
        VALUE x,
        cudnnFilterDescriptor_t w_desc,
        VALUE w,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t y_sec,
        VALUE y,
        size_t max_workspace_size,
        VALUE pad,
        VALUE stride);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUDNN_CONV_H */
