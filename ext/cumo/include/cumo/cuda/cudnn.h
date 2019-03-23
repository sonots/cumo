#ifndef CUMO_CUDA_CUDNN_H
#define CUMO_CUDA_CUDNN_H

#include <ruby.h>
#include <cudnn.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void
cumo_cuda_cudnn_check_status(cudnnStatus_t status);

cudnnHandle_t
cumo_cuda_cudnn_handle();

size_t
cumo_cuda_cudnn_GetConvOutDim(size_t in_dim, size_t kernel_size, size_t stride, size_t pad);

cudnnTensorDescriptor_t
cumo_cuda_cudnn_CreateTensorDescriptor(VALUE a, cudnnDataType_t cudnn_dtype);

cudnnFilterDescriptor_t
cumo_cuda_cudnn_CreateFilterDescriptor(VALUE a, cudnnDataType_t cudnn_dtype);

cudnnConvolutionDescriptor_t
cumo_cuda_cudnn_CreateConvolutionDescriptor(size_t ndim, int* int_stride, int* int_pad, cudnnDataType_t cudnn_dtype);

cudnnConvolutionFwdAlgoPerf_t
cumo_cuda_cudnn_FindConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        cudnnTensorDescriptor_t x_desc,
        VALUE x,
        cudnnFilterDescriptor_t w_desc,
        VALUE w,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t y_sec,
        VALUE y,
        size_t max_workspace_size,
        int* int_stride,
        int* int_pad,
        size_t ndim,
        cudnnDataType_t cudnn_dtype);

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUDNN_H */
