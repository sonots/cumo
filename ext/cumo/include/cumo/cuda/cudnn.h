#ifndef CUMO_CUDA_CUDNN_H
#define CUMO_CUDA_CUDNN_H

#include <ruby.h>
#ifdef CUDNN_FOUND
#include <cudnn.h>
#endif // CUDNN_FOUND
#include "cumo/narray.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#ifdef CUDNN_FOUND

VALUE cumo_na_eShapeError;

#define CUMO_CUDA_CUDNN_DEFAULT_MAX_WORKSPACE_SIZE 8 * 1024 * 1024

// TODO: Move to proper generic place
#define CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x,t)                 \
    if (rb_obj_class(x)!=(t)) {                                \
        rb_raise(rb_eTypeError,"invalid NArray type (class)"); \
    }

// TODO: Move to proper generic place
#define CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(sz1,sz2)        \
    if ((sz1) != (sz2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "size mismatch: %d != %d",     \
                 (int)(sz1), (int)(sz2));            \
    }

// TODO: Move to proper generic place
#define CUMO_CUDA_CUDNN_CHECK_DIM_EQ(nd1,nd2)        \
    if ((nd1) != (nd2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "dimention mismatch: %d != %d",     \
                 (int)(nd1), (int)(nd2));            \
    }

void
cumo_cuda_cudnn_check_status(cudnnStatus_t status);

cudnnHandle_t
cumo_cuda_cudnn_handle();

// TODO: Move to more generic proper place
static inline VALUE
cumo_cuda_cudnn_option_value(VALUE value, VALUE default_value)
{
    switch(TYPE(value)) {
    case T_NIL:
    case T_UNDEF:
        return default_value;
    }
    return value;
}

// VALUE is Ruby Array
static inline void
cumo_cuda_cudnn_get_int_ary(int* int_ary, VALUE ary, size_t ndim, int default_value)
{
    if (ary == Qnil) {
        // default to 1
        for (size_t idim = 0; idim < ndim; ++idim) {
            int_ary[idim] = default_value;
        }
    } else if (TYPE(ary) == T_FIXNUM) {
        for (size_t idim = 0; idim < ndim; ++idim) {
            int_ary[idim] = NUM2INT(ary);
        }
    } else {
        Check_Type(ary, T_ARRAY);
        CUMO_CUDA_CUDNN_CHECK_DIM_EQ((size_t)(RARRAY_LEN(ary)), ndim);
        for (size_t idim = 0; idim < ndim; ++idim) {
            int_ary[idim] = NUM2INT(rb_ary_entry(ary, idim));
        }
    }
}

// VALUE is Ruby Array
static inline size_t
cumo_cuda_cudnn_get_int_axis(int* int_axis, VALUE axis)
{
    size_t axis_ndim;
    Check_Type(axis, T_ARRAY);
    axis_ndim = (size_t)(RARRAY_LEN(axis));
    if (CUMO_NA_MAX_DIMENSION <= axis_ndim) {
        rb_raise(rb_eArgError, "Size of axis must be smaller than %d, but was %d",
                (int)CUMO_NA_MAX_DIMENSION, (int)axis_ndim);
    }
    for (size_t idim = 0; idim < axis_ndim; ++idim) {
        int_axis[idim] = NUM2INT(rb_ary_entry(axis, (long)idim));
    }
    // TODO: check axis is sorted
    return axis_ndim;
}

size_t
cumo_cuda_cudnn_GetConvOutDim(
        size_t in_dim,
        size_t kernel_size,
        size_t stride,
        size_t pad);

size_t
cumo_cuda_cudnn_GetConvTransposeOutDim(
        size_t in_dim,
        size_t kernel_size,
        size_t stride,
        size_t pad);

cudnnStatus_t
cumo_cuda_cudnn_CreateTensorDescriptor(
        cudnnTensorDescriptor_t *desc,
        VALUE a,
        cudnnDataType_t cudnn_dtype);

cudnnStatus_t
cumo_cuda_cudnn_CreateFilterDescriptor(
        cudnnFilterDescriptor_t *desc,
        VALUE a,
        cudnnDataType_t cudnn_dtype);

cudnnStatus_t
cumo_cuda_cudnn_CreateConvolutionDescriptor(
        cudnnConvolutionDescriptor_t *desc,
        size_t ndim,
        int* int_stride,
        int* int_pad,
        cudnnDataType_t cudnn_dtype);

cudnnStatus_t
cumo_cuda_cudnn_CreatePoolingDescriptor(
        cudnnPoolingDescriptor_t *desc,
        cudnnPoolingMode_t mode,
        size_t ndim,
        int* int_kernel_size,
        int* int_stride,
        int* int_pad);

cudnnStatus_t
cumo_cuda_cudnn_FindConvolutionForwardAlgorithm(
        cudnnConvolutionFwdAlgoPerf_t *perf_result,
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

cudnnStatus_t
cumo_cuda_cudnn_FindConvolutionBackwardDataAlgorithm(
        cudnnConvolutionBwdDataAlgoPerf_t *perf_result,
        cudnnHandle_t handle,
        cudnnFilterDescriptor_t w_desc,
        VALUE w,
        cudnnTensorDescriptor_t x_desc,
        VALUE x,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t y_desc,
        VALUE y,
        size_t max_workspace_size,
        int* int_stride,
        int* int_pad,
        size_t ndim,
        cudnnDataType_t cudnn_dtype);

cudnnStatus_t
cumo_cuda_cudnn_FindConvolutionBackwardFilterAlgorithm(
        cudnnConvolutionBwdFilterAlgoPerf_t *perf_result,
        cudnnHandle_t handle,
        cudnnTensorDescriptor_t x_desc,
        VALUE x,
        cudnnTensorDescriptor_t dy_desc,
        VALUE dy,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnFilterDescriptor_t dw_desc,
        VALUE dw,
        size_t max_workspace_size,
        int* int_stride,
        int* int_pad,
        size_t ndim,
        cudnnDataType_t cudnn_dtype);

cudnnBatchNormMode_t
cumo_cuda_cudnn_GetBatchNormMode(size_t ndim, int* int_axis);

cudnnStatus_t
cumo_cuda_cudnn_CreateBNTensorDescriptor(
        cudnnTensorDescriptor_t *desc,
        cudnnTensorDescriptor_t x_desc,
        cudnnBatchNormMode_t mode);

size_t
cumo_cuda_cudnn_ReduceShape(
        size_t *reduced_shape,
        size_t shape_ndim,
        size_t *shape,
        size_t axes_ndim,
        int *axes,
        char keepdims);

#endif // CUDNN_FOUND

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUDNN_H */
