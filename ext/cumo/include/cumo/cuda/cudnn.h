#ifndef CUMO_CUDA_CUDNN_H
#define CUMO_CUDA_CUDNN_H

#include <ruby.h>
#ifdef CUDNN_FOUND
#include <cudnn.h>
#endif // CUDNN_FOUND

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

inline VALUE
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
inline void
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
        size_t ndim, int* int_stride, int* int_pad,
        cudnnDataType_t cudnn_dtype);

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

#endif // CUDNN_FOUND

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_CUDNN_H */
