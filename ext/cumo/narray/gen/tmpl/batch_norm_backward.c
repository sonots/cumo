#ifdef CUDNN_FOUND

<%
  cudnn_dtype =
    case type_name
    when 'sfloat'
      'CUDNN_DATA_FLOAT'
    when 'dfloat'
      'CUDNN_DATA_DOUBLE'
    else
      # CUDNN_DATA_HALF
      raise 'not supported'
    end
%>

// gx, ggamma, gbeta = x.batch_norm_backward(gamma, gy, mean:, inv_std:, eps:, axis:)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype coef_one = 1;
    dtype coef_zero = 0;

    VALUE x=self, gamma, gy, mean, inv_std, eps, axis, gx, ggamma, gbeta;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {
        rb_intern("mean"),
        rb_intern("inv_std"),
        rb_intern("eps"),
        rb_intern("axis"),
        rb_intern("gx"),
        rb_intern("ggamma"),
        rb_intern("gbeta")
    };
    VALUE opts[] = {Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef};

    cumo_narray_t *nx, *ngamma;
    size_t *x_shape, *gamma_shape;
    size_t x_ndim, gamma_ndim;

    VALUE x_cont, gamma_cont, gy_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t bn_desc = 0;
    char *x_cont_ptr, *gamma_cont_ptr, *gy_cont_ptr, *gx_ptr, *ggamma_ptr, *gbeta_ptr;

    cudnnBatchNormMode_t mode;

    // default values
    char *mean_ptr=NULL;
    char *inv_std_ptr=NULL;
    double double_eps = 2e-5;
    int int_axis[CUMO_NA_MAX_DIMENSION] = {0};
    size_t axis_ndim = 1;

    rb_scan_args(argc, argv, "2:", &gamma, &gy, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 7, opts);
    mean = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    inv_std = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    eps = cumo_cuda_cudnn_option_value(opts[2], Qnil);
    axis = cumo_cuda_cudnn_option_value(opts[3], Qnil);
    gx = cumo_cuda_cudnn_option_value(opts[4], Qnil);
    ggamma = cumo_cuda_cudnn_option_value(opts[5], Qnil);
    gbeta = cumo_cuda_cudnn_option_value(opts[6], Qnil);

    if (mean != Qnil) {
        mean_ptr = cumo_na_get_offset_pointer_for_read(mean);
    }
    if (inv_std != Qnil) {
        inv_std_ptr = cumo_na_get_offset_pointer_for_read(inv_std);
    }
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    if (axis != Qnil) {
        axis_ndim = cumo_cuda_cudnn_get_int_axis(int_axis, axis);
    }

    CumoGetNArray(x, nx);
    CumoGetNArray(gamma, ngamma);
    x_ndim = nx->ndim;
    x_shape = nx->shape;
    gamma_ndim = ngamma->ndim;
    gamma_shape = ngamma->shape;

    {
        cumo_narray_t *ngy, *nmean, *ninv_std;
        cumo_cuda_cudnn_shape_t reduced_shape = cumo_cuda_cudnn_ReduceShape(x_ndim, x_shape, axis_ndim, int_axis, 1);
        size_t reduced_total_size = cumo_cuda_cudnn_GetTotalSize(&reduced_shape);

        CumoGetNArray(gy, ngy);
        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(nx->size, ngy->size);

        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(ngamma->size, reduced_total_size);
        if (mean != Qnil) {
            CumoGetNArray(mean, nmean);
            CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(nmean->size, reduced_total_size);
        }
        if (inv_std != Qnil) {
            CumoGetNArray(inv_std, ninv_std);
            CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(ninv_std->size, reduced_total_size);
        }
    }

    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gamma, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gy, cT);
    if (mean != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(mean, cT);
    if (inv_std != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(inv_std, cT);

    x_cont = cumo_na_as_contiguous_array(x);
    gamma_cont = cumo_na_as_contiguous_array(gamma);
    gy_cont = cumo_na_as_contiguous_array(gy);
    if (mean != Qnil && cumo_na_check_contiguous(mean) != Qtrue) {
        rb_raise(rb_eRuntimeError, "mean must be contiguous");
    }
    if (inv_std != Qnil && cumo_na_check_contiguous(inv_std) != Qtrue) {
        rb_raise(rb_eRuntimeError, "inv_std must be contiguous");
    }

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gamma_cont_ptr = cumo_na_get_offset_pointer_for_read(gamma_cont);
    gy_cont_ptr = cumo_na_get_offset_pointer_for_read(gy_cont);

    // TODO: type and shape check
    if (gx == Qnil) gx = cumo_na_new(cT, x_ndim, x_shape);
    gx_ptr = cumo_na_get_offset_pointer_for_write(gx);
    if (ggamma == Qnil) ggamma = cumo_na_new(cT, gamma_ndim, gamma_shape);
    ggamma_ptr = cumo_na_get_offset_pointer_for_write(ggamma);
    if (gbeta == Qnil) gbeta = cumo_na_new(cT, gamma_ndim, gamma_shape);
    gbeta_ptr = cumo_na_get_offset_pointer_for_write(gbeta);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_BACKWARD_ERROR;

    status = cudnnCreateTensorDescriptor(&bn_desc);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_BACKWARD_ERROR;

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    status = cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_BACKWARD_ERROR;
    // TODO: bn_desc may return another type, and may need to cast gamma, gy, mean, var

    handle = cumo_cuda_cudnn_handle();

    status = cudnnBatchNormalizationBackward(
            handle,
            mode,
            (void*)&coef_one,
            (void*)&coef_zero,
            (void*)&coef_one,
            (void*)&coef_zero,
            x_desc,
            x_cont_ptr,
            x_desc,
            gy_cont_ptr,
            x_desc,
            gx_ptr,
            bn_desc,
            gamma_cont_ptr,
            ggamma_ptr,
            gbeta_ptr,
            double_eps,
            mean_ptr,
            inv_std_ptr);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_BACKWARD_ERROR;

BATCH_NORM_BACKWARD_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (bn_desc) cudnnDestroyTensorDescriptor(bn_desc);
    cumo_cuda_cudnn_check_status(status);

    {
        VALUE ret = rb_ary_new2(3);
        rb_ary_push(ret, gx);
        rb_ary_push(ret, ggamma);
        rb_ary_push(ret, gbeta);
        return ret;
    }
}

#else // CUDNN_FOUND
#include "cumo/cuda/cudnn.h"

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
