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

// y = x.fixed_batch_norm(gamma, beta, mean, var, eps:, axis:)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype coef_one = 1;
    dtype coef_zero = 0;

    VALUE x=self, gamma, beta, mean, var, eps, axis, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {
        rb_intern("eps"),
        rb_intern("axis"),
        rb_intern("y")
    };
    VALUE opts[] = {Qundef, Qundef, Qundef};

    cumo_narray_t *nx;
    size_t *x_shape;
    size_t x_ndim;

    VALUE x_cont, gamma_cont, beta_cont, mean_cont, var_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t bn_desc = 0;
    char *x_cont_ptr, *gamma_cont_ptr, *beta_cont_ptr, *mean_cont_ptr, *var_cont_ptr, *y_ptr;

    cudnnBatchNormMode_t mode;

    // default values
    double double_eps = 2e-5;
    int int_axis[CUMO_NA_MAX_DIMENSION] = {0};
    size_t axis_ndim = 1;

    rb_scan_args(argc, argv, "4:", &gamma, &beta, &mean, &var, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);
    eps = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    axis = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    y = cumo_cuda_cudnn_option_value(opts[2], Qnil);

    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    if (axis != Qnil) {
        axis_ndim = cumo_cuda_cudnn_get_int_axis(int_axis, axis);
    }

    CumoGetNArray(x, nx);
    x_ndim = nx->ndim;
    x_shape = nx->shape;

    {
        cumo_narray_t *ngamma, *nbeta, *nmean, *nvar;
        cumo_cuda_cudnn_shape_t reduced_shape = cumo_cuda_cudnn_ReduceShape(x_ndim, x_shape, axis_ndim, int_axis, 1);
        size_t reduced_total_size = cumo_cuda_cudnn_GetTotalSize(&reduced_shape);

        CumoGetNArray(gamma, ngamma);
        CumoGetNArray(beta, nbeta);
        CumoGetNArray(mean, nmean);
        CumoGetNArray(var, nvar);

        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(ngamma->size, reduced_total_size);
        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(nbeta->size, reduced_total_size);
        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(nmean->size, reduced_total_size);
        CUMO_CUDA_CUDNN_CHECK_SIZE_EQ(nvar->size, reduced_total_size);
    }

    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gamma, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(beta, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(mean, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(var, cT);

    x_cont = cumo_na_as_contiguous_array(x);
    gamma_cont = cumo_na_as_contiguous_array(gamma);
    beta_cont = cumo_na_as_contiguous_array(beta);
    mean_cont = cumo_na_as_contiguous_array(mean);
    var_cont = cumo_na_as_contiguous_array(var);

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gamma_cont_ptr = cumo_na_get_offset_pointer_for_read(gamma_cont);
    beta_cont_ptr = cumo_na_get_offset_pointer_for_read(beta_cont);
    mean_cont_ptr = cumo_na_get_offset_pointer_for_read(mean_cont);
    var_cont_ptr = cumo_na_get_offset_pointer_for_read(var_cont);

    // TODO: type and shape check
    if (y == Qnil) y = cumo_na_new(cT, x_ndim, x_shape);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;

    status = cudnnCreateTensorDescriptor(&bn_desc);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    status = cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;
    // TODO: bn_desc may return another type, and may need to cast gamma, beta, mean, var

    handle = cumo_cuda_cudnn_handle();

    status = cudnnBatchNormalizationForwardInference(
            handle,
            mode,
            (void*)&coef_one,
            (void*)&coef_zero,
            x_desc,
            x_cont_ptr,
            x_desc,
            y_ptr,
            bn_desc,
            gamma_cont_ptr,
            beta_cont_ptr,
            mean_cont_ptr,
            var_cont_ptr,
            double_eps);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;

FIXED_BATCH_NORM_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (bn_desc) cudnnDestroyTensorDescriptor(bn_desc);
    cumo_cuda_cudnn_check_status(status);

    return y;
}

#else // CUDNN_FOUND
#include "cumo/cuda/cudnn.h"

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
