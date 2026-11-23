<% param_class = cudnn_param_class == "cT" ? class_name : "SFloat" %>
// The stub comes first so that the function yard documents is the one below.
#ifndef CUDNN_FOUND
#include "cumo/cuda/cudnn.h"

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#else // CUDNN_FOUND


/*
  Batch normalization for inference, on cuDNN, with the given mean and var
  in place of those of the batch. It returns
  gamma * (self - mean) / sqrt(var + eps) + beta.
  cuDNN normalizes each channel, the elements that share an index on axis 1,
  over all the other axes. axis has to name those other axes, as [0] for
  [N, C], [0, 2, 3] for [N, C, H, W] and [0, 2, 3, 4] for five dimensions.
  gamma, beta and the statistics then have C elements in any shape, such as
  [C] or [1, C, 1, 1]<% if param_class != class_name %>, and are Cumo::SFloat for Cumo::<%=class_name%><% end %>.
  Another axis of one or two entries, such as the default [0] for four
  dimensions, also normalizes per channel. gamma and the rest then have as
  many elements as the axis leaves, and an axis that leaves fewer than C
  raises ShapeError. Any other axis of three or more entries raises
  RuntimeError. An axis out of order, repeated or out of range raises
  ArgumentError.

  @overload fixed_batch_norm(gamma, beta, mean, var, eps: 2e-5, axis: [0], y: nil)
    @param [Cumo::<%=param_class%>] gamma  The scale.
    @param [Cumo::<%=param_class%>] beta  The shift.
    @param [Cumo::<%=param_class%>] mean  The mean to normalize with, such as the running_mean of batch_norm.
    @param [Cumo::<%=param_class%>] var  The variance to normalize with, such as the running_var of batch_norm.
    @param [Float] eps  Added to the variance.
    @param [Array<Integer>] axis  The axes other than the channel axis.
    @param [Cumo::<%=class_name%>] y  A contiguous array of the shape of self to write the result into and return.
    @return [Cumo::<%=class_name%>]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    <%=cudnn_scalar_t%> coef_one = 1;
    <%=cudnn_scalar_t%> coef_zero = 0;

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
    eps = cumo_option_value(opts[0], Qnil);
    axis = cumo_option_value(opts[1], Qnil);
    y = cumo_option_value(opts[2], Qnil);

    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    if (axis != Qnil) {
        axis_ndim = cumo_cuda_cudnn_get_int_axis(int_axis, axis);
    }

    CumoGetNArray(x, nx);
    x_ndim = nx->ndim;
    cumo_cuda_cudnn_check_axis(axis_ndim, int_axis, x_ndim);
    x_shape = nx->shape;

    {
        cumo_narray_t *ngamma, *nbeta, *nmean, *nvar;
        cumo_cuda_cudnn_shape_t reduced_shape = cumo_cuda_cudnn_ReduceShape(x_ndim, x_shape, axis_ndim, int_axis, 1);
        size_t reduced_total_size = cumo_cuda_cudnn_GetTotalSize(&reduced_shape);

        cumo_cuda_cudnn_check_reduced_size(reduced_total_size, x_ndim, x_shape);

        CumoGetNArray(gamma, ngamma);
        CumoGetNArray(beta, nbeta);
        CumoGetNArray(mean, nmean);
        CumoGetNArray(var, nvar);

        CUMO_CHECK_SIZE_EQ(ngamma->size, reduced_total_size);
        CUMO_CHECK_SIZE_EQ(nbeta->size, reduced_total_size);
        CUMO_CHECK_SIZE_EQ(nmean->size, reduced_total_size);
        CUMO_CHECK_SIZE_EQ(nvar->size, reduced_total_size);
    }

    CUMO_CHECK_NARRAY_TYPE(x, cT, "x");
    cumo_cuda_cudnn_check_param_type(gamma, <%=cudnn_param_class%>, "gamma");
    cumo_cuda_cudnn_check_param_type(beta, <%=cudnn_param_class%>, "beta");
    cumo_cuda_cudnn_check_param_type(mean, <%=cudnn_param_class%>, "mean");
    cumo_cuda_cudnn_check_param_type(var, <%=cudnn_param_class%>, "var");

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

    if (y == Qnil) {
        y = cumo_na_new(cT, x_ndim, x_shape);
    } else {
        cumo_cuda_cudnn_check_output(y, cT, x_ndim, x_shape);
    }
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    handle = cumo_cuda_cudnn_handle();

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;

    status = cumo_cuda_cudnn_CreateBNTensorDescriptor(&bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto FIXED_BATCH_NORM_ERROR;
    // cuDNN derives the parameter descriptor as float for a half x and as x's
    // own type otherwise, and the checks above hold every parameter to exactly
    // that class, so none of them needs a cast.

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

#endif // CUDNN_FOUND
