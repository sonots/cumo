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
  Batch normalization for training, on cuDNN, with the mean and the variance
  of this batch. It returns gamma * (self - mean) / sqrt(var + eps) + beta.
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

  @overload batch_norm(gamma, beta, running_mean: nil, running_var: nil, mean: nil, inv_std: nil, eps: 2e-5, decay: 0.9, axis: [0], y: nil)
    @param [Cumo::<%=param_class%>] gamma  The scale.
    @param [Cumo::<%=param_class%>] beta  The shift.
    @param [Cumo::<%=param_class%>] running_mean  Updated in place to decay * running_mean + (1 - decay) * mean. Give it with running_var.
    @param [Cumo::<%=param_class%>] running_var  Updated in place in the same way, with the unbiased variance of this batch.
    @param [Cumo::<%=param_class%>] mean  An array to write the mean of this batch into, for batch_norm_backward. Give it with inv_std.
    @param [Cumo::<%=param_class%>] inv_std  An array for cuDNN to save the inverse of the standard deviation into, for batch_norm_backward.
    @param [Float] eps  Added to the variance.
    @param [Float] decay  How much of the running statistics is kept.
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

    VALUE x=self, gamma, beta, running_mean, running_var, eps, decay, axis, mean, inv_std, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {
        rb_intern("running_mean"),
        rb_intern("running_var"),
        rb_intern("mean"),
        rb_intern("inv_std"),
        rb_intern("eps"),
        rb_intern("decay"),
        rb_intern("axis"),
        rb_intern("y")
    };
    VALUE opts[] = {Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef};

    cumo_narray_t *nx;
    size_t *x_shape;
    size_t x_ndim;

    VALUE x_cont, gamma_cont, beta_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t bn_desc = 0;
    char *x_cont_ptr, *gamma_cont_ptr, *beta_cont_ptr, *y_ptr;

    cudnnBatchNormMode_t mode;

    // default values
    char *running_mean_ptr=NULL;
    char *running_var_ptr=NULL;
    char *mean_ptr=NULL;
    char *inv_std_ptr=NULL;
    double double_eps = 2e-5;
    double double_decay = 0.9;
    int int_axis[CUMO_NA_MAX_DIMENSION] = {0};
    size_t axis_ndim = 1;

    rb_scan_args(argc, argv, "2:", &gamma, &beta, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 8, opts);
    running_mean = cumo_option_value(opts[0], Qnil);
    running_var = cumo_option_value(opts[1], Qnil);
    mean = cumo_option_value(opts[2], Qnil);
    inv_std = cumo_option_value(opts[3], Qnil);
    eps = cumo_option_value(opts[4], Qnil);
    decay = cumo_option_value(opts[5], Qnil);
    axis = cumo_option_value(opts[6], Qnil);
    y = cumo_option_value(opts[7], Qnil);

    if (running_mean != Qnil) {
        running_mean_ptr = cumo_na_get_offset_pointer_for_write(running_mean);
    }
    if (running_var != Qnil) {
        running_var_ptr = cumo_na_get_offset_pointer_for_write(running_var);
    }
    if (mean != Qnil) {
        mean_ptr = cumo_na_get_offset_pointer_for_write(mean);
    }
    if (inv_std != Qnil) {
        inv_std_ptr = cumo_na_get_offset_pointer_for_write(inv_std);
    }
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    if (decay != Qnil) {
        double_decay = NUM2DBL(decay);
    }
    if (axis != Qnil) {
        axis_ndim = cumo_cuda_cudnn_get_int_axis(int_axis, axis);
    }

    CumoGetNArray(x, nx);
    x_ndim = nx->ndim;
    cumo_cuda_cudnn_check_axis(axis_ndim, int_axis, x_ndim);
    x_shape = nx->shape;

    {
        cumo_narray_t *ngamma, *nbeta, *nrunning_mean, *nrunning_var, *nmean, *ninv_std;
        cumo_cuda_cudnn_shape_t reduced_shape = cumo_cuda_cudnn_ReduceShape(x_ndim, x_shape, axis_ndim, int_axis, 1);
        size_t reduced_total_size = cumo_cuda_cudnn_GetTotalSize(&reduced_shape);

        cumo_cuda_cudnn_check_reduced_size(reduced_total_size, x_ndim, x_shape);

        CumoGetNArray(gamma, ngamma);
        CUMO_CHECK_SIZE_EQ(ngamma->size, reduced_total_size);
        CumoGetNArray(beta, nbeta);
        CUMO_CHECK_SIZE_EQ(nbeta->size, reduced_total_size);
        if (running_mean != Qnil) {
            CumoGetNArray(running_mean, nrunning_mean);
            CUMO_CHECK_SIZE_EQ(nrunning_mean->size, reduced_total_size);
        }
        if (running_var != Qnil) {
            CumoGetNArray(running_var, nrunning_var);
            CUMO_CHECK_SIZE_EQ(nrunning_var->size, reduced_total_size);
        }
        if (mean != Qnil) {
            CumoGetNArray(mean, nmean);
            CUMO_CHECK_SIZE_EQ(nmean->size, reduced_total_size);
        }
        if (inv_std != Qnil) {
            CumoGetNArray(inv_std, ninv_std);
            CUMO_CHECK_SIZE_EQ(ninv_std->size, reduced_total_size);
        }
    }

    CUMO_CHECK_NARRAY_TYPE(x, cT, "x");
    cumo_cuda_cudnn_check_param_type(gamma, <%=cudnn_param_class%>, "gamma");
    cumo_cuda_cudnn_check_param_type(beta, <%=cudnn_param_class%>, "beta");
    if (running_mean != Qnil) cumo_cuda_cudnn_check_param_type(running_mean, <%=cudnn_param_class%>, "running_mean");
    if (running_var != Qnil) cumo_cuda_cudnn_check_param_type(running_var, <%=cudnn_param_class%>, "running_var");
    if (mean != Qnil) cumo_cuda_cudnn_check_param_type(mean, <%=cudnn_param_class%>, "mean");
    if (inv_std != Qnil) cumo_cuda_cudnn_check_param_type(inv_std, <%=cudnn_param_class%>, "inv_std");

    x_cont = cumo_na_as_contiguous_array(x);
    gamma_cont = cumo_na_as_contiguous_array(gamma);
    beta_cont = cumo_na_as_contiguous_array(beta);
    if (running_mean != Qnil && cumo_na_check_contiguous(running_mean) != Qtrue) {
        rb_raise(rb_eRuntimeError, "running_mean must be contiguous");
    }
    if (running_var != Qnil && cumo_na_check_contiguous(running_var) != Qtrue) {
        rb_raise(rb_eRuntimeError, "running_var must be contiguous");
    }
    if (mean != Qnil && cumo_na_check_contiguous(mean) != Qtrue) {
        rb_raise(rb_eRuntimeError, "mean must be contiguous");
    }
    if (inv_std != Qnil && cumo_na_check_contiguous(inv_std) != Qtrue) {
        rb_raise(rb_eRuntimeError, "inv_std must be contiguous");
    }

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gamma_cont_ptr = cumo_na_get_offset_pointer_for_read(gamma_cont);
    beta_cont_ptr = cumo_na_get_offset_pointer_for_read(beta_cont);

    if (y == Qnil) {
        y = cumo_na_new(cT, x_ndim, x_shape);
    } else {
        cumo_cuda_cudnn_check_output(y, cT, x_ndim, x_shape);
    }
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    handle = cumo_cuda_cudnn_handle();

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;

    status = cumo_cuda_cudnn_CreateBNTensorDescriptor(&bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;
    // cuDNN derives the parameter descriptor as float for a half x and as x's
    // own type otherwise, and the checks above hold every parameter to exactly
    // that class, so none of them needs a cast.

    status = cudnnBatchNormalizationForwardTraining(
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
            1.0 - double_decay,
            running_mean_ptr,
            running_var_ptr,
            double_eps,
            mean_ptr,
            inv_std_ptr);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;

BATCH_NORM_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (bn_desc) cudnnDestroyTensorDescriptor(bn_desc);
    cumo_cuda_cudnn_check_status(status);

    return y;
}

#endif // CUDNN_FOUND
