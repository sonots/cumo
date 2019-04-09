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

// y = x.batch_norm(gamma, beta, running_mean:, running_var:, eps:, decay:, axis:, mean:, inv_std:)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype coef_alpha = 1;
    dtype coef_beta = 0;

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

    cumo_narray_t *nx, *ngamma, *nbeta;
    size_t *x_shape, *gamma_shape, *beta_shape, reduced_shape[CUMO_NA_MAX_DIMENSION];
    size_t x_ndim, gamma_ndim, beta_ndim, reduced_ndim;

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
    running_mean = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    running_var = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    mean = cumo_cuda_cudnn_option_value(opts[2], Qnil);
    inv_std = cumo_cuda_cudnn_option_value(opts[3], Qnil);
    eps = cumo_cuda_cudnn_option_value(opts[4], Qnil);
    decay = cumo_cuda_cudnn_option_value(opts[5], Qnil);
    axis = cumo_cuda_cudnn_option_value(opts[6], Qnil);
    y = cumo_cuda_cudnn_option_value(opts[7], Qnil);

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
        Check_Type(axis, T_ARRAY);
        axis_ndim = (size_t)(RARRAY_LEN(axis));
        for (size_t idim = 0; idim < axis_ndim; ++idim) {
            int_axis[idim] = NUM2INT(rb_ary_entry(axis, (long)idim));
        }
        // TODO: check axis is sorted
    }

    CumoGetNArray(x, nx);
    CumoGetNArray(gamma, ngamma);
    CumoGetNArray(beta, nbeta);
    x_ndim = nx->ndim;
    x_shape = nx->shape;
    gamma_ndim = ngamma->ndim;
    gamma_shape = ngamma->shape;
    beta_ndim = nbeta->ndim;
    beta_shape = nbeta->shape;

    // TODO: Add ndim and shape (same with reduced) for running_mean and running_var if given
    reduced_ndim = cumo_cuda_cudnn_ReduceShape(reduced_shape, x_ndim, x_shape, axis_ndim, int_axis, 1);
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_ndim, gamma_ndim);
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_ndim, beta_ndim);
    for (size_t idim = 0; idim < reduced_ndim; ++idim) {
        CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_shape[idim], gamma_shape[idim]);
        CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_shape[idim], beta_shape[idim]);
    }

    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gamma, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(beta, cT);
    if (running_mean != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(running_mean, cT);
    if (running_var != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(running_var, cT);
    if (mean != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(mean, cT);
    if (inv_std != Qnil) CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(inv_std, cT);

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

    // TODO: type and shape check
    if (y == Qnil) y = cumo_na_new(cT, x_ndim, x_shape);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    status = cumo_cuda_cudnn_CreateBNTensorDescriptor(&bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;
    // TODO: bn_desc may return another type, and may need to cast gamma, beta, mean, var

    handle = cumo_cuda_cudnn_handle();

    status = cudnnBatchNormalizationForwardTraining(
            handle,
            mode,
            (void*)&coef_alpha,
            (void*)&coef_beta,
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

#else // CUDNN_FOUND
VALUE cumo_cuda_eCUDNNError;

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
