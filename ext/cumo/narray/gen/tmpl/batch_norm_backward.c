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

// gx, ggamma, gbeta = x.batch_normalizatoin_backward(gamma, gy, mean:, inv_std:, eps:, axis:)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype coef_alpha = 1;
    dtype coef_beta = 0;

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

    cumo_narray_t *nx, *ngamma; // , *ngy;
    size_t *x_shape, *gamma_shape; // , *gy_shape, reduced_shape[CUMO_NA_MAX_DIMENSION];
    size_t x_ndim, gamma_ndim; // , gy_ndim, reduced_ndim;

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
    rb_get_kwargs(kw_hash, kw_table, 0, 8, opts);
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
        Check_Type(axis, T_ARRAY);
        axis_ndim = (size_t)(RARRAY_LEN(axis));
        for (size_t idim = 0; idim < axis_ndim; ++idim) {
            int_axis[idim] = NUM2INT(rb_ary_entry(axis, (long)idim));
        }
        // TODO: check axis is sorted
    }

    CumoGetNArray(x, nx);
    CumoGetNArray(gamma, ngamma);
    // CumoGetNArray(gy, ngy);
    x_ndim = nx->ndim;
    x_shape = nx->shape;
    gamma_ndim = ngamma->ndim;
    gamma_shape = ngamma->shape;
    // gy_ndim = ngy->ndim;
    // gy_shape = ngy->shape;

    // TODO: Size check of gammma, beta, running_mean, running_var, mean, inv_std
    // are equivalent with either of reduced_shape(keepdims: false) or reduced_shape(keepdims: true)
    // reduced_ndim = cumo_cuda_cudnn_ReduceShape(reduced_shape, x_ndim, x_shape, axis_ndim, int_axis, 1);
    // CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_ndim, gamma_ndim);
    // for (size_t idim = 0; idim < reduced_ndim; ++idim) {
    //     CUMO_CUDA_CUDNN_CHECK_DIM_EQ(reduced_shape[idim], gamma_shape[idim]);
    // }
    // CUMO_CUDA_CUDNN_CHECK_DIM_EQ(x_ndim, gy_ndim);
    // for (size_t idim = 0; idim < x_ndim; ++idim) {
    //     CUMO_CUDA_CUDNN_CHECK_DIM_EQ(x_shape[idim], gy_shape[idim]);
    // }

    // TODO: Add ndim and shape (same with reduced) for mean and inv_std if given

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
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;

    mode = cumo_cuda_cudnn_GetBatchNormMode(axis_ndim, int_axis);
    status = cumo_cuda_cudnn_CreateBNTensorDescriptor(&bn_desc, x_desc, mode);
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;
    // TODO: bn_desc may return another type, and may need to cast gamma, gy, mean, var

    handle = cumo_cuda_cudnn_handle();

    status = cudnnBatchNormalizationBackward(
            handle,
            mode,
            (void*)&coef_alpha,
            (void*)&coef_beta,
            (void*)&coef_alpha,
            (void*)&coef_beta,
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
    if (status != CUDNN_STATUS_SUCCESS) goto BATCH_NORM_ERROR;

BATCH_NORM_ERROR:
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
VALUE cumo_cuda_eCudnnError;

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCudnnError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
