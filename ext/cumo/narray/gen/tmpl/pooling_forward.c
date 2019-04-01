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

// cover_all=true is not supported with CuDNN
// x.pooling_forward(mode, kernel_size, stride: 1, pad: 0, y: nil)
//CUDNN_POOLING_MAX
//CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
//CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
//CUDNN_POOLING_MAX_DETERMINISTIC
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype alpha = 1;
    dtype beta = 0;

    VALUE x=self, mode, kernel_size, stride, pad, y;
    VALUE kw_hash = Qnil;
    ID kw_table[4] = {rb_intern("stride"), rb_intern("pad"), rb_intern("y")};
    VALUE opts[4] = {Qundef, Qundef, Qundef};

    size_t ndim;
    cumo_narray_t *nx;
    size_t *x_shape;

    VALUE x_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t y_desc = 0;
    cudnnPoolingDescriptor_t pool_desc = 0;
    char *x_cont_ptr, *y_ptr;

    cudnnPoolingMode_t int_mode;
    int int_kernel_size[CUMO_NA_MAX_DIMENSION];
    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];

    rb_scan_args(argc, argv, "2:", &mode, &kernel_size, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);
    stride = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    pad = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    y = cumo_cuda_cudnn_option_value(opts[2], Qnil);

    CumoGetNArray(x, nx);

    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CuDNN pooling requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    // required parameter
    int_mode = (cudnnPoolingMode_t)NUM2INT(mode);
    cumo_cuda_cudnn_get_int_ary(int_kernel_size, kernel_size, ndim, 0);
    // default to kernel_size
    if (stride == Qnil) {
        memcpy(int_stride, int_kernel_size, sizeof(int) * ndim);
    } else {
        cumo_cuda_cudnn_get_int_ary(int_stride, stride, ndim, 0);
    }
    // default to 0
    cumo_cuda_cudnn_get_int_ary(int_pad, pad, ndim, 0);

    x_shape = nx->shape;

    if (y != Qnil) {
        CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(y, cT);
    }
    else {
        size_t *y_shape = ALLOCA_N(size_t, ndim + 2);
        // out_shape = (batch_size, num_channels, out_1, out_2, ..., out_N)
        y_shape[0] = x_shape[0];
        y_shape[1] = x_shape[1];
        for (size_t i = 0; i < ndim; ++i) {
            y_shape[i + 2] = cumo_cuda_cudnn_GetConvOutDim(
                    x_shape[i + 2], int_kernel_size[i], int_stride[i], int_pad[i]);
        }
        y = cumo_na_new(cT, ndim + 2, y_shape);
    }

    x_cont = cumo_na_as_contiguous_array(x);

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto POOLING_ERROR;
    status = cumo_cuda_cudnn_CreateTensorDescriptor(&y_desc, y, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto POOLING_ERROR;
    status = cumo_cuda_cudnn_CreatePoolingDescriptor(&pool_desc, int_mode, ndim, int_kernel_size, int_stride, int_pad);
    if (status != CUDNN_STATUS_SUCCESS) goto POOLING_ERROR;

    handle = cumo_cuda_cudnn_handle();
    status = cudnnPoolingForward(
            handle,
            pool_desc,
            (void*)&alpha,
            x_desc,
            (void*)x_cont_ptr,
            (void*)&beta,
            y_desc,
            (void*)y_ptr);
    if (status != CUDNN_STATUS_SUCCESS) goto POOLING_ERROR;

POOLING_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
    if (pool_desc) cudnnDestroyPoolingDescriptor(pool_desc);
    cumo_cuda_cudnn_check_status(status);

    return y;
}

#else // CUDNN_FOUND
VALUE cumo_cuda_eCudnnError;

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCudnnError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
