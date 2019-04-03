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

static void
cumo_cuda_cudnn_get_sizet_ary(size_t *sizet_ary, VALUE ary, size_t ndim)
{
    Check_Type(ary, T_ARRAY);
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ((size_t)(RARRAY_LEN(ary)), ndim);
    for (size_t idim = 0; idim < ndim; ++idim) {
        sizet_ary[idim] = NUM2SIZET(rb_ary_entry(ary, (long)idim));
    }
}

// cover_all=true is not supported with CUDNN
// gw = x.conv_backward_filter(gy, w_shape, stride: 1, pad: 0, gw: nil)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype one = 1;
    dtype zero = 0;

    VALUE x=self, gy, w_shape, stride, pad, gw;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("stride"), rb_intern("pad"), rb_intern("gw")};
    VALUE opts[] = {Qundef, Qundef, Qundef};

    size_t ndim;
    cumo_narray_t *nx, *ngy;

    VALUE x_cont, gy_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t gy_desc = 0;
    cudnnConvolutionDescriptor_t conv_desc = 0;
    cudnnFilterDescriptor_t gw_desc = 0;
    char *x_cont_ptr, *gy_cont_ptr, *gw_ptr;

    cudnnConvolutionBwdFilterAlgoPerf_t perf_result;
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t max_workspace_size = CUMO_CUDA_CUDNN_DEFAULT_MAX_WORKSPACE_SIZE;
    size_t workspace_size;
    char* workspace = 0;

    size_t sizet_w_shape[CUMO_NA_MAX_DIMENSION];
    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];

    rb_scan_args(argc, argv, "2:", &gy, &w_shape, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);
    stride = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    pad = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    gw = cumo_cuda_cudnn_option_value(opts[2], Qnil);

    CumoGetNArray(x, nx);
    CumoGetNArray(gy, ngy);

    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(nx->ndim, ngy->ndim);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gy, cT);
    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CUDNN convolution requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    cumo_cuda_cudnn_get_sizet_ary(sizet_w_shape, w_shape, ndim + 2);
    cumo_cuda_cudnn_get_int_ary(int_stride, stride, ndim, 1);
    cumo_cuda_cudnn_get_int_ary(int_pad, pad, ndim, 0);

    if (gw != Qnil) {
        CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(gw, cT);
        assert(cumo_na_check_contiguous(gw) == Qtrue);
    }
    else {
        gw = cumo_na_new(cT, ndim + 2, sizet_w_shape);
    }
    // w_shape = (out_channels, in_channels, k_1, k_2, ..., k_N)
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    // y_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(nx->shape[0], ngy->shape[0]);
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(sizet_w_shape[0], ngy->shape[1]);
    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(sizet_w_shape[1], nx->shape[1]);

    {
        // shape check of gy
        size_t *y_shape = ngy->shape;
        size_t *x_shape = nx->shape;
        for (size_t i = 0; i < ndim; ++i) {
            // TODO: raise
            assert(y_shape[i + 2] == cumo_cuda_cudnn_GetConvOutDim(
                    x_shape[i + 2], sizet_w_shape[i + 2], int_stride[i], int_pad[i]));
        }
    }

    x_cont = cumo_na_as_contiguous_array(x);
    gy_cont = cumo_na_as_contiguous_array(gy);

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gy_cont_ptr = cumo_na_get_offset_pointer_for_read(gy_cont);
    gw_ptr = cumo_na_get_offset_pointer_for_write(gw);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;
    status = cumo_cuda_cudnn_CreateTensorDescriptor(&gy_desc, gy_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;
    status = cumo_cuda_cudnn_CreateFilterDescriptor(&gw_desc, gw, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;
    status = cumo_cuda_cudnn_CreateConvolutionDescriptor(&conv_desc, ndim, int_stride, int_pad, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;

    handle = cumo_cuda_cudnn_handle();

    // auto tune
    status = cumo_cuda_cudnn_FindConvolutionBackwardFilterAlgorithm(
            &perf_result,
            handle,
            x_desc,
            x_cont,
            gy_desc,
            gy_cont,
            conv_desc,
            gw_desc,
            gw,
            max_workspace_size,
            int_stride,
            int_pad,
            ndim,
            cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;
    algo = perf_result.algo;
    workspace_size = perf_result.memory;

    workspace = cumo_cuda_runtime_malloc(max_workspace_size);
    status = cudnnConvolutionBackwardFilter(
            handle,
            (void*)&one,
            x_desc,
            (void*)x_cont_ptr,
            gy_desc,
            (void*)gy_cont_ptr,
            conv_desc,
            algo,
            (void*)workspace,
            workspace_size,
            (void*)&zero,
            gw_desc,
            (void*)gw_ptr);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_BACKWARD_FILTER_ERROR;

CONV_BACKWARD_FILTER_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (gy_desc) cudnnDestroyTensorDescriptor(gy_desc);
    if (gw_desc) cudnnDestroyFilterDescriptor(gw_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace) cumo_cuda_runtime_free(workspace);
    cumo_cuda_cudnn_check_status(status);

    return gw;
}

#else // CUDNN_FOUND
VALUE cumo_cuda_eCUDNNError;

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
