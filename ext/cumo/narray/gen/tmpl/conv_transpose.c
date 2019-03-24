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

// VALUE is Ruby Array
static void
get_int_out_size(int* int_out_size, VALUE out_size, size_t ndim, size_t* x_shape, size_t* w_shape, int* int_stride, int* int_pad)
{
    if (out_size == Qnil) {
        for (size_t i = 0; i < ndim; ++i) {
            int_out_size[i] = cumo_cuda_cudnn_GetConvTransposeOutDim(
                    x_shape[i + 2], w_shape[i + 2], int_stride[i], int_pad[i]);
        }
    } else {
        Check_Type(out_size, T_ARRAY);
        CUMO_CUDA_CUDNN_CHECK_DIM_EQ((size_t)(RARRAY_LEN(out_size)), ndim);
        for (size_t i = 0; i < ndim; ++i) {
            int_out_size[i] = NUM2INT(rb_ary_entry(out_size, (long)i));
        }
    }
    // only cover_all=false is supported
    for (size_t i = 0; i < ndim; ++i) {
        if (x_shape[i + 2] != cumo_cuda_cudnn_GetConvOutDim(
                    int_out_size[i], w_shape[i + 2], int_stride[i], int_pad[i])) {
            rb_raise(rb_eRuntimeError, "CUDA transposed convolution does not support specified output sizes");
        }
    }
}

// cover_all=true is not supported with CuDNN
// dilation > 1 is not supported yet
// x.conv(w, b: nil, stride: 1, pad: 0, out_size: nil, y: nil)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    cudnnStatus_t status = 0;
    cudnnHandle_t handle = 0;
    dtype alpha = 1;
    dtype beta = 0;

    VALUE x=self, w, b, stride, pad, out_size, y;
    VALUE kw_hash = Qnil;
    ID kw_table[5] = {rb_intern("b"), rb_intern("stride"), rb_intern("pad"), rb_intern("out_size"), rb_intern("y")};
    VALUE opts[5] = {Qundef, Qundef, Qundef, Qundef, Qundef};

    size_t ndim;
    cumo_narray_t *nx, *nw;
    size_t *x_shape, *w_shape;
    size_t out_channels, batch_size;

    VALUE x_cont, w_cont;
    cudnnTensorDescriptor_t x_desc = 0;
    cudnnTensorDescriptor_t y_desc = 0;
    cudnnTensorDescriptor_t b_desc = 0;
    cudnnFilterDescriptor_t w_desc = 0;
    cudnnConvolutionDescriptor_t conv_desc = 0;
    char *x_cont_ptr, *w_cont_ptr, *y_ptr;

    cudnnConvolutionBwdDataAlgoPerf_t perf_result;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t max_workspace_size = CUMO_CUDA_CUDNN_DEFAULT_MAX_WORKSPACE_SIZE;
    size_t workspace_size;
    char* workspace = 0;

    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];
    int int_out_size[CUMO_NA_MAX_DIMENSION];

    rb_scan_args(argc, argv, "1:", &w, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 4, opts);
    b = cumo_cuda_cudnn_option_value(opts[0], Qnil);
    stride = cumo_cuda_cudnn_option_value(opts[1], Qnil);
    pad = cumo_cuda_cudnn_option_value(opts[2], Qnil);
    out_size = cumo_cuda_cudnn_option_value(opts[3], Qnil);
    y = cumo_cuda_cudnn_option_value(opts[4], Qnil);

    CumoGetNArray(x, nx);
    CumoGetNArray(w, nw);

    CUMO_CUDA_CUDNN_CHECK_DIM_EQ(nx->ndim, nw->ndim);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(w, cT);
    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CuDNN convolution requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    x_shape = nx->shape;
    w_shape = nw->shape;
    batch_size = x_shape[0]; // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    out_channels = w_shape[1]; // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)

    cumo_cuda_cudnn_get_int_ary(int_stride, stride, ndim, 1);
    cumo_cuda_cudnn_get_int_ary(int_pad, pad, ndim, 0);
    get_int_out_size(int_out_size, out_size, ndim, x_shape, w_shape, int_stride, int_pad);

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    if (y != Qnil) {
        CUMO_CUDA_CUDNN_CHECK_NARRAY_TYPE(y, cT);
        // TODO: shape check
    }
    else {
        size_t *y_shape = ALLOCA_N(size_t, ndim + 2);
        y_shape[0] = batch_size;
        y_shape[1] = out_channels;
        for (size_t i = 0; i < ndim; ++i) {
            y_shape[i + 2] = int_out_size[i];
        }
        y = cumo_na_new(cT, ndim + 2, y_shape);
    }

    x_cont = cumo_na_as_contiguous_array(x);
    w_cont = cumo_na_as_contiguous_array(w);

    x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    w_cont_ptr = cumo_na_get_offset_pointer_for_read(w_cont);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    status = cumo_cuda_cudnn_CreateTensorDescriptor(&x_desc, x_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;
    status = cumo_cuda_cudnn_CreateTensorDescriptor(&y_desc, y, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;
    status = cumo_cuda_cudnn_CreateFilterDescriptor(&w_desc, w_cont, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;
    status = cumo_cuda_cudnn_CreateConvolutionDescriptor(&conv_desc, ndim, int_stride, int_pad, cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;

    handle = cumo_cuda_cudnn_handle();

    // auto tune
    status = cumo_cuda_cudnn_FindConvolutionBackwardDataAlgorithm(
            &perf_result,
            handle,
            w_desc,
            w_cont,
            x_desc,
            x_cont,
            conv_desc,
            y_desc,
            y,
            max_workspace_size,
            int_stride,
            int_pad,
            ndim,
            cudnn_dtype);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;
    algo = perf_result.algo;
    workspace_size = perf_result.memory;

    workspace = cumo_cuda_runtime_malloc(max_workspace_size);
    status = cudnnConvolutionBackwardData(
            handle,
            (void*)&alpha,
            w_desc,
            (void*)w_cont_ptr,
            x_desc,
            (void*)x_cont_ptr,
            conv_desc,
            algo,
            (void*)workspace,
            workspace_size,
            (void*)&beta,
            y_desc,
            (void*)y_ptr);
    if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;

    if (b != Qnil) {
        if (status != CUDNN_STATUS_SUCCESS) goto CONV_ERROR;
    }

CONV_ERROR:
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
    if (b_desc) cudnnDestroyTensorDescriptor(b_desc);
    if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace) cumo_cuda_runtime_free(workspace);
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
