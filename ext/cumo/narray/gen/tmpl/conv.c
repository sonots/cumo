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

#define CHECK_DIM_EQ(nd1,nd2)                        \
    if ((nd1) != (nd2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "dimention mismatch: %d != %d",     \
                 (int)(nd1), (int)(nd2));            \
    }

#define CheckCudnnError cumo_cuda_cudnn_check_status

static VALUE
cumo_option_value(VALUE value, VALUE default_value)
{
    switch(TYPE(value)) {
    case T_NIL:
    case T_UNDEF:
        return default_value;
    }
    return value;
}

static size_t
GetConvOutDim(size_t in_dim, size_t kernel_size, size_t stride, size_t pad) {
    // assert(stride > 0);
    int64_t numerator;
    // if (cover_all) {
    //     numerator = in_dim + pad * 2 - kernel_size + stride - 1;
    // } else {
    numerator = in_dim + pad * 2 - kernel_size;
    // }
    // if (numerator < 0) {
    //     throw DimensionError{"Output size should be positive."};
    // }
    return (size_t)(numerator / stride + 1);
}

static cudnnTensorDescriptor_t
createCudnnTensorDescriptor(VALUE a) {
    cudnnTensorDescriptor_t desc;
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;

    cumo_narray_t *na;
    int ndim;
    size_t *shape;

    CumoGetNArray(a, na);
    ndim = (int)(na->ndim);
    shape = na->shape;

    assert(cumo_na_check_contiguous(a) == Qtrue);
    CheckCudnnError(cudnnCreateTensorDescriptor(&desc));

    if (ndim == 4) {
        int nchw[4];
        nchw[0] = shape[0];
        nchw[1] = shape[1];
        nchw[2] = shape[2];
        nchw[3] = shape[3];
        // TODO: dtor desc
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnn_dtype, nchw[0], nchw[1], nchw[2], nchw[3]));
    }
    else {
        int int_strides[CUMO_NA_MAX_DIMENSION]; // strides divided by item size
        int int_shape[CUMO_NA_MAX_DIMENSION];
        int idim = 0;
        int stride = 1;
        for (idim = 0; idim < ndim; ++idim) {
            int_shape[idim] = (int)(shape[idim]);
        }
        for (idim = ndim - 1; idim >= 0; --idim) {
            int_strides[idim] = stride;
            stride *= int_shape[idim];
        }
        // TODO: dtor desc
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc, cudnn_dtype, ndim, &int_shape[0], &int_strides[0]));
    }

    return desc;
}

static cudnnFilterDescriptor_t
createCudnnFilterDescriptor(VALUE a) {
    cudnnFilterDescriptor_t desc;
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;

    cumo_narray_t *na;
    int ndim;
    size_t *shape;

    CumoGetNArray(a, na);
    ndim = (int)(na->ndim);
    shape = na->shape;

    assert(cumo_na_check_contiguous(a) == Qtrue);
    CheckCudnnError(cudnnCreateFilterDescriptor(&desc));

    if (ndim == 4) {
        int nchw[4];
        nchw[0] = shape[0];
        nchw[1] = shape[1];
        nchw[2] = shape[2];
        nchw[3] = shape[3];
        // TODO: dtor desc
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc, cudnn_dtype, CUDNN_TENSOR_NCHW, nchw[0], nchw[1], nchw[2], nchw[3]));
    } else {
        int int_shape[CUMO_NA_MAX_DIMENSION];
        int idim = 0;
        for (idim = 0; idim < ndim; ++idim) {
            int_shape[idim] = (int)(shape[idim]);
        }
        // TODO: dtor desc
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc, cudnn_dtype, CUDNN_TENSOR_NCHW, ndim, &int_shape[0]));
    }

    return desc;
}

static cudnnConvolutionDescriptor_t
createCudnnConvolutionDescriptor(size_t ndim, VALUE stride, VALUE pad) {
    cudnnConvolutionDescriptor_t desc;
    cudnnDataType_t compute_type = <%= cudnn_dtype %>;

    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];
    int int_dilation[CUMO_NA_MAX_DIMENSION];

    CHECK_DIM_EQ((size_t)(RARRAY_LEN(stride)), ndim);
    CHECK_DIM_EQ((size_t)(RARRAY_LEN(pad)), ndim);

    for (size_t idim = 0; idim < ndim; ++idim) {
        int_stride[idim]= NUM2INT(rb_ary_entry(stride, (long)idim));
        int_pad[idim]= NUM2INT(rb_ary_entry(pad, (long)idim));
        int_dilation[idim] = 1;
    }

    if (ndim == 2) {
        // TODO: dtor desc
        CheckCudnnError(cudnnSetConvolution2dDescriptor(
                desc,
                int_pad[0],
                int_pad[1],
                int_stride[0],
                int_stride[1],
                int_dilation[0],
                int_dilation[1],
                CUDNN_CROSS_CORRELATION,
                compute_type));
    } else {
        // TODO: dtor desc
        CheckCudnnError(cudnnSetConvolutionNdDescriptor(
                desc, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], CUDNN_CROSS_CORRELATION, compute_type));
    }

    return desc;
}

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    // cover_all is not supported with CuDNN
    VALUE x=self, w, b, y, stride, pad;
    cumo_narray_t *nx, *nw, *nb;

    VALUE kw_hash = Qnil;
    ID kw_table[5] = {rb_intern("b"), rb_intern("stride"), rb_intern("pad")};
    VALUE opts[5] = {Qundef, Qundef, Qundef, Qundef, Qundef};
    size_t ndim;
    size_t *x_shape, *w_shape;
    size_t out_channels, batch_size;

    VALUE x_cont, w_cont;
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    rb_scan_args(argc, argv, "1:", &w, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 4, opts);
    b = cumo_option_value(opts[0], Qnil);
    stride = cumo_option_value(opts[1], Qnil);
    pad = cumo_option_value(opts[2], Qnil);
    y = cumo_option_value(opts[4], Qnil);

    CumoGetNArray(x, nx);
    CumoGetNArray(w, nw);
    Check_Type(stride, T_ARRAY);
    Check_Type(pad, T_ARRAY);

    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CuDNN convolution requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    CHECK_DIM_EQ(nx->ndim, nw->ndim);
    CHECK_DIM_EQ((size_t)(RARRAY_LEN(stride)), ndim);
    CHECK_DIM_EQ((size_t)(RARRAY_LEN(pad)), ndim);

    x_shape = nx->shape;
    w_shape = nw->shape;

    // w.shape = (out_channels, _, k_1, k_2, ..., k_N)
    out_channels = x_shape[0];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    batch_size = w_shape[0];

    if (y == Qnil) { // y is not given.
        size_t *y_shape = ALLOCA_N(size_t, ndim + 2);
        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        y_shape[0] = batch_size;
        y_shape[1] = out_channels;
        for (size_t i = 0; i < ndim; ++i) {
            size_t stride_i = NUM2SIZET(rb_ary_entry(stride, i));
            size_t pad_i = NUM2SIZET(rb_ary_entry(pad, i));
            y_shape[i + 2] = GetConvOutDim(x_shape[i + 2], w_shape[i + 2], stride_i, pad_i);
        }
        y = cumo_na_new(cT, ndim + 2, y_shape);
    }

    x_cont = cumo_na_check_contiguous(x) == Qtrue ? x : rb_funcall(x, rb_intern("dup"), 0);
    w_cont = cumo_na_check_contiguous(w) == Qtrue ? w : rb_funcall(w, rb_intern("dup"), 0);

    x_desc = createCudnnTensorDescriptor(x_cont);
    y_desc = createCudnnTensorDescriptor(y);
    w_desc = createCudnnFilterDescriptor(w_cont);
    conv_desc = createCudnnConvolutionDescriptor(ndim, pad, stride);

    // TODO: get max workspace size
    // TODO: autotune
    //
    // Get cudnn handle
    // call conv
    // add bias

    return y;
}

#undef CHECK_DIM_EQ
#undef CheckCudnnError
