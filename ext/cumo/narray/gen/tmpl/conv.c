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

#define CHECK_DIM_EQ(nd1,nd2)                        \
    if ((nd1) != (nd2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "dimention mismatch: %d != %d",     \
                 (int)(nd1), (int)(nd2));            \
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

    // cast x into contiguous array
    // cast w into contiguous array

    // cudnn tensor descriptr for x_cont
    // cudnn tensor descriptr for y
    // cudnn filter descriptr for w
    // cudnn conv descriptor for convdtype, pad, stride, null, 1

    // TODO: get max workspace size
    // TODO: autotune
    //
    // Get cudnn handle
    // call conv
    // add bias

    return y;
}

#undef CHECK_DIM_EQ
