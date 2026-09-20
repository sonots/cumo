#ifdef CUDNN_FOUND


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
        CUMO_CHECK_DIM_EQ((size_t)(RARRAY_LEN(out_size)), ndim);
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

// Everything here holds a descriptor or the workspace, and the two allocations
// below raise on their own when the device is full.
static VALUE
<%=c_iter%>_run(VALUE v)
{
    cumo_cuda_cudnn_conv_run_t *r = (cumo_cuda_cudnn_conv_run_t*)v;
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    <%=cudnn_scalar_t%> alpha = 1;
    <%=cudnn_scalar_t%> beta = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_result;
    size_t workspace_size;

    r->status = cumo_cuda_cudnn_CreateTensorDescriptor(&r->held.x_desc, r->x_cont, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateTensorDescriptor(&r->held.y_desc, r->y, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateFilterDescriptor(&r->held.w_desc, r->w_cont, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateConvolutionDescriptor(&r->held.conv_desc, r->ndim, r->int_stride, r->int_pad, <%=cudnn_compute_dtype%>, <%=cudnn_math_type%>);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;

    // auto tune
    r->status = cumo_cuda_cudnn_FindConvolutionBackwardDataAlgorithm(
            &perf_result,
            r->handle,
            r->held.w_desc,
            r->w_cont,
            r->held.x_desc,
            r->x_cont,
            r->held.conv_desc,
            r->held.y_desc,
            r->y,
            cumo_cuda_cudnn_max_workspace_size(),
            r->int_stride,
            r->int_pad,
            r->ndim,
            cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    // The convolution needs the descriptor to name the math type the search
    // settled on, except where the descriptor refused tensor cores.
    r->status = cumo_cuda_cudnn_SetConvolutionMathTypeFromPerf(r->held.conv_desc, perf_result.mathType);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;

    // The search may look at algorithms needing up to max_workspace_size,
    // but only the one it picked has to be paid for.
    workspace_size = perf_result.memory;
    if (workspace_size > 0) r->held.workspace = cumo_cuda_runtime_malloc(workspace_size);
    r->status = cudnnConvolutionBackwardData(
            r->handle,
            (void*)&alpha,
            r->held.w_desc,
            (void*)r->w_cont_ptr,
            r->held.x_desc,
            (void*)r->x_cont_ptr,
            r->held.conv_desc,
            perf_result.algo,
            (void*)r->held.workspace,
            workspace_size,
            (void*)&beta,
            r->held.y_desc,
            (void*)r->y_ptr);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;

    r->status = cumo_cuda_cudnn_AddBias(r, cudnn_dtype, &alpha);
    return Qnil;
}

// cover_all=true is not supported with CUDNN
// dilation > 1 is not supported yet
// x.conv(w, b: nil, stride: 1, pad: 0, out_size: nil, y: nil)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    cumo_cuda_cudnn_conv_run_t r;

    VALUE x=self, w, b, stride, pad, out_size, y;
    VALUE kw_hash = Qnil;
    ID kw_table[5] = {rb_intern("b"), rb_intern("stride"), rb_intern("pad"), rb_intern("out_size"), rb_intern("y")};
    VALUE opts[5] = {Qundef, Qundef, Qundef, Qundef, Qundef};

    size_t ndim;
    cumo_narray_t *nx, *nw;
    size_t *x_shape, *w_shape;
    size_t out_channels, batch_size;

    VALUE x_cont, w_cont;
    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];
    int int_out_size[CUMO_NA_MAX_DIMENSION];

    memset(&r, 0, sizeof(r));
    rb_scan_args(argc, argv, "1:", &w, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 5, opts);
    b = cumo_option_value(opts[0], Qnil);
    stride = cumo_option_value(opts[1], Qnil);
    pad = cumo_option_value(opts[2], Qnil);
    out_size = cumo_option_value(opts[3], Qnil);
    y = cumo_option_value(opts[4], Qnil);

    CumoGetNArray(x, nx);
    CumoGetNArray(w, nw);

    CUMO_CHECK_DIM_EQ(nx->ndim, nw->ndim);
    CUMO_CHECK_NARRAY_TYPE(x, cT, "x");
    CUMO_CHECK_NARRAY_TYPE(w, cT, "w");
    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CUDNN convolution requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    x_shape = nx->shape;
    w_shape = nw->shape;
    batch_size = x_shape[0]; // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    out_channels = w_shape[1]; // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)
    if (x_shape[1] != w_shape[0]) {
        rb_raise(cumo_na_eShapeError, "x_shape[1]:%d does not match with w_shape[0]:%d",
                (int)x_shape[1], (int)w_shape[0]);
    }

    cumo_cuda_cudnn_get_int_ary(int_stride, stride, ndim, 1);
    cumo_cuda_cudnn_get_int_ary(int_pad, pad, ndim, 0);
    get_int_out_size(int_out_size, out_size, ndim, x_shape, w_shape, int_stride, int_pad);

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    {
        size_t *y_shape = ALLOCA_N(size_t, ndim + 2);
        y_shape[0] = batch_size;
        y_shape[1] = out_channels;
        for (size_t i = 0; i < ndim; ++i) {
            y_shape[i + 2] = int_out_size[i];
        }
        if (y == Qnil) {
            y = cumo_na_new(cT, ndim + 2, y_shape);
        }
        else {
            cumo_cuda_cudnn_check_output(y, cT, ndim + 2, y_shape);
        }
    }

    x_cont = cumo_na_as_contiguous_array(x);
    w_cont = cumo_na_as_contiguous_array(w);

    r.x_cont = x_cont;
    r.w_cont = w_cont;
    r.y = y;
    r.x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    r.w_cont_ptr = cumo_na_get_offset_pointer_for_read(w_cont);
    r.y_ptr = cumo_na_get_offset_pointer_for_write(y);
    r.ndim = ndim;
    r.int_stride = int_stride;
    r.int_pad = int_pad;

    // Taking the bias runs Ruby, which can raise, so it is settled before the
    // ensure below has anything to give back.
    if (b != Qnil) {
        cumo_cuda_cudnn_check_input(b, cT, 1, &out_channels);
        r.b_cont = cumo_na_as_contiguous_array(b);
        r.b_cont_ptr = cumo_na_get_offset_pointer_for_read(r.b_cont);
    }
    r.handle = cumo_cuda_cudnn_handle();

    rb_ensure(<%=c_iter%>_run, (VALUE)&r, cumo_cuda_cudnn_release_conv_held, (VALUE)&r.held);
    RB_GC_GUARD(r.b_cont);
    cumo_cuda_cudnn_check_status(r.status);
    cumo_cuda_runtime_check_status(r.held.wait_status);

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
