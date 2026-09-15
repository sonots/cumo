#ifdef CUDNN_FOUND


static void
cumo_cuda_cudnn_get_sizet_ary(size_t *sizet_ary, VALUE ary, size_t ndim)
{
    Check_Type(ary, T_ARRAY);
    CUMO_CHECK_DIM_EQ((size_t)(RARRAY_LEN(ary)), ndim);
    for (size_t idim = 0; idim < ndim; ++idim) {
        sizet_ary[idim] = NUM2SIZET(rb_ary_entry(ary, (long)idim));
    }
}

typedef struct {
    cumo_cuda_cudnn_conv_held_t held;
    cudnnHandle_t handle;
    VALUE   x_cont, gy_cont, gw;
    char   *x_cont_ptr, *gy_cont_ptr, *gw_ptr;
    size_t  ndim;
    int    *int_stride, *int_pad;
    cudnnStatus_t status;
} <%=c_iter%>_run_t;

// Everything here holds a descriptor or the workspace, and the two allocations
// below raise on their own when the device is full.
static VALUE
<%=c_iter%>_run(VALUE v)
{
    <%=c_iter%>_run_t *r = (<%=c_iter%>_run_t*)v;
    cudnnDataType_t cudnn_dtype = <%= cudnn_dtype %>;
    <%=cudnn_scalar_t%> one = 1;
    <%=cudnn_scalar_t%> zero = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perf_result;
    size_t workspace_size;

    r->status = cumo_cuda_cudnn_CreateTensorDescriptor(&r->held.x_desc, r->x_cont, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateTensorDescriptor(&r->held.y_desc, r->gy_cont, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateFilterDescriptor(&r->held.w_desc, r->gw, cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    r->status = cumo_cuda_cudnn_CreateConvolutionDescriptor(&r->held.conv_desc, r->ndim, r->int_stride, r->int_pad, <%=cudnn_compute_dtype%>, <%=cudnn_math_type%>);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;

    // auto tune
    r->status = cumo_cuda_cudnn_FindConvolutionBackwardFilterAlgorithm(
            &perf_result,
            r->handle,
            r->held.x_desc,
            r->x_cont,
            r->held.y_desc,
            r->gy_cont,
            r->held.conv_desc,
            r->held.w_desc,
            r->gw,
            cumo_cuda_cudnn_max_workspace_size(),
            r->int_stride,
            r->int_pad,
            r->ndim,
            cudnn_dtype);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;
    // The descriptor asked for a math type; use the one the search settled on.
    r->status = cudnnSetConvolutionMathType(r->held.conv_desc, perf_result.mathType);
    if (r->status != CUDNN_STATUS_SUCCESS) return Qnil;

    // The search may look at algorithms needing up to max_workspace_size,
    // but only the one it picked has to be paid for.
    workspace_size = perf_result.memory;
    if (workspace_size > 0) r->held.workspace = cumo_cuda_runtime_malloc(workspace_size);
    r->status = cudnnConvolutionBackwardFilter(
            r->handle,
            (void*)&one,
            r->held.x_desc,
            (void*)r->x_cont_ptr,
            r->held.y_desc,
            (void*)r->gy_cont_ptr,
            r->held.conv_desc,
            perf_result.algo,
            (void*)r->held.workspace,
            workspace_size,
            (void*)&zero,
            r->held.w_desc,
            (void*)r->gw_ptr);
    return Qnil;
}

// cover_all=true is not supported with CUDNN
// gw = x.conv_grad_w(gy, w_shape, stride: 1, pad: 0, gw: nil)
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    <%=c_iter%>_run_t r;

    VALUE x=self, gy, w_shape, stride, pad, gw;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("stride"), rb_intern("pad"), rb_intern("gw")};
    VALUE opts[] = {Qundef, Qundef, Qundef};

    size_t ndim;
    cumo_narray_t *nx, *ngy;

    VALUE x_cont, gy_cont;

    size_t sizet_w_shape[CUMO_NA_MAX_DIMENSION];
    int int_stride[CUMO_NA_MAX_DIMENSION];
    int int_pad[CUMO_NA_MAX_DIMENSION];

    memset(&r, 0, sizeof(r));
    rb_scan_args(argc, argv, "2:", &gy, &w_shape, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);
    stride = cumo_option_value(opts[0], Qnil);
    pad = cumo_option_value(opts[1], Qnil);
    gw = cumo_option_value(opts[2], Qnil);

    CumoGetNArray(x, nx);
    CumoGetNArray(gy, ngy);

    CUMO_CHECK_DIM_EQ(nx->ndim, ngy->ndim);
    CUMO_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CHECK_NARRAY_TYPE(gy, cT);
    if (nx->ndim - 2 < 2) {
        rb_raise(cumo_na_eShapeError, "CUDNN convolution requires number of spatial "
                "dimensions to be greater than or equal to 2, but %d", nx->ndim - 2);
    }
    ndim = nx->ndim - 2;  // Number of spatial dimensions

    cumo_cuda_cudnn_get_sizet_ary(sizet_w_shape, w_shape, ndim + 2);
    cumo_cuda_cudnn_get_int_ary(int_stride, stride, ndim, 1);
    cumo_cuda_cudnn_get_int_ary(int_pad, pad, ndim, 0);

    if (gw == Qnil) {
        gw = cumo_na_new(cT, ndim + 2, sizet_w_shape);
    }
    else {
        cumo_cuda_cudnn_check_output(gw, cT, ndim + 2, sizet_w_shape);
    }
    // w_shape = (out_channels, in_channels, k_1, k_2, ..., k_N)
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    // y_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    CUMO_CHECK_DIM_EQ(nx->shape[0], ngy->shape[0]);
    CUMO_CHECK_DIM_EQ(sizet_w_shape[0], ngy->shape[1]);
    CUMO_CHECK_DIM_EQ(sizet_w_shape[1], nx->shape[1]);

    {
        // cuDNN rejects a gy of the wrong spatial size with CUDNN_STATUS_BAD_PARAM,
        // which does not say which dimension is wrong. This check was an assert,
        // so a release build said nothing at all.
        size_t *y_shape = ngy->shape;
        size_t *x_shape = nx->shape;
        for (size_t i = 0; i < ndim; ++i) {
            size_t out_dim = cumo_cuda_cudnn_GetConvOutDim(
                    x_shape[i + 2], sizet_w_shape[i + 2], int_stride[i], int_pad[i]);
            if (y_shape[i + 2] != out_dim) {
                rb_raise(cumo_na_eShapeError,
                        "gy_shape[%d]:%d does not match with the convolution output size %d",
                        (int)(i + 2), (int)y_shape[i + 2], (int)out_dim);
            }
        }
    }

    x_cont = cumo_na_as_contiguous_array(x);
    gy_cont = cumo_na_as_contiguous_array(gy);

    r.x_cont = x_cont;
    r.gy_cont = gy_cont;
    r.gw = gw;
    r.x_cont_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    r.gy_cont_ptr = cumo_na_get_offset_pointer_for_read(gy_cont);
    r.gw_ptr = cumo_na_get_offset_pointer_for_write(gw);
    r.ndim = ndim;
    r.int_stride = int_stride;
    r.int_pad = int_pad;
    r.handle = cumo_cuda_cudnn_handle();

    rb_ensure(<%=c_iter%>_run, (VALUE)&r, cumo_cuda_cudnn_release_conv_held, (VALUE)&r.held);
    cumo_cuda_cudnn_check_status(r.status);
    cumo_cuda_runtime_check_status(r.held.wait_status);

    return gw;
}

#else // CUDNN_FOUND
#include "cumo/cuda/cudnn.h"

static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    rb_raise(cumo_cuda_eCUDNNError, "cuDNN is not available");
}
#endif // CUDNN_FOUND
