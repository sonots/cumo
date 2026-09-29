#include "cumo/cuda/cusolver.h"

#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/intern.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eCusolverError;
VALUE cumo_cuda_mCusolver;
#define eCusolverError cumo_cuda_eCusolverError
#define mCusolver cumo_cuda_mCusolver

#ifdef CUSOLVER_FOUND

static const char*
get_cusolver_error_msg(cusolverStatus_t error)
{
    switch (error) {
#define RETURN_MSG(msg) \
    case msg:           \
        return #msg

        RETURN_MSG(CUSOLVER_STATUS_SUCCESS);
        RETURN_MSG(CUSOLVER_STATUS_NOT_INITIALIZED);
        RETURN_MSG(CUSOLVER_STATUS_ALLOC_FAILED);
        RETURN_MSG(CUSOLVER_STATUS_INVALID_VALUE);
        RETURN_MSG(CUSOLVER_STATUS_ARCH_MISMATCH);
        RETURN_MSG(CUSOLVER_STATUS_MAPPING_ERROR);
        RETURN_MSG(CUSOLVER_STATUS_EXECUTION_FAILED);
        RETURN_MSG(CUSOLVER_STATUS_INTERNAL_ERROR);
        RETURN_MSG(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        RETURN_MSG(CUSOLVER_STATUS_NOT_SUPPORTED);
        RETURN_MSG(CUSOLVER_STATUS_ZERO_PIVOT);
        RETURN_MSG(CUSOLVER_STATUS_INVALID_LICENSE);
        RETURN_MSG(CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED);
        RETURN_MSG(CUSOLVER_STATUS_IRS_PARAMS_INVALID);
        RETURN_MSG(CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC);
        RETURN_MSG(CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE);
        RETURN_MSG(CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER);
        RETURN_MSG(CUSOLVER_STATUS_IRS_INTERNAL_ERROR);
        RETURN_MSG(CUSOLVER_STATUS_IRS_NOT_SUPPORTED);
        RETURN_MSG(CUSOLVER_STATUS_IRS_OUT_OF_RANGE);
        RETURN_MSG(CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES);
        RETURN_MSG(CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED);
        RETURN_MSG(CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED);
        RETURN_MSG(CUSOLVER_STATUS_IRS_MATRIX_SINGULAR);
        RETURN_MSG(CUSOLVER_STATUS_INVALID_WORKSPACE);

#undef RETURN_MSG
    default:
        return "CUSOLVER_STATUS_UNKNOWN";
    }
}

NORETURN(static void raise_cusolver_error(cusolverStatus_t status));

static void
raise_cusolver_error(cusolverStatus_t status)
{
    rb_raise(cumo_cuda_eCusolverError, "%s (error=%d)", get_cusolver_error_msg(status), status);
}

void
cumo_cuda_cusolver_check_status(cusolverStatus_t status)
{
    cumo_cuda_runtime_note_device_write();
    if (status != CUSOLVER_STATUS_SUCCESS) {
        raise_cusolver_error(status);
    }
}

static void
check_call(cusolverStatus_t status)
{
    if (status != CUSOLVER_STATUS_SUCCESS) {
        raise_cusolver_error(status);
    }
}

/*
  Returns the version of the cuSOLVER library cumo is linked with.

  @return [Integer] major * 1000 + minor * 100 + patch
 */
static VALUE
rb_cusolver_version(VALUE self)
{
    int version;
    check_call(cusolverGetVersion(&version));
    return INT2NUM(version);
}

typedef struct {
    cusolverDnHandle_t handle;
    cusolverDnParams_t params;
} cusolver_context_t;

static cusolver_context_t
cusolver_context(void)
{
    static __thread cusolver_context_t *contexts = 0;
    int device;
    if (contexts == 0) {
        contexts = ZALLOC_N(cusolver_context_t, cumo_cuda_runtime_get_device_count());
    }
    device = cumo_cuda_runtime_get_device();
    if (contexts[device].handle == 0) {
        check_call(cusolverDnCreate(&contexts[device].handle));
    }
    if (contexts[device].params == 0) {
        check_call(cusolverDnCreateParams(&contexts[device].params));
    }
    check_call(cusolverDnSetStream(contexts[device].handle, cumo_cuda_stream()));
    return contexts[device];
}

static cudaDataType
cusolver_dtype(VALUE a)
{
    VALUE klass = rb_obj_class(a);
    if (klass == cumo_cSFloat) return CUDA_R_32F;
    if (klass == cumo_cDFloat) return CUDA_R_64F;
    if (klass == cumo_cSComplex) return CUDA_C_32F;
    if (klass == cumo_cDComplex) return CUDA_C_64F;
    rb_raise(rb_eTypeError, "invalid data type for cuSOLVER: %"PRIsVALUE, klass);
}

static cumo_narray_t*
check_contiguous_array(VALUE a, VALUE klass, int ndim, const char *name)
{
    cumo_narray_t *na;
    if (!rb_obj_is_kind_of(a, cumo_cNArray)) {
        rb_raise(rb_eTypeError, "%s must be Cumo::NArray", name);
    }
    if (klass != Qnil && rb_obj_class(a) != klass) {
        rb_raise(rb_eTypeError, "%s must be %"PRIsVALUE, name, klass);
    }
    CumoGetNArray(a, na);
    if (CUMO_NA_NDIM(na) != ndim) {
        rb_raise(cumo_na_eShapeError, "%s must be %d-dimensional", name, ndim);
    }
    if (cumo_na_check_contiguous(a) != Qtrue) {
        rb_raise(rb_eArgError, "%s must be contiguous", name);
    }
    return na;
}

typedef struct {
    cusolver_context_t ctx;
    cudaDataType dtype;
    cublasOperation_t trans;
    int64_t m;
    int64_t n;
    int64_t nrhs;
    void *a;
    int64_t *ipiv;
    void *b;
    char *d_work;
    void *h_work;
    int *d_info;
    int info;
} cusolver_call_t;

static int
read_info(int *d_info)
{
    int info;
    cumo_cuda_runtime_check_status(cumo_cuda_runtime_memcpy_to_host(&info, d_info, sizeof(int)));
    return info;
}

static VALUE
getrf_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    size_t d_size = 0;
    size_t h_size = 0;

    check_call(cusolverDnXgetrf_bufferSize(
            c->ctx.handle, c->ctx.params, c->m, c->n, c->dtype, c->a, c->m, c->dtype, &d_size, &h_size));
    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    if (d_size > 0) c->d_work = cumo_cuda_runtime_malloc(d_size);
    if (h_size > 0) c->h_work = ruby_xmalloc(h_size);
    cumo_cuda_cusolver_check_status(cusolverDnXgetrf(
            c->ctx.handle, c->ctx.params, c->m, c->n, c->dtype, c->a, c->m, c->ipiv,
            c->dtype, c->d_work, d_size, c->h_work, h_size, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
getrs_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    cumo_cuda_cusolver_check_status(cusolverDnXgetrs(
            c->ctx.handle, c->ctx.params, c->trans, c->n, c->nrhs, c->dtype, c->a, c->n, c->ipiv,
            c->dtype, c->b, c->n, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
call_ensure(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cumo_cuda_runtime_return_scratch(c->d_work, 1, NULL);
    cumo_cuda_runtime_return_scratch((char*)c->d_info, 0, NULL);
    ruby_xfree(c->h_work);
    return Qnil;
}

static void
check_pivots(const int64_t *d_ipiv, int64_t n)
{
    int64_t *h_ipiv = ALLOC_N(int64_t, n);
    cudaError_t status = cumo_cuda_runtime_memcpy_to_host(h_ipiv, d_ipiv, sizeof(int64_t) * n);
    int64_t i;
    int in_range = 1;

    for (i = 0; status == cudaSuccess && i < n; ++i) {
        if (h_ipiv[i] < 1 || h_ipiv[i] > n) {
            in_range = 0;
            break;
        }
    }
    ruby_xfree(h_ipiv);
    cumo_cuda_runtime_check_status(status);
    if (!in_range) {
        rb_raise(rb_eArgError, "input array ipiv must be in 1..%"PRId64, n);
    }
}

/*
  Factorizes a matrix in place as P A = L U with cusolverDnXgetrf.

  @param a [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, m]: the column-major m by n matrix A, overwritten
    with L and U
  @return [Array] the pivot indices, 1-based, as a Cumo::Int64 of length
    min(m, n), and the info cuSOLVER reports
 */
static VALUE
rb_cusolver_getrf(VALUE self, VALUE a)
{
    cusolver_call_t c = {0};
    cumo_narray_t *na = check_contiguous_array(a, Qnil, 2, "a");
    size_t k;
    VALUE ipiv;

    c.dtype = cusolver_dtype(a);
    c.m = (int64_t)CUMO_NA_SHAPE(na)[1];
    c.n = (int64_t)CUMO_NA_SHAPE(na)[0];
    k = (size_t)(c.m < c.n ? c.m : c.n);
    ipiv = cumo_na_new(cumo_cInt64, 1, &k);
    if (k == 0) {
        return rb_assoc_new(ipiv, INT2FIX(0));
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.ipiv = (int64_t*)cumo_na_get_offset_pointer_for_write(ipiv);
    c.ctx = cusolver_context();
    rb_ensure(getrf_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return rb_assoc_new(ipiv, INT2NUM(c.info));
}

/*
  Solves op(A) X = B in place with cusolverDnXgetrs, from the factors getrf
  answered.

  @param lu [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, n]: the column-major factors
  @param ipiv [Cumo::Int64] contiguous, of length n, each in 1..n, which is checked
  @param b [Cumo::NArray] of the class of lu, contiguous, of shape [n] or
    [nrhs, n]: the column-major right-hand sides, overwritten with X
  @param trans [String] "N", "T" or "C"
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_getrs(VALUE self, VALUE lu, VALUE ipiv, VALUE b, VALUE trans)
{
    cusolver_call_t c = {0};
    const char *t = StringValueCStr(trans);
    cumo_narray_t *nlu;
    cumo_narray_t *nipiv;
    cumo_narray_t *nb;

    if (strcmp(t, "N") == 0) {
        c.trans = CUBLAS_OP_N;
    } else if (strcmp(t, "T") == 0) {
        c.trans = CUBLAS_OP_T;
    } else if (strcmp(t, "C") == 0) {
        c.trans = CUBLAS_OP_C;
    } else {
        rb_raise(rb_eArgError, "trans must be \"N\", \"T\", or \"C\"");
    }
    nlu = check_contiguous_array(lu, Qnil, 2, "lu");
    nipiv = check_contiguous_array(ipiv, cumo_cInt64, 1, "ipiv");
    c.dtype = cusolver_dtype(lu);
    c.n = (int64_t)CUMO_NA_SHAPE(nlu)[0];
    if ((int64_t)CUMO_NA_SHAPE(nlu)[1] != c.n) {
        rb_raise(cumo_na_eShapeError, "lu must be square");
    }
    if ((int64_t)CUMO_NA_SHAPE(nipiv)[0] != c.n) {
        rb_raise(cumo_na_eShapeError, "ipiv must have %"PRId64" elements", c.n);
    }
    if (!rb_obj_is_kind_of(b, cumo_cNArray)) {
        rb_raise(rb_eTypeError, "b must be Cumo::NArray");
    }
    CumoGetNArray(b, nb);
    nb = check_contiguous_array(b, rb_obj_class(lu), CUMO_NA_NDIM(nb) == 1 ? 1 : 2, "b");
    if ((int64_t)CUMO_NA_SHAPE(nb)[CUMO_NA_NDIM(nb) - 1] != c.n) {
        rb_raise(cumo_na_eShapeError, "b must have %"PRId64" rows", c.n);
    }
    c.nrhs = CUMO_NA_NDIM(nb) == 1 ? 1 : (int64_t)CUMO_NA_SHAPE(nb)[0];
    if (c.n == 0 || c.nrhs == 0) {
        return INT2FIX(0);
    }
    c.ipiv = (int64_t*)cumo_na_get_offset_pointer_for_read(ipiv);
    check_pivots(c.ipiv, c.n);
    c.a = cumo_na_get_offset_pointer_for_read(lu);
    c.b = cumo_na_get_offset_pointer_for_read_write(b);
    c.ctx = cusolver_context();
    rb_ensure(getrs_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

#endif // CUSOLVER_FOUND

/*
  Returns availability of cuSOLVER.

  @return [Boolean] Returns true if cuSOLVER is available
 */
static VALUE
rb_cusolver_available_p(VALUE self)
{
#ifdef CUSOLVER_FOUND
    return Qtrue;
#else
    return Qfalse;
#endif
}

void
Init_cumo_cuda_cusolver(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    /*
      Document-module: Cumo::CUDA::Cusolver
    */
    mCusolver = rb_define_module_under(mCUDA, "Cusolver");
    eCusolverError = rb_define_class_under(mCUDA, "CusolverError", rb_eStandardError);

    rb_define_singleton_method(mCusolver, "available?", rb_cusolver_available_p, 0);
#ifdef CUSOLVER_FOUND
    rb_define_singleton_method(mCusolver, "version", rb_cusolver_version, 0);
    rb_define_singleton_method(mCusolver, "getrf", rb_cusolver_getrf, 1);
    rb_define_singleton_method(mCusolver, "getrs", rb_cusolver_getrs, 4);
    rb_funcall(mCusolver, rb_intern("private_class_method"), 2, ID2SYM(rb_intern("getrf")), ID2SYM(rb_intern("getrs")));
#endif // CUSOLVER_FOUND
}
