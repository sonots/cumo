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
    cublasFillMode_t uplo;
    cusolverEigMode_t jobz;
    int range;
    int64_t il;
    int64_t iu;
    int64_t meig;
    void *w;
    void *u;
    void *vt;
    signed char job;
    void *tau;
    int64_t k;
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
potrf_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    size_t d_size = 0;
    size_t h_size = 0;

    check_call(cusolverDnXpotrf_bufferSize(
            c->ctx.handle, c->ctx.params, c->uplo, c->n, c->dtype, c->a, c->n, c->dtype, &d_size, &h_size));
    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    if (d_size > 0) c->d_work = cumo_cuda_runtime_malloc(d_size);
    if (h_size > 0) c->h_work = ruby_xmalloc(h_size);
    cumo_cuda_cusolver_check_status(cusolverDnXpotrf(
            c->ctx.handle, c->ctx.params, c->uplo, c->n, c->dtype, c->a, c->n,
            c->dtype, c->d_work, d_size, c->h_work, h_size, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
potrs_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    cumo_cuda_cusolver_check_status(cusolverDnXpotrs(
            c->ctx.handle, c->ctx.params, c->uplo, c->n, c->nrhs, c->dtype, c->a, c->n,
            c->dtype, c->b, c->n, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
potri_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cusolverDnHandle_t h = c->ctx.handle;
    int n = (int)c->n;
    int lwork = 0;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    switch (c->dtype) {
#define POTRI(prefix, type)                                                                        \
        check_call(cusolverDn##prefix##potri_bufferSize(h, c->uplo, n, (type*)c->a, n, &lwork));  \
        c->d_work = cumo_cuda_runtime_malloc(sizeof(type) * (lwork > 0 ? lwork : 1));             \
        cumo_cuda_cusolver_check_status(cusolverDn##prefix##potri(                                  \
                h, c->uplo, n, (type*)c->a, n, (type*)c->d_work, lwork, c->d_info));                \
        break
    case CUDA_R_32F: POTRI(S, float);
    case CUDA_R_64F: POTRI(D, double);
    case CUDA_C_32F: POTRI(C, cuComplex);
    default: POTRI(Z, cuDoubleComplex);
#undef POTRI
    }
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
syevd_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cudaDataType wtype = (c->dtype == CUDA_R_32F || c->dtype == CUDA_C_32F) ? CUDA_R_32F : CUDA_R_64F;
    float zero_f = 0;
    double zero_d = 0;
    void *zero = wtype == CUDA_R_32F ? (void*)&zero_f : (void*)&zero_d;
    size_t d_size = 0;
    size_t h_size = 0;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    if (c->range) {
        check_call(cusolverDnXsyevdx_bufferSize(
                c->ctx.handle, c->ctx.params, c->jobz, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, c->n,
                c->dtype, c->a, c->n, zero, zero, c->il, c->iu, &c->meig, wtype, c->w, c->dtype,
                &d_size, &h_size));
    } else {
        check_call(cusolverDnXsyevd_bufferSize(
                c->ctx.handle, c->ctx.params, c->jobz, CUBLAS_FILL_MODE_UPPER, c->n,
                c->dtype, c->a, c->n, wtype, c->w, c->dtype, &d_size, &h_size));
    }
    if (d_size > 0) c->d_work = cumo_cuda_runtime_malloc(d_size);
    if (h_size > 0) c->h_work = ruby_xmalloc(h_size);
    if (c->range) {
        cumo_cuda_cusolver_check_status(cusolverDnXsyevdx(
                c->ctx.handle, c->ctx.params, c->jobz, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, c->n,
                c->dtype, c->a, c->n, zero, zero, c->il, c->iu, &c->meig, wtype, c->w, c->dtype,
                c->d_work, d_size, c->h_work, h_size, c->d_info));
    } else {
        cumo_cuda_cusolver_check_status(cusolverDnXsyevd(
                c->ctx.handle, c->ctx.params, c->jobz, CUBLAS_FILL_MODE_UPPER, c->n,
                c->dtype, c->a, c->n, wtype, c->w, c->dtype, c->d_work, d_size, c->h_work, h_size, c->d_info));
        c->meig = c->n;
    }
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
sygvd_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cusolverDnHandle_t h = c->ctx.handle;
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    int n = (int)c->n;
    int il = (int)c->il;
    int iu = (int)c->iu;
    int meig = n;
    int lwork = 0;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    switch (c->dtype) {
#define SYGVD(prefix, type, wtype)                                                                          \
        if (c->range) {                                                                                     \
            check_call(cusolverDn##prefix##dx_bufferSize(                                                   \
                    h, itype, c->jobz, CUSOLVER_EIG_RANGE_I, uplo, n, (type*)c->a, n, (type*)c->b, n,        \
                    0, 0, il, iu, &meig, (wtype*)c->w, &lwork));                                              \
            c->d_work = cumo_cuda_runtime_malloc(sizeof(type) * (lwork > 0 ? lwork : 1));                     \
            cumo_cuda_cusolver_check_status(cusolverDn##prefix##dx(                                          \
                    h, itype, c->jobz, CUSOLVER_EIG_RANGE_I, uplo, n, (type*)c->a, n, (type*)c->b, n,        \
                    0, 0, il, iu, &meig, (wtype*)c->w, (type*)c->d_work, lwork, c->d_info));                  \
        } else {                                                                                            \
            check_call(cusolverDn##prefix##d_bufferSize(                                                    \
                    h, itype, c->jobz, uplo, n, (type*)c->a, n, (type*)c->b, n, (wtype*)c->w, &lwork));       \
            c->d_work = cumo_cuda_runtime_malloc(sizeof(type) * (lwork > 0 ? lwork : 1));                     \
            cumo_cuda_cusolver_check_status(cusolverDn##prefix##d(                                           \
                    h, itype, c->jobz, uplo, n, (type*)c->a, n, (type*)c->b, n, (wtype*)c->w,                  \
                    (type*)c->d_work, lwork, c->d_info));                                                     \
        }                                                                                                   \
        break
    case CUDA_R_32F: SYGVD(Ssygv, float, float);
    case CUDA_R_64F: SYGVD(Dsygv, double, double);
    case CUDA_C_32F: SYGVD(Chegv, cuComplex, float);
    default: SYGVD(Zhegv, cuDoubleComplex, double);
#undef SYGVD
    }
    c->meig = meig;
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
gesvd_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cudaDataType stype = (c->dtype == CUDA_R_32F || c->dtype == CUDA_C_32F) ? CUDA_R_32F : CUDA_R_64F;
    int64_t ldvt = c->n;
    size_t d_size = 0;
    size_t h_size = 0;

    check_call(cusolverDnXgesvd_bufferSize(
            c->ctx.handle, c->ctx.params, c->job, c->job, c->m, c->n, c->dtype, c->a, c->m,
            stype, c->w, c->dtype, c->u, c->m, c->dtype, c->vt, ldvt, c->dtype, &d_size, &h_size));
    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    if (d_size > 0) c->d_work = cumo_cuda_runtime_malloc(d_size);
    if (h_size > 0) c->h_work = ruby_xmalloc(h_size);
    cumo_cuda_cusolver_check_status(cusolverDnXgesvd(
            c->ctx.handle, c->ctx.params, c->job, c->job, c->m, c->n, c->dtype, c->a, c->m,
            stype, c->w, c->dtype, c->u, c->m, c->dtype, c->vt, ldvt, c->dtype,
            c->d_work, d_size, c->h_work, h_size, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
geqrf_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    size_t d_size = 0;
    size_t h_size = 0;

    check_call(cusolverDnXgeqrf_bufferSize(
            c->ctx.handle, c->ctx.params, c->m, c->n, c->dtype, c->a, c->m, c->dtype, c->tau, c->dtype,
            &d_size, &h_size));
    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    if (d_size > 0) c->d_work = cumo_cuda_runtime_malloc(d_size);
    if (h_size > 0) c->h_work = ruby_xmalloc(h_size);
    cumo_cuda_cusolver_check_status(cusolverDnXgeqrf(
            c->ctx.handle, c->ctx.params, c->m, c->n, c->dtype, c->a, c->m, c->dtype, c->tau, c->dtype,
            c->d_work, d_size, c->h_work, h_size, c->d_info));
    c->info = read_info(c->d_info);
    return Qnil;
}

static VALUE
orgqr_body(VALUE arg)
{
    cusolver_call_t *c = (cusolver_call_t*)arg;
    cusolverDnHandle_t h = c->ctx.handle;
    int m = (int)c->m;
    int n = (int)c->n;
    int k = (int)c->k;
    int lwork = 0;

    c->d_info = (int*)cumo_cuda_runtime_malloc(sizeof(int));
    switch (c->dtype) {
#define ORGQR(fn, type)                                                                             \
        check_call(cusolverDn##fn##_bufferSize(h, m, n, k, (type*)c->a, m, (type*)c->tau, &lwork));  \
        c->d_work = cumo_cuda_runtime_malloc(sizeof(type) * (lwork > 0 ? lwork : 1));                 \
        cumo_cuda_cusolver_check_status(cusolverDn##fn(                                              \
                h, m, n, k, (type*)c->a, m, (type*)c->tau, (type*)c->d_work, lwork, c->d_info));      \
        break
    case CUDA_R_32F: ORGQR(Sorgqr, float);
    case CUDA_R_64F: ORGQR(Dorgqr, double);
    case CUDA_C_32F: ORGQR(Cungqr, cuComplex);
    default: ORGQR(Zungqr, cuDoubleComplex);
#undef ORGQR
    }
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
static int64_t check_square_matrix(VALUE a, const char *name);

static VALUE
rb_cusolver_getrs(VALUE self, VALUE lu, VALUE ipiv, VALUE b, VALUE trans)
{
    cusolver_call_t c = {0};
    const char *t = StringValueCStr(trans);
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
    c.n = check_square_matrix(lu, "lu");
    nipiv = check_contiguous_array(ipiv, cumo_cInt64, 1, "ipiv");
    c.dtype = cusolver_dtype(lu);
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

static char
uplo_char(VALUE uplo)
{
    char c = NUM2CHR(uplo);
    if (c != 'U' && c != 'L') {
        rb_raise(rb_eArgError, "uplo must be 'U' or 'L'");
    }
    return c;
}

static cublasFillMode_t
parse_uplo(VALUE uplo)
{
    return uplo_char(uplo) == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

/*
  Reads uplo as LAPACK does, by NUM2CHR.

  @param uplo [String, Integer]
  @return [String] "U" or "L"
 */
static VALUE
rb_cusolver_uplo(VALUE self, VALUE uplo)
{
    char c = uplo_char(uplo);
    return rb_str_new(&c, 1);
}

static int64_t
check_square_matrix(VALUE a, const char *name)
{
    cumo_narray_t *na = check_contiguous_array(a, Qnil, 2, name);
    if (CUMO_NA_SHAPE(na)[0] != CUMO_NA_SHAPE(na)[1]) {
        rb_raise(cumo_na_eShapeError, "%s must be square", name);
    }
    return (int64_t)CUMO_NA_SHAPE(na)[0];
}

/*
  Factorizes a Hermitian positive definite matrix in place as A = U^H U or
  A = L L^H with cusolverDnXpotrf.

  @param a [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, n]: the column-major matrix, whose uplo
    triangle is overwritten with the factor
  @param uplo [String] "U" or "L"
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_potrf(VALUE self, VALUE a, VALUE uplo)
{
    cusolver_call_t c = {0};

    c.uplo = parse_uplo(uplo);
    c.n = check_square_matrix(a, "a");
    c.dtype = cusolver_dtype(a);
    if (c.n == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.ctx = cusolver_context();
    rb_ensure(potrf_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

/*
  Solves A X = B in place with cusolverDnXpotrs, from the factor potrf
  answered.

  @param a [Cumo::NArray] contiguous, of shape [n, n]: the column-major
    factor in its uplo triangle
  @param b [Cumo::NArray] of the class of a, contiguous, of shape [n] or
    [nrhs, n]: the column-major right-hand sides, overwritten with X
  @param uplo [String] "U" or "L"
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_potrs(VALUE self, VALUE a, VALUE b, VALUE uplo)
{
    cusolver_call_t c = {0};
    cumo_narray_t *nb;

    c.uplo = parse_uplo(uplo);
    c.n = check_square_matrix(a, "a");
    c.dtype = cusolver_dtype(a);
    if (!rb_obj_is_kind_of(b, cumo_cNArray)) {
        rb_raise(rb_eTypeError, "b must be Cumo::NArray");
    }
    CumoGetNArray(b, nb);
    nb = check_contiguous_array(b, rb_obj_class(a), CUMO_NA_NDIM(nb) == 1 ? 1 : 2, "b");
    if ((int64_t)CUMO_NA_SHAPE(nb)[CUMO_NA_NDIM(nb) - 1] != c.n) {
        rb_raise(cumo_na_eShapeError, "b must have %"PRId64" rows", c.n);
    }
    c.nrhs = CUMO_NA_NDIM(nb) == 1 ? 1 : (int64_t)CUMO_NA_SHAPE(nb)[0];
    if (c.n == 0 || c.nrhs == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read(a);
    c.b = cumo_na_get_offset_pointer_for_read_write(b);
    c.ctx = cusolver_context();
    rb_ensure(potrs_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

/*
  Computes the inverse of A in place with cusolverDn<t>potri, from the
  factor potrf answered.

  @param a [Cumo::NArray] contiguous, of shape [n, n]: the column-major
    factor in its uplo triangle, which is overwritten with that triangle of
    the inverse
  @param uplo [String] "U" or "L"
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_potri(VALUE self, VALUE a, VALUE uplo)
{
    cusolver_call_t c = {0};

    c.uplo = parse_uplo(uplo);
    c.n = check_square_matrix(a, "a");
    c.dtype = cusolver_dtype(a);
    if (c.n > INT_MAX) {
        rb_raise(rb_eArgError, "a is too large for potri");
    }
    if (c.n == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.ctx = cusolver_context();
    rb_ensure(potri_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

static VALUE
eigen_real_class(cudaDataType dtype)
{
    return (dtype == CUDA_R_32F || dtype == CUDA_C_32F) ? cumo_cSFloat : cumo_cDFloat;
}

static void
parse_eigen_args(cusolver_call_t *c, VALUE a, VALUE w, VALUE vectors, VALUE il, VALUE iu)
{
    cumo_narray_t *nw;

    c->n = check_square_matrix(a, "a");
    c->dtype = cusolver_dtype(a);
    nw = check_contiguous_array(w, eigen_real_class(c->dtype), 1, "w");
    if ((int64_t)CUMO_NA_SHAPE(nw)[0] != c->n) {
        rb_raise(cumo_na_eShapeError, "w must have %"PRId64" elements", c->n);
    }
    c->jobz = RTEST(vectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    c->range = !NIL_P(il);
    if (c->range) {
        c->il = NUM2LL(il);
        c->iu = NUM2LL(iu);
        if (c->il < 1 || c->iu < c->il || c->iu > c->n) {
            rb_raise(rb_eArgError, "il and iu must satisfy 1 <= il <= iu <= n");
        }
    }
}

/*
  Computes the eigenvalues, and the eigenvectors with vectors, of a
  Hermitian matrix in place with cusolverDnXsyevd, or with cusolverDnXsyevdx
  for the il-th to the iu-th eigenvalue. The upper triangle is read.

  @param a [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, n]: the column-major matrix, overwritten with
    the eigenvectors in its first columns
  @param w [Cumo::SFloat, Cumo::DFloat] contiguous, of length n, filled
    with the eigenvalues in ascending order
  @param vectors [Boolean]
  @param il [Integer, nil] 1-based
  @param iu [Integer, nil] 1-based
  @return [Array] the number of eigenvalues found, and the info cuSOLVER
    reports
 */
static VALUE
rb_cusolver_syevd(VALUE self, VALUE a, VALUE w, VALUE vectors, VALUE il, VALUE iu)
{
    cusolver_call_t c = {0};

    parse_eigen_args(&c, a, w, vectors, il, iu);
    if (c.n == 0) {
        return rb_assoc_new(INT2FIX(0), INT2FIX(0));
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.w = cumo_na_get_offset_pointer_for_write(w);
    c.ctx = cusolver_context();
    rb_ensure(syevd_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return rb_assoc_new(LL2NUM(c.meig), INT2NUM(c.info));
}

/*
  Computes the eigenvalues, and the eigenvectors with vectors, of
  A x = lambda B x for Hermitian A and Hermitian positive definite B in place
  with cusolverDn<t>sygvd or hegvd, or with sygvdx or hegvdx for the il-th
  to the iu-th eigenvalue. The upper triangles are read.

  @param a [Cumo::NArray] contiguous, of shape [n, n]: the column-major A,
    overwritten with the eigenvectors in its first columns
  @param b [Cumo::NArray] of the class and shape of a: the column-major B,
    overwritten
  @param w [Cumo::SFloat, Cumo::DFloat] contiguous, of length n
  @param vectors [Boolean]
  @param il [Integer, nil] 1-based
  @param iu [Integer, nil] 1-based
  @return [Array] the number of eigenvalues found, and the info cuSOLVER
    reports
 */
static VALUE
rb_cusolver_sygvd(VALUE self, VALUE a, VALUE b, VALUE w, VALUE vectors, VALUE il, VALUE iu)
{
    cusolver_call_t c = {0};

    parse_eigen_args(&c, a, w, vectors, il, iu);
    check_contiguous_array(b, rb_obj_class(a), 2, "b");
    if (check_square_matrix(b, "b") != c.n) {
        rb_raise(cumo_na_eShapeError, "b must have the shape of a");
    }
    if (c.n > INT_MAX) {
        rb_raise(rb_eArgError, "a is too large for sygvd");
    }
    if (c.n == 0) {
        return rb_assoc_new(INT2FIX(0), INT2FIX(0));
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.b = cumo_na_get_offset_pointer_for_read_write(b);
    c.w = cumo_na_get_offset_pointer_for_write(w);
    c.ctx = cusolver_context();
    rb_ensure(sygvd_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return rb_assoc_new(LL2NUM(c.meig), INT2NUM(c.info));
}

static void
check_shape(VALUE x, VALUE klass, int64_t rows, int64_t cols, const char *name)
{
    cumo_narray_t *nx = check_contiguous_array(x, klass, 2, name);
    if ((int64_t)CUMO_NA_SHAPE(nx)[0] != rows || (int64_t)CUMO_NA_SHAPE(nx)[1] != cols) {
        rb_raise(cumo_na_eShapeError, "%s must be of shape [%"PRId64", %"PRId64"]", name, rows, cols);
    }
}

/*
  Computes the singular value decomposition A = U S V^H in place with
  cusolverDnXgesvd, for m >= n.

  @param a [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, m]: the column-major m by n matrix, overwritten
  @param s [Cumo::SFloat, Cumo::DFloat] contiguous, of length n, filled with
    the singular values in descending order
  @param u [Cumo::NArray, nil] of the class of a: the column-major U, of
    shape [m, m] for job "A" and [n, m] for "S", or nil for "N"
  @param vt [Cumo::NArray, nil] of the class of a and of shape [n, n]: the
    column-major V^H, or nil for "N"
  @param job [String] "A", "S" or "N", for U and V^H alike
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_gesvd(VALUE self, VALUE a, VALUE s, VALUE u, VALUE vt, VALUE job)
{
    cusolver_call_t c = {0};
    cumo_narray_t *na;
    cumo_narray_t *ns;
    const char *j = StringValueCStr(job);

    if (strcmp(j, "A") != 0 && strcmp(j, "S") != 0 && strcmp(j, "N") != 0) {
        rb_raise(rb_eArgError, "job must be \"A\", \"S\" or \"N\"");
    }
    c.job = j[0];
    na = check_contiguous_array(a, Qnil, 2, "a");
    c.dtype = cusolver_dtype(a);
    c.m = (int64_t)CUMO_NA_SHAPE(na)[1];
    c.n = (int64_t)CUMO_NA_SHAPE(na)[0];
    if (c.m < c.n) {
        rb_raise(cumo_na_eShapeError, "a must have at least as many rows as columns");
    }
    ns = check_contiguous_array(s, eigen_real_class(c.dtype), 1, "s");
    if ((int64_t)CUMO_NA_SHAPE(ns)[0] != c.n) {
        rb_raise(cumo_na_eShapeError, "s must have %"PRId64" elements", c.n);
    }
    if (c.job == 'N') {
        if (!NIL_P(u) || !NIL_P(vt)) {
            rb_raise(rb_eArgError, "u and vt must be nil for job \"N\"");
        }
    } else {
        check_shape(u, rb_obj_class(a), c.job == 'A' ? c.m : c.n, c.m, "u");
        check_shape(vt, rb_obj_class(a), c.n, c.n, "vt");
    }
    if (c.n == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.w = cumo_na_get_offset_pointer_for_write(s);
    if (c.job != 'N') {
        c.u = cumo_na_get_offset_pointer_for_write(u);
        c.vt = cumo_na_get_offset_pointer_for_write(vt);
    }
    c.ctx = cusolver_context();
    rb_ensure(gesvd_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

static void
check_tau(VALUE tau, VALUE a, int64_t k)
{
    cumo_narray_t *nt = check_contiguous_array(tau, rb_obj_class(a), 1, "tau");
    if ((int64_t)CUMO_NA_SHAPE(nt)[0] != k) {
        rb_raise(cumo_na_eShapeError, "tau must have %"PRId64" elements", k);
    }
}

/*
  Computes the QR factorization A = Q R in place with cusolverDnXgeqrf, as
  LAPACK's geqrf does: R in the upper triangle and the Householder vectors
  of Q below it.

  @param a [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]
    contiguous, of shape [n, m]: the column-major m by n matrix, overwritten
  @param tau [Cumo::NArray] of the class of a, contiguous, of length
    min(m, n): filled with the scalar factors of the reflectors
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_geqrf(VALUE self, VALUE a, VALUE tau)
{
    cusolver_call_t c = {0};
    cumo_narray_t *na = check_contiguous_array(a, Qnil, 2, "a");

    c.dtype = cusolver_dtype(a);
    c.m = (int64_t)CUMO_NA_SHAPE(na)[1];
    c.n = (int64_t)CUMO_NA_SHAPE(na)[0];
    check_tau(tau, a, c.m < c.n ? c.m : c.n);
    if (c.m == 0 || c.n == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.tau = cumo_na_get_offset_pointer_for_write(tau);
    c.ctx = cusolver_context();
    rb_ensure(geqrf_body, (VALUE)&c, call_ensure, (VALUE)&c);
    return INT2NUM(c.info);
}

/*
  Forms the m by n matrix Q with orthonormal columns from the first k
  reflectors geqrf answered, in place with cusolverDn<t>orgqr or ungqr.

  @param a [Cumo::NArray] contiguous, of shape [n, m] with m >= n: the
    column-major reflectors, overwritten with Q
  @param tau [Cumo::NArray] of the class of a, of length k
  @return [Integer] the info cuSOLVER reports
 */
static VALUE
rb_cusolver_orgqr(VALUE self, VALUE a, VALUE tau)
{
    cusolver_call_t c = {0};
    cumo_narray_t *na = check_contiguous_array(a, Qnil, 2, "a");
    cumo_narray_t *nt;

    c.dtype = cusolver_dtype(a);
    c.m = (int64_t)CUMO_NA_SHAPE(na)[1];
    c.n = (int64_t)CUMO_NA_SHAPE(na)[0];
    nt = check_contiguous_array(tau, rb_obj_class(a), 1, "tau");
    c.k = (int64_t)CUMO_NA_SHAPE(nt)[0];
    if (c.m < c.n || c.n < c.k) {
        rb_raise(cumo_na_eShapeError, "orgqr needs m >= n >= k");
    }
    if (c.m > INT_MAX) {
        rb_raise(rb_eArgError, "a is too large for orgqr");
    }
    if (c.n == 0) {
        return INT2FIX(0);
    }
    c.a = cumo_na_get_offset_pointer_for_read_write(a);
    c.tau = cumo_na_get_offset_pointer_for_read(tau);
    c.ctx = cusolver_context();
    rb_ensure(orgqr_body, (VALUE)&c, call_ensure, (VALUE)&c);
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
    rb_define_singleton_method(mCusolver, "potrf", rb_cusolver_potrf, 2);
    rb_define_singleton_method(mCusolver, "potrs", rb_cusolver_potrs, 3);
    rb_define_singleton_method(mCusolver, "potri", rb_cusolver_potri, 2);
    rb_define_singleton_method(mCusolver, "uplo", rb_cusolver_uplo, 1);
    rb_define_singleton_method(mCusolver, "syevd", rb_cusolver_syevd, 5);
    rb_define_singleton_method(mCusolver, "sygvd", rb_cusolver_sygvd, 6);
    rb_define_singleton_method(mCusolver, "gesvd", rb_cusolver_gesvd, 5);
    rb_define_singleton_method(mCusolver, "geqrf", rb_cusolver_geqrf, 2);
    rb_define_singleton_method(mCusolver, "orgqr", rb_cusolver_orgqr, 2);
    rb_funcall(mCusolver, rb_intern("private_class_method"), 11,
               ID2SYM(rb_intern("geqrf")), ID2SYM(rb_intern("orgqr")), ID2SYM(rb_intern("gesvd")),
               ID2SYM(rb_intern("syevd")), ID2SYM(rb_intern("sygvd")), ID2SYM(rb_intern("uplo")),
               ID2SYM(rb_intern("getrf")), ID2SYM(rb_intern("getrs")),
               ID2SYM(rb_intern("potrf")), ID2SYM(rb_intern("potrs")), ID2SYM(rb_intern("potri")));
#endif // CUSOLVER_FOUND
}
