#include "cumo/cuda/cufft.h"

#include <ruby.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include "cumo/narray.h"
#include "cumo/intern.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

VALUE cumo_cuda_eCufftError;
VALUE cumo_cuda_mCufft;
#define eCufftError cumo_cuda_eCufftError
#define mCufft cumo_cuda_mCufft

#ifdef CUFFT_FOUND

#define CUFFT_FUNCTIONS(X) \
    X(cufftGetVersion) X(cufftCreate) X(cufftSetAutoAllocation) X(cufftMakePlanMany64) \
    X(cufftDestroy) X(cufftSetWorkArea) X(cufftSetStream) X(cufftExecC2C) X(cufftExecZ2Z) \
    X(cufftExecR2C) X(cufftExecD2Z) X(cufftExecC2R) X(cufftExecZ2D)

static struct {
#define CUFFT_POINTER(f) __typeof__(&f) f;
    CUFFT_FUNCTIONS(CUFFT_POINTER)
#undef CUFFT_POINTER
} cufft_functions;
static pthread_once_t cufft_once = PTHREAD_ONCE_INIT;
static char cufft_error[256];

static const char*
dl_error(void)
{
    const char *e = dlerror();
    return e ? e : "unknown error";
}

static void
load_cufft(void)
{
    char soname[32];
    void *lib;

    snprintf(soname, sizeof(soname), "libcufft.so.%d", CUFFT_VER_MAJOR);
    lib = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) {
        snprintf(cufft_error, sizeof(cufft_error), "cannot load %s: %s", soname, dl_error());
        return;
    }
#define CUFFT_RESOLVE(f)                                                                      \
    cufft_functions.f = (__typeof__(&f))dlsym(lib, #f);                                        \
    if (cufft_functions.f == NULL) {                                                           \
        snprintf(cufft_error, sizeof(cufft_error), "%s has no %s: %s", soname, #f, dl_error()); \
        dlclose(lib);                                                                          \
        return;                                                                                \
    }
    CUFFT_FUNCTIONS(CUFFT_RESOLVE)
#undef CUFFT_RESOLVE
}

static const __typeof__(cufft_functions)*
cufft_api(void)
{
    pthread_once(&cufft_once, load_cufft);
    if (cufft_error[0] != '\0') {
        rb_raise(cumo_cuda_eCufftError, "%s", cufft_error);
    }
    return &cufft_functions;
}

static const char*
get_cufft_error_msg(cufftResult error)
{
    switch (error) {
#define RETURN_MSG(msg) \
    case msg:           \
        return #msg

        RETURN_MSG(CUFFT_SUCCESS);
        RETURN_MSG(CUFFT_INVALID_PLAN);
        RETURN_MSG(CUFFT_ALLOC_FAILED);
        RETURN_MSG(CUFFT_INVALID_TYPE);
        RETURN_MSG(CUFFT_INVALID_VALUE);
        RETURN_MSG(CUFFT_INTERNAL_ERROR);
        RETURN_MSG(CUFFT_EXEC_FAILED);
        RETURN_MSG(CUFFT_SETUP_FAILED);
        RETURN_MSG(CUFFT_INVALID_SIZE);
        RETURN_MSG(CUFFT_UNALIGNED_DATA);
        RETURN_MSG(CUFFT_INVALID_DEVICE);
        RETURN_MSG(CUFFT_NO_WORKSPACE);
        RETURN_MSG(CUFFT_NOT_IMPLEMENTED);
        RETURN_MSG(CUFFT_NOT_SUPPORTED);

#undef RETURN_MSG
    default:
        return "CUFFT_UNKNOWN";
    }
}

NORETURN(static void raise_cufft_error(cufftResult status));

static void
raise_cufft_error(cufftResult status)
{
    rb_raise(cumo_cuda_eCufftError, "%s (error=%d)", get_cufft_error_msg(status), status);
}

void
cumo_cuda_cufft_check_status(cufftResult status)
{
    cumo_cuda_runtime_note_device_write();
    if (status != CUFFT_SUCCESS) {
        raise_cufft_error(status);
    }
}

static void
check_call(cufftResult status)
{
    if (status != CUFFT_SUCCESS) {
        raise_cufft_error(status);
    }
}

/*
  Returns the version of the cuFFT library cumo is linked with.

  @return [Integer] the version cufftGetVersion reports
 */
static VALUE
rb_cufft_version(VALUE self)
{
    int version;
    check_call(cufft_api()->cufftGetVersion(&version));
    return INT2NUM(version);
}

#define CUFFT_PLAN_CACHE_SIZE 16
#define CUFFT_MAX_RANK 3

typedef struct {
    cufftHandle handle;
    cudaEvent_t done;
    cufftType type;
    int rank;
    long long n[CUFFT_MAX_RANK];
    long long batch;
    size_t work_size;
    unsigned long long used;
} cufft_plan_t;

typedef struct {
    cufft_plan_t plans[CUFFT_PLAN_CACHE_SIZE];
    int count;
    unsigned long long clock;
} cufft_plan_cache_t;

static cumo_cuda_thread_local_t caches;

static void
evict_oldest_plan(cufft_plan_cache_t *cache)
{
    int i, oldest = 0;
    cufft_plan_t old;
    cudaError_t wait;

    for (i = 1; i < cache->count; i++) {
        if (cache->plans[i].used < cache->plans[oldest].used) {
            oldest = i;
        }
    }
    old = cache->plans[oldest];
    cache->plans[oldest] = cache->plans[--cache->count];
    wait = cudaEventSynchronize(old.done);
    cufft_functions.cufftDestroy(old.handle);
    cudaEventDestroy(old.done);
    cumo_cuda_runtime_check_status(wait);
}

static void
cufft_plan_cache_release(void *entry)
{
    cufft_plan_cache_t *cache = (cufft_plan_cache_t*)entry;
    int i;

    for (i = 0; i < cache->count; i++) {
        cudaEventSynchronize(cache->plans[i].done);
        cufft_functions.cufftDestroy(cache->plans[i].handle);
        cudaEventDestroy(cache->plans[i].done);
    }
    cudaGetLastError();
    memset(cache, 0, sizeof(*cache));
}

static cufftResult
make_plan(const __typeof__(cufft_functions) *api, cufft_plan_t *p)
{
    cufftResult status = api->cufftCreate(&p->handle);

    if (status != CUFFT_SUCCESS) {
        return status;
    }
    status = api->cufftSetAutoAllocation(p->handle, 0);
    if (status == CUFFT_SUCCESS) {
        status = api->cufftMakePlanMany64(p->handle, p->rank, p->n, NULL, 1, 0, NULL, 1, 0, p->type, p->batch, &p->work_size);
    }
    if (status != CUFFT_SUCCESS) {
        api->cufftDestroy(p->handle);
    }
    return status;
}

static cufft_plan_t
cufft_plan(cufftType type, int rank, const long long *n, long long batch)
{
    const __typeof__(cufft_functions) *api = cufft_api();
    cufft_plan_cache_t *cache = (cufft_plan_cache_t*)cumo_cuda_thread_local_get(&caches);
    cufft_plan_t p;
    cufftResult status;
    cudaError_t event_status;
    int i;

    for (i = 0; i < cache->count; i++) {
        cufft_plan_t *q = &cache->plans[i];
        if (q->type == type && q->rank == rank && q->batch == batch &&
            memcmp(q->n, n, sizeof(long long) * rank) == 0) {
            q->used = ++cache->clock;
            return *q;
        }
    }

    if (cache->count == CUFFT_PLAN_CACHE_SIZE) {
        evict_oldest_plan(cache);
    }
    memset(&p, 0, sizeof(p));
    p.type = type;
    p.rank = rank;
    memcpy(p.n, n, sizeof(long long) * rank);
    p.batch = batch;
    status = make_plan(api, &p);
    if (status == CUFFT_ALLOC_FAILED || status == CUFFT_INTERNAL_ERROR) {
        cufft_plan_cache_release(cache);
        rb_funcall(rb_define_module("GC"), rb_intern("start"), 0);
        rb_funcall(rb_path2class("Cumo::CUDA::MemoryPool"), rb_intern("free_all_blocks"), 0);
        status = make_plan(api, &p);
    }
    if (status != CUFFT_SUCCESS) {
        raise_cufft_error(status);
    }
    event_status = cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming);
    if (event_status != cudaSuccess) {
        api->cufftDestroy(p.handle);
        cumo_cuda_runtime_check_status(event_status);
    }
    if (cache->count == CUFFT_PLAN_CACHE_SIZE) {
        evict_oldest_plan(cache);
    }
    p.used = ++cache->clock;
    cache->plans[cache->count++] = p;
    return p;
}

typedef enum {
    CUFFT_KIND_C2C,
    CUFFT_KIND_R2C,
    CUFFT_KIND_C2R,
} cufft_kind_t;

typedef struct {
    cufft_plan_t plan;
    int direction;
    void *in;
    void *out;
    size_t in_bytes;
    int copy_in;
    char *in_copy;
    char *work;
} cufft_call_t;

static VALUE
exec_body(VALUE arg)
{
    cufft_call_t *c = (cufft_call_t*)arg;
    const __typeof__(cufft_functions) *api = cufft_api();
    cufftHandle h = c->plan.handle;
    void *in = c->in;
    cufftResult status;

    if (c->copy_in) {
        c->in_copy = cumo_cuda_runtime_malloc(c->in_bytes);
        cumo_cuda_runtime_check_status(
                cudaMemcpyAsync(c->in_copy, c->in, c->in_bytes, cudaMemcpyDeviceToDevice, cumo_cuda_stream()));
        in = c->in_copy;
    }
    if (c->plan.work_size > 0) {
        c->work = cumo_cuda_runtime_malloc(c->plan.work_size);
    }
    check_call(api->cufftSetWorkArea(h, c->work));
    check_call(api->cufftSetStream(h, cumo_cuda_stream()));
    switch (c->plan.type) {
    case CUFFT_C2C:
        status = api->cufftExecC2C(h, (cufftComplex*)in, (cufftComplex*)c->out, c->direction);
        break;
    case CUFFT_Z2Z:
        status = api->cufftExecZ2Z(h, (cufftDoubleComplex*)in, (cufftDoubleComplex*)c->out, c->direction);
        break;
    case CUFFT_R2C:
        status = api->cufftExecR2C(h, (cufftReal*)in, (cufftComplex*)c->out);
        break;
    case CUFFT_D2Z:
        status = api->cufftExecD2Z(h, (cufftDoubleReal*)in, (cufftDoubleComplex*)c->out);
        break;
    case CUFFT_C2R:
        status = api->cufftExecC2R(h, (cufftComplex*)in, (cufftReal*)c->out);
        break;
    case CUFFT_Z2D:
        status = api->cufftExecZ2D(h, (cufftDoubleComplex*)in, (cufftDoubleReal*)c->out);
        break;
    default:
        status = CUFFT_INVALID_TYPE;
    }
    cumo_cuda_cufft_check_status(status);
    cumo_cuda_runtime_check_status(cudaEventRecord(c->plan.done, cumo_cuda_stream()));
    return Qnil;
}

static VALUE
exec_ensure(VALUE arg)
{
    cufft_call_t *c = (cufft_call_t*)arg;
    cumo_cuda_runtime_return_scratch(c->work, 0, NULL);
    cumo_cuda_runtime_return_scratch(c->in_copy, 0, NULL);
    return Qnil;
}

static VALUE
transform(VALUE x, VALUE vrank, cufft_kind_t kind, int direction, VALUE vn, int overwrite)
{
    VALUE in_class = rb_obj_class(x);
    VALUE out_class;
    cumo_narray_t *na;
    cufftType type;
    size_t *shape, *out_shape;
    size_t in_elmsz;
    long long n[CUFFT_MAX_RANK];
    long long batch = 1;
    int single, rank, ndim, i;
    VALUE y;
    cufft_call_t c;

    if (kind == CUFFT_KIND_R2C) {
        if (in_class == cumo_cSFloat) {
            single = 1;
        } else if (in_class == cumo_cDFloat) {
            single = 0;
        } else {
            rb_raise(rb_eTypeError, "invalid data type for cuFFT: %"PRIsVALUE, in_class);
        }
    } else if (in_class == cumo_cSComplex) {
        single = 1;
    } else if (in_class == cumo_cDComplex) {
        single = 0;
    } else {
        rb_raise(rb_eTypeError, "invalid data type for cuFFT: %"PRIsVALUE, in_class);
    }
    switch (kind) {
    case CUFFT_KIND_C2C:
        type = single ? CUFFT_C2C : CUFFT_Z2Z;
        out_class = in_class;
        in_elmsz = single ? sizeof(cufftComplex) : sizeof(cufftDoubleComplex);
        break;
    case CUFFT_KIND_R2C:
        type = single ? CUFFT_R2C : CUFFT_D2Z;
        out_class = single ? cumo_cSComplex : cumo_cDComplex;
        in_elmsz = single ? sizeof(cufftReal) : sizeof(cufftDoubleReal);
        break;
    default:
        type = single ? CUFFT_C2R : CUFFT_Z2D;
        out_class = single ? cumo_cSFloat : cumo_cDFloat;
        in_elmsz = single ? sizeof(cufftComplex) : sizeof(cufftDoubleComplex);
        break;
    }

    rank = NUM2INT(vrank);
    if (rank < 1 || rank > CUFFT_MAX_RANK) {
        rb_raise(rb_eArgError, "rank must be 1 to %d", CUFFT_MAX_RANK);
    }
    CumoGetNArray(x, na);
    ndim = CUMO_NA_NDIM(na);
    if (ndim < rank) {
        rb_raise(cumo_na_eDimensionError, "a %d-dimensional array has no %d axes to transform", ndim, rank);
    }
    if (cumo_na_check_contiguous(x) != Qtrue) {
        rb_raise(rb_eArgError, "array must be contiguous");
    }
    shape = ALLOCA_N(size_t, ndim);
    out_shape = ALLOCA_N(size_t, ndim);
    for (i = 0; i < ndim; i++) {
        shape[i] = out_shape[i] = CUMO_NA_SHAPE(na)[i];
    }
    for (i = 0; i < rank; i++) {
        n[i] = (long long)shape[ndim - rank + i];
    }
    if (kind == CUFFT_KIND_R2C) {
        out_shape[ndim - 1] = shape[ndim - 1] / 2 + 1;
    } else if (kind == CUFFT_KIND_C2R) {
        n[rank - 1] = NUM2LL(vn);
        if (n[rank - 1] < 1 || shape[ndim - 1] != (size_t)(n[rank - 1] / 2 + 1)) {
            rb_raise(rb_eArgError, "a last axis of %"SZF"u does not hold the half spectrum of %lld points",
                     shape[ndim - 1], n[rank - 1]);
        }
        out_shape[ndim - 1] = (size_t)n[rank - 1];
    }
    for (i = 0; i < rank; i++) {
        if (n[i] < 1) {
            rb_raise(rb_eArgError, "Invalid number of FFT data points (%lld) specified.", n[i]);
        }
    }
    for (i = 0; i < ndim - rank; i++) {
        batch *= (long long)shape[i];
    }

    y = cumo_na_new(out_class, ndim, out_shape);
    if (batch == 0) {
        return y;
    }

    memset(&c, 0, sizeof(c));
    c.out = cumo_na_get_pointer_for_write(y);
    c.in = cumo_na_get_offset_pointer_for_read(x);
    CumoGetNArray(x, na);
    if (CUMO_NA_NDIM(na) != ndim || memcmp(CUMO_NA_SHAPE(na), shape, sizeof(size_t) * ndim) != 0 ||
        cumo_na_check_contiguous(x) != Qtrue) {
        rb_raise(rb_eRuntimeError, "NArray changed while taking its pointer");
    }
    CumoGetNArray(y, na);
    if (CUMO_NA_NDIM(na) != ndim || memcmp(CUMO_NA_SHAPE(na), out_shape, sizeof(size_t) * ndim) != 0) {
        rb_raise(rb_eRuntimeError, "NArray changed while taking its pointer");
    }
    c.in_bytes = in_elmsz * (size_t)batch;
    for (i = ndim - rank; i < ndim; i++) {
        c.in_bytes *= shape[i];
    }
    c.copy_in = (kind == CUFFT_KIND_C2R) && !overwrite;
    c.direction = direction;
    c.plan = cufft_plan(type, rank, n, batch);
    rb_ensure(exec_body, (VALUE)&c, exec_ensure, (VALUE)&c);
    RB_GC_GUARD(x);
    return y;
}

static VALUE
rb_cufft_c2c(VALUE self, VALUE x, VALUE rank, VALUE inverse)
{
    return transform(x, rank, CUFFT_KIND_C2C, RTEST(inverse) ? CUFFT_INVERSE : CUFFT_FORWARD, Qnil, 0);
}

static VALUE
rb_cufft_r2c(VALUE self, VALUE x, VALUE rank)
{
    return transform(x, rank, CUFFT_KIND_R2C, CUFFT_FORWARD, Qnil, 0);
}

static VALUE
rb_cufft_c2r(VALUE self, VALUE x, VALUE rank, VALUE n, VALUE overwrite)
{
    return transform(x, rank, CUFFT_KIND_C2R, CUFFT_INVERSE, n, RTEST(overwrite));
}

#endif // CUFFT_FOUND

/*
  Returns availability of cuFFT.

  @return [Boolean] Returns true if cuFFT is available
 */
static VALUE
rb_cufft_available_p(VALUE self)
{
#ifdef CUFFT_FOUND
    return Qtrue;
#else
    return Qfalse;
#endif
}

void
Init_cumo_cuda_cufft(void)
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");

    /*
      Document-module: Cumo::CUDA::Cufft
    */
    mCufft = rb_define_module_under(mCUDA, "Cufft");
    eCufftError = rb_define_class_under(mCUDA, "CufftError", rb_eStandardError);

    rb_define_singleton_method(mCufft, "available?", rb_cufft_available_p, 0);
#ifdef CUFFT_FOUND
    cumo_cuda_thread_local_init(&caches, sizeof(cufft_plan_cache_t));
    cumo_cuda_thread_local_on_exit(&caches, cufft_plan_cache_release);
    rb_define_singleton_method(mCufft, "version", rb_cufft_version, 0);
    rb_define_singleton_method(mCufft, "c2c", rb_cufft_c2c, 3);
    rb_define_singleton_method(mCufft, "r2c", rb_cufft_r2c, 2);
    rb_define_singleton_method(mCufft, "c2r", rb_cufft_c2r, 4);
    rb_funcall(mCufft, rb_intern("private_class_method"), 3,
               ID2SYM(rb_intern("c2c")), ID2SYM(rb_intern("r2c")), ID2SYM(rb_intern("c2r")));
#endif // CUFFT_FOUND
}
