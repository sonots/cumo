#include "cumo/cuda/cusolver.h"

#include <ruby.h>
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

/*
  Returns the version of the cuSOLVER library cumo is linked with.

  @return [Integer] major * 1000 + minor * 100 + patch
 */
static VALUE
rb_cusolver_version(VALUE self)
{
    int version;
    cusolverStatus_t status = cusolverGetVersion(&version);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        raise_cusolver_error(status);
    }
    return INT2NUM(version);
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
#endif // CUSOLVER_FOUND
}
