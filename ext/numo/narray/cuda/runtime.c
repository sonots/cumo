#include <ruby.h>
#include <assert.h>
#include "numo/cuda/runtime.h"

VALUE numo_cuda_runtime_eCUDARuntimeError;

void
numo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(numo_cuda_runtime_eCUDARuntimeError, "CUDA error: %d %s", status, cudaGetErrorString(status));
    }
}

void
Init_numo_cuda_runtime()
{
    numo_cuda_runtime_eCUDARuntimeError = rb_define_class_under(numo_cNArray, "CUDARuntimeError", rb_eStandardError);
}
