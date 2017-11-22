/*
  runtime.c
  Thin wrapper of CUDA runtime
*/

#include <ruby.h>
#include <assert.h>
#include "numo/cuda/runtime.h"

VALUE cumo_cuda_runtime_eCUDARuntimeError;

void
cumo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_runtime_eCUDARuntimeError, "CUDA error: %d %s", status, cudaGetErrorString(status));
    }
}

void
Init_cuda_runtime()
{
    cumo_cuda_runtime_eCUDARuntimeError = rb_define_class_under(numo_cNArray, "CUDARuntimeError", rb_eStandardError);
}
