#include <ruby.h>
#include <cuda_runtime.h>
#include "cumo/cuda/memory.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

cumo::internal::MemoryPool pool{};

char*
cumo_cuda_runtime_malloc(size_t size)
{
    try {
        return reinterpret_cast<char*>(pool.Malloc(size));
    } catch (const cumo::internal::CUDARuntimeError& e) {
        cumo_cuda_runtime_check_status(e.status());
    }
    return 0; // should not reach here
}

void
cumo_cuda_runtime_free(char *ptr)
{
    try {
        pool.Free(reinterpret_cast<intptr_t>(ptr));
    } catch (const cumo::internal::CUDARuntimeError& e) {
        cumo_cuda_runtime_check_status(e.status());
    }
}
