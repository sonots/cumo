#include <ruby.h>
#include <cuda_runtime.h>
#include "cumo/cuda/memory.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

#include <cstdlib>
#include <string>

cumo::internal::MemoryPool pool{};

// default: false (yet)
bool
cumo_cuda_memory_pool_p()
{
    const static char* env = std::getenv("CUMO_MEMORY_POOL");
    static bool enabled = (env != nullptr && std::string(env) != "OFF" && std::string(env) != "0" && std::string(env) != "NO");
    return enabled;
}

char*
cumo_cuda_runtime_malloc(size_t size)
{
    if (cumo_cuda_memory_pool_p()) {
        try {
            // TODO(sonots): Get current CUDA stream and pass it
            return reinterpret_cast<char*>(pool.Malloc(size));
        } catch (const cumo::internal::CUDARuntimeError& e) {
            cumo_cuda_runtime_check_status(e.status());
        } catch (const cumo::internal::OutOfMemoryError& e) {
            rb_raise(cumo_cuda_eOutOfMemoryError, "%s", e.what());
        }
    } else {
        void *ptr = 0;
        cumo_cuda_runtime_check_status(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
        return reinterpret_cast<char*>(ptr);
    }
    return 0; // should not reach here
}

void
cumo_cuda_runtime_free(char *ptr)
{
    if (cumo_cuda_memory_pool_p()) {
        try {
            // TODO(sonots): Get current CUDA stream and pass it
            pool.Free(reinterpret_cast<intptr_t>(ptr));
        } catch (const cumo::internal::CUDARuntimeError& e) {
            cumo_cuda_runtime_check_status(e.status());
        }
    } else {
        cumo_cuda_runtime_check_status(cudaFree((void*)ptr));
    }
}

void
Init_cumo_cuda_memory()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    cumo_cuda_eOutOfMemoryError = rb_define_class_under(mCUDA, "OutOfMemoryError", rb_eStandardError);
}
