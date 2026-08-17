#include <ruby.h>
#include <cuda_runtime.h>
#include "memory_pool_impl.hpp"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

static cumo::internal::MemoryPool pool{};
static bool memory_pool_enabled;

VALUE cumo_cuda_eOutOfMemoryError;

// How a call into the pool ended, so that the raise can wait until the handler
// has been left. rb_raise longjmps, and a longjmp out of an active handler
// skips __cxa_end_catch: the exception object is never destroyed and leaks for
// the life of the process.
enum pool_outcome {
    POOL_OK,
    POOL_CUDA_ERROR,
    POOL_OUT_OF_MEMORY,
    POOL_UNKNOWN_ERROR,
};

struct pool_error {
    cudaError_t status;
    char message[256];
};

static enum pool_outcome
pool_malloc(size_t size, char **ptr, struct pool_error *err)
{
    try {
        // TODO(sonots): Get current CUDA stream and pass it
        *ptr = reinterpret_cast<char*>(pool.Malloc(size));
        return POOL_OK;
    } catch (const cumo::internal::CUDARuntimeError& e) {
        err->status = e.status();
        return POOL_CUDA_ERROR;
    } catch (const cumo::internal::OutOfMemoryError& e) {
        // A std::string here would in turn be leaked by the raise.
        std::snprintf(err->message, sizeof(err->message), "%s", e.what());
        return POOL_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        // Nothing may reach the C caller: it has no handler, so an escaping
        // exception is std::terminate.
        std::snprintf(err->message, sizeof(err->message), "%s", e.what());
        return POOL_UNKNOWN_ERROR;
    } catch (...) {
        std::snprintf(err->message, sizeof(err->message), "unknown C++ exception");
        return POOL_UNKNOWN_ERROR;
    }
}

char*
cumo_cuda_runtime_malloc(size_t size)
{
    char *ptr = 0;
    struct pool_error err;
    enum pool_outcome outcome;

    if (!memory_pool_enabled) {
        void *raw = 0;
        cumo_cuda_runtime_check_status(cudaMallocManaged(&raw, size, cudaMemAttachGlobal));
        return reinterpret_cast<char*>(raw);
    }

    outcome = pool_malloc(size, &ptr, &err);
    if (outcome == POOL_OUT_OF_MEMORY) {
        // retry after GC
        rb_funcall(rb_define_module("GC"), rb_intern("start"), 0);
        outcome = pool_malloc(size, &ptr, &err);
    }

    switch (outcome) {
    case POOL_OK:
        return ptr;
    case POOL_CUDA_ERROR:
        cumo_cuda_runtime_check_status(err.status);
        break;
    case POOL_OUT_OF_MEMORY:
        rb_raise(cumo_cuda_eOutOfMemoryError, "%s", err.message);
    case POOL_UNKNOWN_ERROR:
        rb_raise(rb_eRuntimeError, "%s", err.message);
    }
    return 0; // should not reach here
}

static cudaError_t
runtime_free(char *ptr)
{
    // Always offer the pointer to the pool first, whatever memory_pool_enabled
    // says now: MemoryPool.enable/disable is public, so the state can differ
    // from what it was at allocation time. Handing a pooled chunk to cudaFree
    // releases memory the pool still hands out, and fails outright for a chunk
    // which is not at the head of its buffer.
    try {
        if (pool.Free(reinterpret_cast<intptr_t>(ptr))) {
            return cudaSuccess;
        }
    } catch (const cumo::internal::CUDARuntimeError& e) {
        return e.status();
    } catch (...) {
        return cudaErrorUnknown;
    }
    // No pool owns it, so it came straight from cudaMallocManaged.
    return cudaFree((void*)ptr);
}

void
cumo_cuda_runtime_free(char *ptr)
{
    cumo_cuda_runtime_check_status(runtime_free(ptr));
}

void
cumo_cuda_runtime_free_no_raise(char *ptr)
{
    cudaError_t status = runtime_free(ptr);
    // rb_raise from a GC free hook longjmps out of the sweep, so report the
    // failure the way ~Memory does. cudaFree only fails once the context is
    // unusable, and the next runtime call reports the same status anyway.
    if (status == cudaSuccess || status == cudaErrorCudartUnloading) {
        return;
    }
    std::fprintf(stderr, "cumo: failed to free device memory: %s (error=%d)\n",
                 cudaGetErrorString(status), static_cast<int>(status));
}

/*
  Enable memory pool.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_memory_pool_enable(VALUE self)
{
    VALUE ret = (memory_pool_enabled ? Qtrue : Qfalse);
    memory_pool_enabled = true;
    return ret;
}

/*
  Disable memory pool.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_memory_pool_disable(VALUE self)
{
    VALUE ret = (memory_pool_enabled ? Qtrue : Qfalse);
    memory_pool_enabled = false;
    return ret;
}

/*
  Returns whether memory pool is enabled or not.

  @return [Boolean] Returns the state (true if enabled)
 */
static VALUE
rb_memory_pool_enabled_p(VALUE self)
{
    return (memory_pool_enabled ? Qtrue : Qfalse);
}

/*
  Free all **non-split** chunks in all arenas.
 */
static VALUE
rb_memory_pool_free_all_blocks(int argc, VALUE* argv, VALUE self)
{
    // TODO(sonots): FIX if we create a Stream object
    cudaStream_t stream_ptr = (argc < 1) ? 0 : (cudaStream_t)NUM2SIZET(argv[0]);
    cudaError_t status = cudaSuccess;

    try {
        if (argc < 1) {
            pool.FreeAllBlocks();
        } else {
            pool.FreeAllBlocks(stream_ptr);
        }
    } catch (const cumo::internal::CUDARuntimeError& e) {
        status = e.status();
    } catch (...) {
        status = cudaErrorUnknown;
    }
    cumo_cuda_runtime_check_status(status);
    return Qnil;
}

/*
  Count the total number of free blocks.

  @return [Integer] The total number of free blocks.
 */
static VALUE
rb_memory_pool_n_free_blocks(VALUE self)
{
    return SIZET2NUM(pool.GetNumFreeBlocks());
}

/*
  Get the total number of bytes used.

  @return [Integer] The total number of bytes used.
 */
static VALUE
rb_memory_pool_used_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetUsedBytes());
}

/*
  Get the total number of bytes acquired but not used in the pool.

  @return [Integer] The total number of bytes acquired but not used in the pool.
 */
static VALUE
rb_memory_pool_free_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetFreeBytes());
}

/*
  Get the total number of bytes acquired in the pool.

  @return [Integer] The total number of bytes acquired in the pool.
 */
static VALUE
rb_memory_pool_total_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetTotalBytes());
}

void
Init_cumo_cuda_memory_pool()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    VALUE mMemoryPool = rb_define_module_under(mCUDA, "MemoryPool");
    cumo_cuda_eOutOfMemoryError = rb_define_class_under(mCUDA, "OutOfMemoryError", rb_eStandardError);
    
    rb_define_singleton_method(mMemoryPool, "enable", RUBY_METHOD_FUNC(rb_memory_pool_enable), 0);
    rb_define_singleton_method(mMemoryPool, "disable", RUBY_METHOD_FUNC(rb_memory_pool_disable), 0);
    rb_define_singleton_method(mMemoryPool, "enabled?", RUBY_METHOD_FUNC(rb_memory_pool_enabled_p), 0);
    rb_define_singleton_method(mMemoryPool, "free_all_blocks", RUBY_METHOD_FUNC(rb_memory_pool_free_all_blocks), -1);
    rb_define_singleton_method(mMemoryPool, "n_free_blocks", RUBY_METHOD_FUNC(rb_memory_pool_n_free_blocks), 0);
    rb_define_singleton_method(mMemoryPool, "used_bytes", RUBY_METHOD_FUNC(rb_memory_pool_used_bytes), 0);
    rb_define_singleton_method(mMemoryPool, "free_bytes", RUBY_METHOD_FUNC(rb_memory_pool_free_bytes), 0);
    rb_define_singleton_method(mMemoryPool, "total_bytes", RUBY_METHOD_FUNC(rb_memory_pool_total_bytes), 0);

    // default is true
    const char* env = std::getenv("CUMO_MEMORY_POOL");
    memory_pool_enabled = env == nullptr || (std::string(env) != "OFF" && std::string(env) != "0" && std::string(env) != "NO");
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
